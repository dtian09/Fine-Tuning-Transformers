import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
import wandb
import random

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Configuration ---
os.environ["XLA_USE_BF16"] = "1"

DEVICE = xm.xla_device()
BATCH_SIZE = 16
MAX_CAPTION_LEN = 32
PERCENTAGE = 0.1
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "token"

# --- W&B Init ---
wandb.init(
    project="clip-llama-captioning",
    entity="dtian",
    config={
        "batch_size": BATCH_SIZE,
        "max_caption_len": MAX_CAPTION_LEN,
        "percentage_used": PERCENTAGE,
        "tuning_method": "LoRA",
        "embedding_type": "custom nn.Embedding",
        "model": model_id
    }
)

# --- Tokenizer & Model ---
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<sos>'})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '<eos>'})

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    token=HF_TOKEN
)
model.resize_token_embeddings(len(tokenizer))

# --- CLIP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

@torch.no_grad()
def encode_image_batch(images):  # images: [B, C, H, W]
    pil_images = [transforms.ToPILImage()(img) for img in images]
    inputs = clip_processor(images=pil_images, return_tensors="pt", do_rescale=False).to(DEVICE)
    return clip_model.get_image_features(**inputs)

# --- Flickr30k Dataset ---
class Flickr30kDataset(Dataset):
    def __init__(self, split="train", transform=None, max_length=MAX_CAPTION_LEN):
        self.max_length = max_length
        if os.path.isdir("flickr30k_" + split + "_filtered"):
            self.dataset = load_from_disk("flickr30k_" + split + "_filtered")
        else:
            dataset = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)
            dataset = dataset.filter(lambda x: x["split"] == split, keep_in_memory=False)
            self.dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in {"caption", "image"}]
            )
            self.dataset.save_to_disk("flickr30k_" + split + "_filtered")

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item["image"].convert("RGB"))
        caption = str(item["caption"])

        encoding = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length - 1,
            return_tensors="pt",
            add_special_tokens=False
        )

        sos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id

        input_ids = torch.cat([torch.tensor([sos_id]), encoding["input_ids"].squeeze(0)], dim=0)
        labels = torch.cat([encoding["input_ids"].squeeze(0), torch.tensor([eos_id])], dim=0)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {"image": image, "input_ids": input_ids, "labels": labels, "caption": caption}

# --- Data Setup ---
full_dataset = Flickr30kDataset(split="train")
subset_size = int(PERCENTAGE * len(full_dataset))
subset_indices = random.sample(range(len(full_dataset)), subset_size)
dataset = Subset(full_dataset, subset_indices)

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    captions = [item["caption"] for item in batch]
    img_embeds = encode_image_batch(images)
    return input_ids, labels, img_embeds, captions

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
train_loader = pl.MpDeviceLoader(loader, device=DEVICE)

# --- Model Extensions ---
peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config).to(DEVICE)
model.print_trainable_parameters()

custom_caption_embed = nn.Embedding(len(tokenizer), model.config.hidden_size).to(DEVICE)
proj = nn.Linear(clip_model.config.projection_dim, model.config.hidden_size).to(DEVICE, dtype=model.dtype)

# --- Optimizer ---
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(custom_caption_embed.parameters()) + list(proj.parameters()), lr=2e-5
)

# --- Training ---
for epoch in range(2):
    print(f"\nEpoch {epoch+1}")
    for step, (input_ids, labels, img_emb, captions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        img_emb = img_emb.to(DEVICE).to(dtype=proj.weight.dtype)

        img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)
        token_embeds = custom_caption_embed(input_ids).to(dtype=model.dtype)
        inputs_embeds = torch.cat([img_proj, token_embeds], dim=1)

        labels[labels == tokenizer.pad_token_id] = -100
        labels = torch.cat([torch.full((labels.size(0), 1), -100, device=DEVICE, dtype=labels.dtype), labels], dim=1)

        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad()

        if step % 10 == 0:
            wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

# --- Save ---
save_path = "trained_clip_llama"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
torch.save(custom_caption_embed.state_dict(), os.path.join(save_path, "caption_embedding.pt"))
torch.save(proj.state_dict(), os.path.join(save_path, "image_projection.pt"))

artifact = wandb.Artifact("clip-llama-captioning-model", type="model")
artifact.add_dir(save_path)
wandb.log_artifact(artifact)

# --- Inference ---
def generate_caption(image_path):
    img = transforms.Resize((224, 224))(Image.open(image_path).convert("RGB"))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
    img_emb = encode_image_batch(img_tensor).to(DEVICE)
    img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)
    generated_ids = model.generate(inputs_embeds=img_proj, max_length=MAX_CAPTION_LEN)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Test
print(generate_caption("image1.jpg"))
