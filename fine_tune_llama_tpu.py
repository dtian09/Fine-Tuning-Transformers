import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
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
from huggingface_hub import login

# --- Config ---
os.environ["XLA_USE_BF16"] = "1"
torch.set_default_dtype(torch.bfloat16)

DEVICE = xm.xla_device()
BATCH_SIZE = 16
MAX_CAPTION_LEN = 32
PERCENTAGE = 1
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "token"

# --- Authenticate Hugging Face (required for gated model access) ---
login(token=HF_TOKEN)

# --- W&B ---
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

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.add_special_tokens({'bos_token': '<sos>', 'eos_token': '<eos>'})
sos_id = tokenizer.bos_token_id
eos_id = tokenizer.eos_token_id

# --- Load Models ---
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN)
model.resize_token_embeddings(len(tokenizer))
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE, dtype=torch.bfloat16)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# --- Dataset with Cached CLIP Features ---
class Flickr30kCachedDataset(Dataset):
    def __init__(self, split="train", transform=None, max_length=MAX_CAPTION_LEN, cache_dir="clip_cache"):
        self.max_length = max_length
        self.cache_dir = os.path.join(cache_dir, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.isdir("flickr30k_" + split + "_filtered"):
            self.dataset = load_from_disk("flickr30k_" + split + "_filtered")
        else:
            dataset = load_dataset("nlphuji/flickr30k", split="train")
            dataset = dataset.filter(lambda x: x["split"] == split)
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in {"caption", "image"}])
            dataset.save_to_disk("flickr30k_" + split + "_filtered")
            self.dataset = dataset

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def _get_clip_embedding(self, image_tensor, idx):
        cache_path = os.path.join(self.cache_dir, f"{idx}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location="cpu")
        with torch.no_grad():
            pil_image = transforms.ToPILImage()(image_tensor.to(dtype=torch.float32))
            inputs = clip_processor(images=pil_image, return_tensors="pt", do_rescale=False).to(DEVICE)
            embedding = clip_model.get_image_features(**inputs).squeeze(0).cpu()
            torch.save(embedding, cache_path)
        return embedding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_tensor = self.transform(item["image"].convert("RGB"))
        caption = item["caption"]
        img_feat = self._get_clip_embedding(image_tensor, idx)

        enc = tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_length - 1, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([torch.tensor([sos_id]), enc["input_ids"].squeeze(0)], dim=0)[:self.max_length]
        labels = torch.cat([enc["input_ids"].squeeze(0), torch.tensor([eos_id])], dim=0)[:self.max_length]

        return {
            "clip_embedding": img_feat,
            "input_ids": input_ids,
            "labels": labels,
            "caption": caption
        }

# --- Load and Subset Dataset ---
full_dataset = Flickr30kCachedDataset(split="train")
subset_size = int(PERCENTAGE * len(full_dataset))
subset_indices = random.sample(range(len(full_dataset)), subset_size)
dataset = Subset(full_dataset, subset_indices)

# --- Collate Function ---
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    captions = [item["caption"] for item in batch]
    img_feats = torch.stack([item["clip_embedding"] for item in batch])
    return input_ids, labels, img_feats, captions

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
train_loader = pl.MpDeviceLoader(loader, device=DEVICE)

# --- LoRA + Custom Embedding/Projection ---
peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, peft_config).to(DEVICE, dtype=torch.bfloat16)
model.print_trainable_parameters()
custom_caption_embed = nn.Embedding(len(tokenizer), model.config.hidden_size).to(DEVICE, dtype=torch.bfloat16)
proj = nn.Linear(clip_model.config.projection_dim, model.config.hidden_size).to(DEVICE, dtype=torch.bfloat16)

# --- Optimizer ---
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(custom_caption_embed.parameters()) + list(proj.parameters()), lr=2e-5
)

# --- Training Loop ---
for epoch in range(2):
    print(f"\nEpoch {epoch+1}")
    for step, (input_ids, labels, img_feats, captions) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        img_feats = img_feats.to(DEVICE, dtype=torch.bfloat16)

        img_proj = proj(img_feats).unsqueeze(1)  # [B, 1, H]
        token_embeds = custom_caption_embed(input_ids)  # [B, T, H]
        inputs_embeds = torch.cat([img_proj, token_embeds], dim=1)

        labels[labels == tokenizer.pad_token_id] = -100
        labels = torch.cat([torch.full((labels.size(0), 1), -100, device=DEVICE), labels], dim=1)

        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        optimizer.zero_grad()
        xm.mark_step()

        if step % 10 == 0:
            wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

# --- Save Artifacts ---
save_path = "trained_clip_llama"
os.makedirs(save_path, exist_ok=True)

model.cpu()
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
torch.save(custom_caption_embed.cpu().state_dict(), os.path.join(save_path, "caption_embedding.pt"))
torch.save(proj.cpu().state_dict(), os.path.join(save_path, "image_projection.pt"))

artifact = wandb.Artifact("clip-llama-captioning-model", type="model")
artifact.add_dir(save_path)
wandb.log_artifact(artifact)
