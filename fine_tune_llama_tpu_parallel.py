import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig
import wandb

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Configuration ---
os.environ["XLA_USE_BF16"] = "1"

BATCH_SIZE = 8  # per core
MAX_CAPTION_LEN = 32
PERCENTAGE = 1 #100%
NUM_EPOCHS = 2
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "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"

# --- CLIP Setup ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

@torch.no_grad()
def encode_image_batch(images, device):
    pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
    inputs = clip_processor(images=pil_images, return_tensors="pt", do_rescale=False).to(device)
    return clip_model.get_image_features(**inputs)

# --- Dataset ---
class Flickr30kDataset(Dataset):
    def __init__(self, split="train", transform=None, max_length=MAX_CAPTION_LEN):
        self.max_length = max_length
        if os.path.isdir("flickr30k_" + split + "_filtered"):
            self.dataset = load_from_disk("flickr30k_" + split + "_filtered")
        else:
            dataset = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)
            dataset = dataset.filter(lambda x: x["split"] == split, keep_in_memory=False)
            self.dataset = dataset.remove_columns([
                col for col in dataset.column_names if col not in {"caption", "image"}
            ])
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

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    captions = [item["caption"] for item in batch]
    return input_ids, labels, images, captions

def train_fn(rank):
    device = xm.xla_device()

    if xm.is_master_ordinal():
        wandb.init(
            project="clip-llama-captioning-parallel",
            entity="dtian",
            config={"batch_size": BATCH_SIZE, "epochs": NUM_EPOCHS, "percentage": PERCENTAGE}
        )

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<sos>'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<eos>'})

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        token=HF_TOKEN
    )
    model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config).to(device)

    caption_embed = nn.Embedding(len(tokenizer), model.config.hidden_size).to(device)
    proj = nn.Linear(clip_model.config.projection_dim, model.config.hidden_size).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(caption_embed.parameters()) + list(proj.parameters()), lr=2e-5
    )

    dataset = Flickr30kDataset(split="train")
    subset_size = int(PERCENTAGE * len(dataset))
    subset_indices = random.sample(range(len(dataset)), subset_size)
    subset = Subset(dataset, subset_indices)

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    mp_loader = pl.MpDeviceLoader(loader, device=device)

    for epoch in range(NUM_EPOCHS):
        for step, (input_ids, labels, images, captions) in enumerate(mp_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            images = images.to(device)

            img_emb = encode_image_batch(images, device).to(device)
            img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)
            token_embeds = caption_embed(input_ids).to(dtype=model.dtype)
            inputs_embeds = torch.cat([img_proj, token_embeds], dim=1)

            labels[labels == tokenizer.pad_token_id] = -100
            labels = torch.cat([torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device), labels], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            optimizer.zero_grad()

            if xm.is_master_ordinal() and step % 10 == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

        xm.rendezvous("epoch_end")

    if xm.is_master_ordinal():
        os.makedirs("trained_parallel_clip_llama", exist_ok=True)
        model.save_pretrained("trained_parallel_clip_llama")
        tokenizer.save_pretrained("trained_parallel_clip_llama")
        torch.save(caption_embed.state_dict(), "trained_parallel_clip_llama/caption_embedding.pt")
        torch.save(proj.state_dict(), "trained_parallel_clip_llama/image_projection.pt")

        artifact = wandb.Artifact("clip-llama-captioning-model-parallel", type="model")
        artifact.add_dir("trained_parallel_clip_llama")
        wandb.log_artifact(artifact)

# --- Launch Training ---
if __name__ == "__main__":
    xmp.spawn(train_fn, start_method='fork')
