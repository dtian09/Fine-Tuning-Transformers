import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import wandb
from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# --- Constants ---
BATCH_SIZE = 4
MAX_CAPTION_LEN = 32
PERCENTAGE = 1
model_id = "gpt2"

# --- Dataset ---
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
        image = item["image"].convert("RGB")
        image = self.transform(image)
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

        input_ids = torch.cat([
            torch.tensor([sos_id]),
            encoding["input_ids"].squeeze(0)
        ], dim=0)

        labels = torch.cat([
            encoding["input_ids"].squeeze(0),
            torch.tensor([eos_id])
        ], dim=0)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        return {
            "image": image,
            "input_ids": input_ids,
            "labels": labels,
            "caption": caption
        }

def encode_image(image_batch, clip_model, clip_processor, device):
    inputs = clip_processor(images=image_batch, return_tensors="pt", do_rescale=False).to(device)
    return clip_model.get_image_features(**inputs)

def collate(batch):
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    captions = [item["caption"] for item in batch]
    return input_ids, labels, images, captions

def train(index):
    # --- Device ---
    device = xm.xla_device()

    # --- Tokenizer ---
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '<|sos|>'})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<|eos|>'})

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.resize_token_embeddings(len(tokenizer))

    # --- LoRA ---
    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    # --- Image Encoder ---
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --- Projections ---
    caption_embed = nn.Embedding(len(tokenizer), model.config.n_embd).to(device)
    proj = nn.Linear(clip_model.config.projection_dim, model.config.n_embd).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(caption_embed.parameters()) + list(proj.parameters()),
        lr=2e-5
    )

    # --- Dataset ---
    full_dataset = Flickr30kDataset(split="train")
    subset_size = int(PERCENTAGE * len(full_dataset))
    subset_indices = random.sample(range(len(full_dataset)), subset_size)
    subset = Subset(full_dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    # --- Parallel Loader ---
    para_loader = pl.MpDeviceLoader(dataloader, device)

    for epoch in range(2):
        print(f"Epoch {epoch+1} (core {index})")
        for step, (input_ids, labels, images, captions) in enumerate(tqdm(para_loader)):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            img_emb = encode_image(images, clip_model, clip_processor, device).to(dtype=proj.weight.dtype)
            img_proj = proj(img_emb).unsqueeze(1)

            token_embeds = caption_embed(input_ids)
            inputs_embeds = torch.cat([img_proj, token_embeds], dim=1)

            labels[labels == tokenizer.pad_token_id] = -100
            labels = torch.cat([torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device), labels], dim=1)

            outputs = model(inputs_embeds=inputs_embeds, labels=labels)
            loss = outputs.loss
            loss.backward()

            xm.optimizer_step(optimizer)
            optimizer.zero_grad()
            xm.mark_step()

            if step % 10 == 0:
                print(f"[Core {index}] Step {step} | Loss: {loss.item():.4f}")

    if index == 0:
        model.save_pretrained("trained_clip_gpt2_tpu")
        tokenizer.save_pretrained("trained_clip_gpt2_tpu")
        torch.save(caption_embed.state_dict(), "trained_clip_gpt2_tpu/caption_embedding.pt")
        torch.save(proj.state_dict(), "trained_clip_gpt2_tpu/image_projection.pt")

if __name__ == "__main__":
    xmp.spawn(train, args=())
