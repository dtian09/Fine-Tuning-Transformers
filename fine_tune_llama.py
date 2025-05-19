import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, load_from_disk
import transformers
from peft import get_peft_model, LoraConfig
import os
import random
import wandb
from tqdm import tqdm

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_CAPTION_LEN = 32
PERCENTAGE = 0.1  # Use 10% of the training data

# --- Initialize Weights & Biases ---
wandb.init(
   project="clip-llama-captioning", 
   entity="dtian", 
   config={
    "batch_size": BATCH_SIZE,
    "max_caption_len": MAX_CAPTION_LEN,
    "percentage_used": PERCENTAGE,
    "tuning_method": "LoRA",
    "embedding_type": "custom nn.Embedding",
    "model": "meta-llama/Llama-3-8B-Instruct"
})

# --- Tokenizer and Model (LLaMA-3) ---
HF_TOKEN = "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"  # You must set this properly

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct", 
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct", 
    device_map="auto", 
    torch_dtype="auto", 
    token=HF_TOKEN
)

# --- CLIP Image Encoder ---
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    img_emb = clip_model.get_image_features(**inputs)
    return img_emb  # shape: [1, 512]

# --- Flickr30k Dataset Class ---
from torch.utils.data import Dataset
class Flickr30kDataset(Dataset):
    def __init__(self, split="train", transform=None, tokenizer_name='meta-llama/Llama-3-8B-Instruct', max_length=31):
        self.max_length = max_length

        if os.path.isdir("flickr30k_" + split + "_filtered"):
            self.dataset = load_from_disk("flickr30k_" + split + "_filtered")
        else:
            dataset = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)
            dataset = dataset.filter(lambda x: x["split"] == split, keep_in_memory=False)
            self.dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in {"caption", "image"}]
            )
            dataset.save_to_disk("flickr30k_" + split + "_filtered")

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        image = self.transform(image)

        caption = str(item["caption"])

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "caption": caption
        }

# --- Dataset & Subset Sampling ---
full_dataset = Flickr30kDataset(split="train")
total_samples = len(full_dataset)
subset_size = int(PERCENTAGE * total_samples)
subset_indices = random.sample(range(total_samples), subset_size)
dataset = Subset(full_dataset, subset_indices)

def collate(batch):
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attn_mask = torch.stack([item["attention_mask"] for item in batch])
    captions = [item["caption"] for item in batch]

    image_embeddings = torch.cat([encode_image(img).unsqueeze(0) for img in images], dim=0)

    return input_ids, attn_mask, image_embeddings, captions

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

# --- Apply LoRA to LLaMA-3 ---
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- Custom Caption Embedding ---
custom_caption_embed = nn.Embedding(tokenizer.vocab_size, model.config.hidden_size).to(DEVICE)

# --- Training Loop with Progress Bar ---
optimizer = torch.optim.Adam(list(model.parameters()) + list(custom_caption_embed.parameters()), lr=2e-5)

for epoch in range(2):
    print(f"\nEpoch {epoch+1}")
    for step, (input_ids, attn_mask, img_emb, captions) in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        img_emb = img_emb.to(DEVICE)

        proj = nn.Linear(img_emb.size(-1), model.config.hidden_size).to(DEVICE)
        img_proj = proj(img_emb).unsqueeze(1)  # shape: [B, 1, hidden]

        token_embeds = custom_caption_embed(input_ids)
        inputs_embeds = torch.cat([img_proj, token_embeds], dim=1)  # prepend image token

        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(inputs_embeds=inputs_embeds, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": step})

# --- Save Model and Upload to W&B ---
save_path = "trained_clip_llama"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
custom_caption_embed_path = os.path.join(save_path, "caption_embedding.pt")
torch.save(custom_caption_embed.state_dict(), custom_caption_embed_path)

artifact = wandb.Artifact("clip-llama-captioning-model", type="model")
artifact.add_dir(save_path)
wandb.log_artifact(artifact)

# --- Inference Example ---
def generate_caption(image_path):
    img = transforms.Resize((224, 224))(Image.open(image_path).convert("RGB"))
    img = transforms.ToTensor()(img)
    img_emb = encode_image(img)
    proj = nn.Linear(img_emb.size(-1), model.config.hidden_size).to(DEVICE)
    img_proj = proj(img_emb).unsqueeze(1)

    generated_ids = model.generate(inputs_embeds=img_proj, max_length=MAX_CAPTION_LEN)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(generate_caption("test_images/image_0.jpg"))
