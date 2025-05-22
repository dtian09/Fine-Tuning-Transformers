''' 
inference using cached Q, K and V 
'''
# --- Imports ---
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel, CLIPProcessor
from peft import PeftModel
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CAPTION_LEN = 32
adapter_path = "trained_clip_llama"
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model_dir = "./cached_llama_model"  # local directory for caching
hf_token = "token"
# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# --- Load Base Model from disk if available ---
if os.path.exists(base_model_dir):
    print(f"Loading base model from disk: {base_model_dir}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype="auto"
    )
else:
    print(f"Downloading base model from Hugging Face: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype="auto",
        token=hf_token
    )
    os.makedirs(base_model_dir, exist_ok=True)
    base_model.save_pretrained(base_model_dir)

# --- Load LoRA Adapter ---
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# --- Load Custom Caption Embedding ---
custom_caption_embed = nn.Embedding(len(tokenizer), model.config.hidden_size).to(DEVICE)
custom_caption_embed.load_state_dict(
    torch.load(os.path.join(adapter_path, "caption_embedding.pt"), map_location=DEVICE)
)

# --- Load CLIP Encoder + Image Projection Layer ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

proj = nn.Linear(clip_model.config.projection_dim, model.config.hidden_size).to(DEVICE, dtype=model.dtype)
proj.load_state_dict(
    torch.load(os.path.join(adapter_path, "image_projection.pt"), map_location=DEVICE)
)

# --- Image Feature Encoder ---
@torch.no_grad()
def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(DEVICE)
    img_emb = clip_model.get_image_features(**inputs)
    return img_emb  # shape: [1, 512]

# --- Inference with Cached QKV ---
@torch.no_grad()
def generate_caption(image_path):
    # Load and preprocess image
    img = transforms.Resize((224, 224))(Image.open(image_path).convert("RGB"))
    img = transforms.ToTensor()(img)

    # Encode image and project to LLaMA dimension
    img_emb = encode_image(img).to(dtype=proj.weight.dtype)
    img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)  # (1, 1, hidden_dim)

    # Add <sos> token
    sos_id = tokenizer.bos_token_id
    sos_embed = custom_caption_embed(torch.tensor([[sos_id]], device=DEVICE)).to(dtype=model.dtype)

    # Initial input: [img_proj, sos_embed]
    inputs_embeds = torch.cat([img_proj, sos_embed], dim=1)

    # First forward pass
    outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    # First token prediction
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    generated = [next_token.item()]

    # Autoregressive decoding loop
    for _ in range(MAX_CAPTION_LEN - 2):
        token_embed = custom_caption_embed(next_token.unsqueeze(0)).to(dtype=model.dtype)

        outputs = model(
            inputs_embeds=token_embed,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
    
    # Optional: Filter weird punctuation tokens (customize as needed)
    bad_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in [",", ".", "'", '"']]
    generated = [tid for tid in generated if tid not in bad_token_ids]

    return tokenizer.decode(generated, skip_special_tokens=True)

# --- Example ---
if __name__ == "__main__":
    image_path = "image1.jpg"  # Ensure this image exists
    caption = generate_caption(image_path)
    print("Generated Caption:", caption)
