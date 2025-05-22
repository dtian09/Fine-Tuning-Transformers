'''
inference using cached Q, K and V with top-k sampling and disk-based base model loading
'''
# --- Imports ---
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPModel, CLIPProcessor
from peft import PeftModel
import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from huggingface_hub import login

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CAPTION_LEN = 32
TOP_K = 50
TEMPERATURE = 1.0
#TEMPERATURE = 0.7 #more focused and fluent captions.
adapter_path = "trained_clip_llama"
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
base_model_dir = "./cached_llama_model"

HF_TOKEN = os.getenv("HF_TOKEN") #running command: export HF_TOKEN=hf_token and run python script.py

# --- Authenticate Hugging Face (required for gated model access) ---
login(token=HF_TOKEN) 

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(adapter_path, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# --- Load Base Model (from disk if exists, else download and save) ---
if os.path.exists(base_model_dir):
    print(f"Loading base model from disk at: {base_model_dir}")
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
        token=HF_TOKEN
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

# --- Load CLIP Encoder + Projection ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

proj = nn.Linear(clip_model.config.projection_dim, model.config.hidden_size).to(DEVICE, dtype=model.dtype)
proj.load_state_dict(
    torch.load(os.path.join(adapter_path, "image_projection.pt"), map_location=DEVICE)
)

# --- Image Encoder ---
@torch.no_grad()
def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(DEVICE)
    img_emb = clip_model.get_image_features(**inputs)
    return img_emb  # shape: [1, 512]

# --- Sampling Function ---
def sample_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    values, _ = torch.topk(logits, top_k)
    threshold = values[:, -1, None]
    logits[logits < threshold] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# --- Inference with Cached QKV ---
@torch.no_grad()
def generate_caption(image):
    # Ensure image is resized and normalized
    img = transforms.Resize((224, 224))(image.convert("RGB"))
    img = transforms.ToTensor()(img)

    # Encode image and project
    img_emb = encode_image(img).to(dtype=proj.weight.dtype)
    img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)

    # Add <sos> token
    sos_id = tokenizer.bos_token_id
    sos_embed = custom_caption_embed(torch.tensor([[sos_id]], device=DEVICE)).to(dtype=model.dtype)

    # Initial input: [img_proj, sos_embed]
    inputs_embeds = torch.cat([img_proj, sos_embed], dim=1)

    outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    next_token = sample_token(logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K)
    generated = [next_token.item()]

    for _ in range(MAX_CAPTION_LEN - 2):
        token_embed = custom_caption_embed(next_token).to(dtype=model.dtype)
        outputs = model(
            inputs_embeds=token_embed,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token = sample_token(logits[:, -1, :], temperature=TEMPERATURE, top_k=TOP_K)
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

    # Filter out punctuation tokens (optional)
    bad_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in [",", ".", "'", '"']]
    generated = [tid for tid in generated if tid not in bad_token_ids]

    return tokenizer.decode(generated, skip_special_tokens=True)


from datasets import load_dataset, load_from_disk

if __name__ == "__main__":
    # --- Load Flickr30k Testset ---
    if os.path.isdir("flickr30k_testset"):
        testset = load_from_disk("flickr30k_testset")
    else:
        testset = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)
        testset = testset.filter(lambda x: x["split"] == "test", keep_in_memory=False)
        testset = testset.remove_columns(
            [col for col in testset.column_names if col not in {"caption", "image"}]
        )
        testset.save_to_disk("flickr30k_testset")

    # --- Run Inference on selected Test Images ---
    start_index = 3 #4
    num_images = 1 #5 #number of images to select
    num_captions = 3 #captions per image

    for idx in range(start_index, start_index + num_images):
        sample = testset[idx]
        image: Image.Image = sample["image"]  # PIL image
        print(f"\n Test Image {idx}")
        print(f"Target Caption: {sample['caption']}")

        for i in range(num_captions):
            caption = generate_caption(image)
            print(f"Generated Caption {i+1}: {caption}")