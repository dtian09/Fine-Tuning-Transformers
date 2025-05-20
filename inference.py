# --- Load Trained Model, Tokenizer, and Custom Layers ---
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import os
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CAPTION_LEN = 32
save_path = "trained_clip_llama"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(save_path)

# Load model
model = AutoModelForCausalLM.from_pretrained(save_path, device_map="auto", torch_dtype="auto")
model.eval()

# Load custom caption embedding
custom_caption_embed = nn.Embedding(len(tokenizer), model.config.hidden_size).to(DEVICE)
custom_caption_embed.load_state_dict(torch.load(os.path.join(save_path, "caption_embedding.pt"), map_location=DEVICE))

# Load image projection layer
from transformers import CLIPModel, CLIPProcessor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

proj = nn.Linear(clip_model.config.projection_dim, model.config.hidden_size).to(DEVICE, dtype=model.dtype)
proj.load_state_dict(torch.load(os.path.join(save_path, "image_projection.pt"), map_location=DEVICE))

@torch.no_grad()
def encode_image(image):
    inputs = clip_processor(images=image, return_tensors="pt", do_rescale=False).to(DEVICE)
    img_emb = clip_model.get_image_features(**inputs)
    return img_emb  # shape: [1, 512]

# --- Inference ---
@torch.no_grad()
def generate_caption(image_path):
    # Preprocess image
    img = transforms.Resize((224, 224))(Image.open(image_path).convert("RGB"))
    img = transforms.ToTensor()(img)

    # Encode and project image
    img_emb = encode_image(img).to(DEVICE)
    img_emb = img_emb.to(dtype=proj.weight.dtype)
    img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)  # shape: (1, 1, hidden_dim)

    # Add <sos> token embedding
    sos_id = tokenizer.bos_token_id
    sos_embed = custom_caption_embed(torch.tensor([[sos_id]], device=DEVICE)).to(dtype=model.dtype)  # (1, 1, hidden_dim)

    # Concatenate image + <sos>
    inputs_embeds = torch.cat([img_proj, sos_embed], dim=1)  # shape: (1, 2, hidden_dim)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE)

    # Generate caption
    generated_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_length=MAX_CAPTION_LEN,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- Example Inference ---
print(generate_caption("image1.jpg"))
