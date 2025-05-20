# --- Required Imports ---
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch_xla.core.xla_model as xm

# --- Inference Helper Function ---
@torch.no_grad()
def generate_caption(image_path, model, tokenizer, proj, device):
    model.eval()
    img = transforms.Resize((224, 224))(Image.open(image_path).convert("RGB"))
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # Load and process image through CLIP encoder
    from transformers import CLIPProcessor, CLIPModel
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    inputs = clip_processor(images=[Image.open(image_path).convert("RGB")], return_tensors="pt", do_rescale=False).to(device)
    img_emb = clip_model.get_image_features(**inputs)
    img_proj = proj(img_emb).unsqueeze(1).to(dtype=model.dtype)

    generated_ids = model.generate(inputs_embeds=img_proj, max_length=32)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- Inference Execution (only on TPU master core) ---
if xm.is_master_ordinal():
    model_path = "trained_parallel_clip_llama"
    device = xm.xla_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    proj = nn.Linear(512, model.config.hidden_size).to(device)
    proj.load_state_dict(torch.load(os.path.join(model_path, "image_projection.pt")))
    proj.eval()

    print(generate_caption("image1.jpg", model, tokenizer, proj, device))
