import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS","1")
from huggingface_hub import snapshot_download
p = snapshot_download("stabilityai/stable-diffusion-2-inpainting",
                      cache_dir="models/stable-diffusion-2-inpainting",
                      allow_patterns=["*.json","*.txt","*fp16.safetensors","tokenizer/*","scheduler/*"])
print("SD2 inpainting cached at:", p)
