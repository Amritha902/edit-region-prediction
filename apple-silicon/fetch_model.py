"""Resumable, retrying fetch of the inpainting weights.

The connection drops partway through the 1.7 GB UNet, so: single worker to keep
one stable connection, long timeout, and retry in a loop. huggingface_hub keeps
partial blobs, so each attempt resumes rather than restarting.
safety_checker is skipped - the pipeline is constructed with safety_checker=None.
"""
import os, time
os.environ["HF_HUB_DISABLE_XET"] = "1"        # xet CAS backend errors on this network
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
from huggingface_hub import snapshot_download

REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PATTERNS = ["*.json", "*.txt", "tokenizer/*", "scheduler/*",
            "unet/*fp16.safetensors", "vae/*fp16.safetensors",
            "text_encoder/*fp16.safetensors"]

for attempt in range(1, 21):
    try:
        p = snapshot_download(REPO, cache_dir="models/stable-diffusion-inpainting",
                              allow_patterns=PATTERNS, max_workers=1)
        print("cached:", p, flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(5)
else:
    print("gave up after 20 attempts", flush=True)
