# Instruct-Seg-Edit on Apple Silicon

Port of the CUDA `app` branch to Apple M-series (tested on **M5, 10 cores, 24 GB
unified memory**, macOS, PyTorch 2.13 MPS).

## What had to change, and why

| Original | Problem on Apple Silicon | Fix |
|---|---|---|
| `FastSAM(path).cuda(device=...)` | `.cuda()` raises without CUDA | Load on CPU, pass `device=` per call; `.cuda()` kept behind a CUDA guard |
| `device: str = "cuda"` defaults | no CUDA device exists | `src/device.py` → `pick_device()`: CUDA → MPS → CPU |
| `torch==2.8.0+cu121`, 15 × `nvidia-*-cu12` | CUDA-only wheels | plain `torch` / `torchvision` from PyPI |
| `triton==3.4.0` | CUDA-only, no MPS equivalent | removed |
| **`nunchaku` (FLUX.1-Fill backend)** | ships as a `linux_x86_64` wheel with custom **int4 CUDA kernels** — cannot run here at all | import made lazy; backend auto-falls back |
| `torch.float16` assumed everywhere | CPU fp16 is emulated and slow | `pick_dtype()`: fp16 on CUDA/MPS, fp32 on CPU |
| `stabilityai/stable-diffusion-2-inpainting` | gated on Hugging Face (401) | default backend is now `sd15`, the open mirror |
| no memory management | 24 GB unified memory is shared with the OS | attention slicing on MPS + `free_memory()` between stages |

The FLUX path is the one genuine loss. Everything else is a like-for-like port.

## Measured on M5

Segmentation, `best.pt` (71.8 M params, YOLOv8-seg / FastSAM), min of 3 runs:

| Workload | MPS | CPU |
|---|---|---|
| segment @640 | **78 ms** | 77 ms |
| segment @1024 | **185 ms** | 192 ms |
| segment + CLIP text match @640 | **89 ms** | 90 ms |
| segment + CLIP text match @1024 | **198 ms** | 200 ms |

For reference the project's own CUDA run logged `93.5 ms inference`. So Apple
Silicon lands in the same band for Stage 1.

**MPS and CPU are level here, and that is expected.** At 71.8 M parameters the
model is too small for the GPU to pull away from the CPU across shared memory —
kernel launch and transfer overhead eat the gain. The first MPS call also costs
~480 ms of one-off Metal graph compilation; benchmark only after a warm-up.

### Stage 2 — inpainting, measured

SD1.5 inpainting, fp16, attention slicing, 512x512:

| | |
|---|---|
| pipeline load | 2.7 s |
| 25 denoising steps | **15.8 s** (633 ms/step) |
| **peak MPS memory** | **2.1 GB** |

That 2.1 GB against 19 GB addressable is the real headline. The CUDA code had
to fight for VRAM — `enable_attention_slicing`, `enable_model_cpu_offload`, and
int4 quantisation of the FLUX transformer through Nunchaku. On 24 GB of unified
memory none of that is necessary: there is roughly 9x headroom, so SDXL or SD3
can be loaded directly instead of the smallest backend.

### End to end

Verified on one image: FastSAM segments the drone from the prompt "a drone",
the mask feeds the inpainting stage, and `"a bright red drone"` recolours only
the masked pixels. The grass is untouched. Both stages ran on MPS.

## Running it

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-macos.txt
./run.sh                      # FastAPI on 127.0.0.1:8000
```

`run.sh` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` — several diffusers ops still have
no MPS kernel and will raise rather than fall back without it.

Override the backend with `INPAINT_BACKEND=sdxl ./run.sh`. Valid on this machine:
`sd15`, `sd2` (gated), `sdxl`, `sd3`, `api`. **`flux` is not** — it needs Nunchaku.
