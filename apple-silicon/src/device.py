"""Device selection for Instruct-Seg-Edit.

The original code hard-codes CUDA: FastSAM(...).cuda(), device="cuda", fp16
everywhere, and a FLUX backend that loads Nunchaku int4 weights through a
linux_x86_64 wheel with custom CUDA kernels. None of that exists on Apple
Silicon, so this module centralises the choice instead.

Order of preference: CUDA, then Apple MPS, then CPU.
"""
from __future__ import annotations
import os, torch

# Several diffusers/ultralytics ops still have no MPS kernel. Without this they
# raise instead of silently running on CPU, which kills a whole pipeline run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(preferred: str | None = None) -> str:
    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    """fp16 on CUDA and MPS, fp32 on CPU.

    MPS handles fp16 for diffusion fine and it roughly halves memory, which
    matters because unified memory is shared with the rest of the system.
    CPU fp16 is emulated and much slower, so stay in fp32 there.
    """
    return torch.float32 if device == "cpu" else torch.float16


def default_backend(device: str) -> str:
    """The FLUX path needs Nunchaku, which is CUDA-only. Pick something that
    actually loads on this machine."""
    # sd2 is gated on Hugging Face; sd15 is the open mirror and the
    # lightest pipeline that still takes a mask + ControlNet.
    return "flux" if device.startswith("cuda") else "sd15"


def describe(device: str) -> str:
    if device.startswith("cuda"):
        return f"CUDA · {torch.cuda.get_device_name(0)}"
    if device == "mps":
        import subprocess
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            chip = "Apple Silicon"
        total = torch.mps.recommended_max_memory() / 1e9
        return f"Apple MPS · {chip} · {total:.0f} GB recommended max"
    return "CPU"


def free_memory(device: str) -> None:
    """Release cached allocations between the two stages. Both models will not
    fit in 24 GB of unified memory at once."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
