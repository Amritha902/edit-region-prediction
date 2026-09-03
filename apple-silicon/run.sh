#!/bin/zsh
# Instruct-Seg-Edit on Apple Silicon
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1     # some diffusers ops have no MPS kernel
export INPAINT_BACKEND="${INPAINT_BACKEND:-sd2}"
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
