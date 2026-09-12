"""Memory-mapped feature cache.

The 2,278-sample cache fit in RAM as one 3.8 GB .pt file. The full 51-shard set
is ~7,200 usable samples and the prototypes alone are 11.8 GB, so torch.load of
a single tensor would need that plus a same-sized staging copy on a 26 GB
machine. Store each field as a flat C-ordered memmap instead: a batch of 32
touches 52 MB, the page cache keeps the hot set resident, and peak RSS during
precompute stays flat.

Layout under train/cache_full/:
    proto.f16  (N,32,160,160)   ctx.f16  (N,640)
    text.f16   (N,512)          mask.u8  (N,1,160,160)
    meta.json  {n, kind, rows_done}
"""
import json
from pathlib import Path

import numpy as np, torch

PROTO_SHAPE = (32, 160, 160)
CTX_DIM, TEXT_DIM, M = 640, 512, 160

FIELDS = {
    "proto": (PROTO_SHAPE, np.float16),
    "ctx":   ((CTX_DIM,),  np.float16),
    "text":  ((TEXT_DIM,), np.float16),
    "mask":  ((1, M, M),   np.uint8),
}


def paths(root):
    root = Path(root)
    return {k: root / f"{k}.{'f16' if d == np.float16 else 'u8'}"
            for k, (s, d) in FIELDS.items()}


def open_maps(root, n, mode="r"):
    p = paths(root)
    return {k: np.memmap(p[k], dtype=d, mode=mode, shape=(n,) + s)
            for k, (s, d) in FIELDS.items()}


class _Arr:
    """Torch-tensor view over one memmapped field."""

    def __init__(self, a, n):
        self.a = a
        self.shape = (n,) + a.shape[1:]

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.numpy()
        return torch.from_numpy(np.ascontiguousarray(self.a[idx]))


class MMCache:
    """Dict-like stand-in for the tensors torch.load used to return."""

    def __init__(self, root):
        self.root = Path(root)
        meta = json.loads((self.root / "meta.json").read_text())
        self.n = meta["n"]
        self.kind = meta["kind"]
        maps = open_maps(self.root, self.n, mode="r")
        self.f = {k: _Arr(v, self.n) for k, v in maps.items()}

    def __getitem__(self, k):
        return self.kind if k == "kind" else self.f[k]
