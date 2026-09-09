"""MagicBrush -> (image, instruction, precise edit mask) triples.

We do not use MagicBrush's own masks as targets. Measured over all 528 dev
turns, they are region-of-interest scribbles: 62.4% of the frame on average
against a real change of 10.3%, i.e. 9.1x too large, precision 15.8%, IoU 0.16
(experiments/exp07_magicbrush). Insertions are worst at 11.9x.

Instead we keep MagicBrush's real human-made source/target pairs and derive the
target mask ourselves by CIE76 deltaE in L*a*b*. That is the same measurement
used to expose the coarseness, turned into supervision.

Note on polarity: MagicBrush masks are bright = PRESERVE. We only use them as a
sanity filter, never as the target.
"""
import glob, io, sys
from pathlib import Path

import numpy as np, cv2, torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"datagen"))
import mask_extraction as ME

IMGSZ = 640
PROTO = IMGSZ // 4          # 160 - prototypes are at stride 4

# Reject degenerate supervision: a change too small to learn from, or so large
# the editor restyled the whole frame.
MIN_CHANGE, MAX_CHANGE = 0.004, 0.45


def kind_of(ins):
    s = ins.strip().lower()
    if s.startswith(("put ", "add ")):     return "insert"
    if s.startswith(("remove", "delete")): return "remove"
    return "modify"


def _decode(cell):
    return Image.open(io.BytesIO(cell["bytes"]))


class MagicBrushEdits(Dataset):
    """Yields image tensor, tokenised instruction, target mask at prototype scale."""

    def __init__(self, split="train", limit=None, imgsz=IMGSZ, cache=True):
        import pandas as pd
        pat = str(ROOT/f"data/magicbrush/data/{split}-*.parquet")
        shards = sorted(glob.glob(pat))
        if not shards:
            raise FileNotFoundError(f"no shards at {pat}")
        self.imgsz = imgsz
        self.rows = []
        kept = dropped = 0
        for f in shards:
            df = pd.read_parquet(f)
            for i in range(len(df)):
                self.rows.append((f, i))
            if limit and len(self.rows) >= limit:
                break
        if limit: self.rows = self.rows[:limit]
        self._df_cache = {}
        self.cache = cache
        self._item_cache = {}

    def __len__(self): return len(self.rows)

    def _df(self, f):
        if f not in self._df_cache:
            import pandas as pd
            self._df_cache = {f: pd.read_parquet(f)}   # one shard resident
        return self._df_cache[f]

    def __getitem__(self, idx):
        if self.cache and idx in self._item_cache:
            return self._item_cache[idx]
        f, i = self.rows[idx]
        df = self._df(f)
        src = _decode(df.source_img.iloc[i]).convert("RGB")
        tgt = _decode(df.target_img.iloc[i]).convert("RGB").resize(src.size)
        ins = str(df.instruction.iloc[i])

        o = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
        e = cv2.cvtColor(np.asarray(tgt), cv2.COLOR_RGB2BGR)
        ch = ME.clean_mask(((ME.delta_e(o, e) > ME.DELTA_E_THRESHOLD)*255).astype(np.uint8)) > 0
        frac = float(ch.mean())

        img = src.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img)).permute(2,0,1).float()/255.0
        m = cv2.resize(ch.astype(np.uint8), (PROTO, PROTO), interpolation=cv2.INTER_AREA)
        y = torch.from_numpy((m > 0).astype(np.float32))[None]

        item = dict(image=x, mask=y, text=ins, kind=kind_of(ins), frac=frac,
                    usable=bool(MIN_CHANGE <= frac <= MAX_CHANGE))
        if self.cache: self._item_cache[idx] = item
        return item


def collate(batch):
    keep = [b for b in batch if b["usable"]]
    if not keep: keep = batch[:1]
    return dict(image=torch.stack([b["image"] for b in keep]),
                mask=torch.stack([b["mask"] for b in keep]),
                text=[b["text"] for b in keep],
                kind=[b["kind"] for b in keep])
