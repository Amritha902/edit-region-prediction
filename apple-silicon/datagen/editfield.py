"""EditField - an edit-ready representation computed once per image.

The premise being tested: an editor re-derives the whole scene on every
instruction. If the parts of that analysis which do not depend on the
instruction are computed once and reused, then N instructions on one image
should cost one analysis plus N cheap resolutions, not N analyses.

What the field holds, per image:
  regions      instance masks with class, box, area
  free_space   pixels no instance covers - where insertions can go
  supports     which region rests on / is adjacent to which
  slots        affordance anchors: for each region, the free-space band ON,
               ABOVE and BESIDE it, which is where an inserted object lands
  attributes   per-region editability flags (removable, recolorable, ...)

Only `resolve()` depends on the instruction, and it touches no network.

Prior art, honestly: structured scene representations for editing exist -
I2E (arXiv 2601.03741) builds amodal depth-ordered object layers, and
LazyDiffusion (ECCV 2024) compresses global context for a given mask. Neither
reuses one representation across many instructions, and neither resolves an
instruction to empty space. Those two are what this file is for.
"""
import json, time
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


# ---------------------------------------------------------------- construction

def build(det_result, names, topology=None):
    """Build the field from one YOLOv8-seg result. Instruction-independent."""
    t0 = time.time()
    H, W = det_result.orig_shape
    regions = []
    if det_result.masks is not None:
        masks = det_result.masks.data.cpu().numpy() > 0.5
        boxes = det_result.boxes.xyxy.cpu().numpy()
        clses = det_result.boxes.cls.cpu().numpy()
        confs = det_result.boxes.conf.cpu().numpy()
        for i, (m, b, c, cf) in enumerate(zip(masks, boxes, clses, confs)):
            if m.shape != (H, W):
                continue
            area = int(m.sum())
            if area < 64:
                continue
            regions.append(dict(id=i, cls=names[int(c)], conf=float(cf),
                                box=[float(v) for v in b], area=area,
                                area_frac=area / (H * W), mask=m))

    occupied = np.zeros((H, W), bool)
    for r in regions:
        occupied |= r["mask"]
    free_space = ~occupied

    for r in regions:
        r["attributes"] = _attributes(r, regions, free_space, H, W)
        r["slots"] = _slots(r, free_space, H, W)
    supports = _supports(regions)

    return dict(H=H, W=W, regions=regions, free_space=free_space,
                supports=supports, build_sec=time.time() - t0)


def _attributes(r, regions, free_space, H, W):
    """Editability flags. Cheap, deterministic, instruction-independent."""
    m = r["mask"]
    ring = binary_dilation(m, iterations=6) & ~m
    # Removable if there is enough surrounding non-object context to inpaint from.
    ctx_free = float((ring & free_space).sum()) / max(ring.sum(), 1)
    # Boundary complexity: perimeter over sqrt(area). High = fiddly edges (hair, foliage).
    perim = float((m & ~binary_erosion(m)).sum())
    complexity = perim / max(np.sqrt(r["area"]), 1e-6)
    # Occluded if another, larger region overlaps its dilated body.
    occluded = any(o["id"] != r["id"] and o["area"] > r["area"]
                   and (binary_dilation(m, iterations=3) & o["mask"]).any()
                   for o in regions)
    return dict(
        removable=bool(ctx_free > 0.35),
        recolorable=True,                      # any visible surface can be recoloured
        supports_insertion=bool(r["slots_ok"]) if "slots_ok" in r else True,
        occlusion_boundary=bool(occluded),
        context_free_ratio=round(ctx_free, 3),
        boundary_complexity=round(complexity, 2),
        # Fiddly boundaries and heavy occlusion are where editors leak.
        leak_risk=round(min(1.0, complexity / 40.0 + (0.3 if occluded else 0.0)), 3),
    )


def _slots(r, free_space, H, W):
    """Affordance anchors: the free-space bands ON, ABOVE and BESIDE a region.

    This is the piece selection-based methods cannot express. "Put a hat on the
    man" resolves to ABOVE(person) intersected with free space - a region that
    contains no object and therefore has no mask to select.
    """
    x1, y1, x2, y2 = r["box"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    w, h = max(x2 - x1, 1), max(y2 - y1, 1)
    out = {}

    def band(ya, yb, xa, xb):
        ya, yb = max(0, ya), min(H, yb)
        xa, xb = max(0, xa), min(W, xb)
        z = np.zeros((H, W), bool)
        if yb > ya and xb > xa:
            z[ya:yb, xa:xb] = True
        return z

    # ABOVE - a band the height of ~35% of the object, sitting on its top edge
    out["above"] = band(y1 - int(0.35*h), y1 + int(0.05*h), x1 - int(0.1*w), x2 + int(0.1*w)) & free_space
    # ON - the upper third of the object itself (surface placement: hat, collar)
    out["on"]    = band(y1, y1 + int(0.33*h), x1, x2) & r["mask"]
    # BESIDE - flanking bands at object height
    out["beside"] = (band(y1, y2, x2, x2 + int(0.5*w)) | band(y1, y2, x1 - int(0.5*w), x1)) & free_space
    # BELOW - ground contact, for shadows and standing objects
    out["below"] = band(y2 - int(0.05*h), y2 + int(0.25*h), x1, x2) & free_space
    return {k: v for k, v in out.items() if v.sum() > 32}


def _supports(regions):
    """Crude support/adjacency graph: A supports B if B sits on A's top edge."""
    out = []
    for a in regions:
        ax1, ay1, ax2, ay2 = a["box"]
        for b in regions:
            if a["id"] == b["id"]: continue
            bx1, by1, bx2, by2 = b["box"]
            overlap_x = min(ax2, bx2) - max(ax1, bx1)
            if overlap_x <= 0: continue
            if abs(by2 - ay1) < 0.06 * (ay2 - ay1 + 1):
                out.append(dict(support=a["cls"], on=b["cls"], a=a["id"], b=b["id"]))
    return out


# ------------------------------------------------------------------- resolving

# Longest phrase first, and matched on WORD BOUNDARIES. Plain substring
# matching fails here exactly as it did in the EDA: "on" is inside "pers-on-",
# so every instruction mentioning a person resolved to the ON slot, including
# "put a bird above the person".
import re as _re
_PREPS = [("on top of", "on"), ("draped over", "on"), ("next to", "beside"),
          ("above", "above"), ("over", "above"), ("beside", "beside"),
          ("below", "below"), ("under", "below"), ("around", "on"),
          ("on", "on"), ("in", "on")]

def _prep_of(text):
    """First placement preposition present as a whole word, longest phrase first."""
    for phrase, slot in _PREPS:
        if _re.search(rf"\b{_re.escape(phrase)}\b", text):
            return phrase, slot
    return None, None

def resolve(field, instruction):
    """Instruction -> edit region, using only the precomputed field. No network.

    Returns (mask, meta). This is the cheap per-query step whose cost the
    amortisation claim depends on.
    """
    t0 = time.time()
    text = instruction.lower()
    kind = ("insert" if text.startswith(("put ", "add ")) else
            "remove" if text.startswith(("remove", "delete")) else "modify")

    # which region does the instruction name?
    target, pos = None, None
    for r in sorted(field["regions"], key=lambda z: -len(z["cls"])):
        if _re.search(rf"\b{_re.escape(r['cls'].lower())}\b", text):
            same = [q for q in field["regions"] if q["cls"] == r["cls"]]
            if len(same) > 1:
                pos = next((p for p in ("left","right","top","bottom","middle")
                            if _re.search(rf"\b{p}\b", text)), None)
                target = _disambiguate(same, pos) or r
            else:
                target = r
            break
    if target is None:
        return None, dict(reason="no region named in instruction", sec=time.time()-t0)

    if kind == "insert":
        _, slot = _prep_of(text)
        if slot is not None and slot not in target["slots"]:
            slot = None                      # named placement has no free space
        if slot is None:
            slot = "on" if "on" in target["slots"] else next(iter(target["slots"]), None)
        if slot is None:
            return None, dict(reason="no free slot", sec=time.time()-t0)
        mask = target["slots"][slot]
        meta = dict(kind=kind, region=target["cls"], slot=slot, pos=pos)
    else:
        mask = target["mask"]
        meta = dict(kind=kind, region=target["cls"], slot=None, pos=pos)

    meta.update(sec=time.time()-t0, leak_risk=target["attributes"]["leak_risk"],
                removable=target["attributes"]["removable"])
    return mask, meta


def _disambiguate(same, pos):
    if not pos: return None
    cx = [(r["box"][0]+r["box"][2])/2 for r in same]
    cy = [(r["box"][1]+r["box"][3])/2 for r in same]
    if pos == "left":   return same[int(np.argmin(cx))]
    if pos == "right":  return same[int(np.argmax(cx))]
    if pos == "top":    return same[int(np.argmin(cy))]
    if pos == "bottom": return same[int(np.argmax(cy))]
    if pos == "middle" and len(same) == 3:
        return same[int(np.argsort(cx)[1])]
    return None


def summary(field):
    return dict(regions=len(field["regions"]),
                classes=sorted({r["cls"] for r in field["regions"]}),
                free_frac=round(float(field["free_space"].mean()), 3),
                slots=sum(len(r["slots"]) for r in field["regions"]),
                supports=len(field["supports"]),
                build_sec=round(field["build_sec"], 3))
