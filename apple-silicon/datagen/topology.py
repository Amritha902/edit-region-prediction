"""Topology descriptors for an edit region.

Every current system treats the edit region as a bag of pixels. The shape of
that region carries information a mask does not: "put a necklace on her" is a
thin closed band, "remove the person" is one solid blob, "add a crack" is a
thin branching curve, "make it sunset" is a large connected field. Those need
different amounts of precision in different places.

This module turns a binary mask into a fixed-length descriptor. It is
deterministic image analysis, not learning - which is the point: it gives us
supervised targets a model can later be trained to predict directly from
(image, instruction), before the editor is ever run.

Prior art note: region-aware acceleration (RegionE, SpotEdit) and sufficient-
context compression (LazyDiffusion, ECCV 2024) are occupied. Predicting the
region - and its shape - from the instruction alone is not.
"""
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops, euler_number


def describe(mask):
    """Topology descriptor for a binary mask (H, W) bool.

    Returns a flat dict. Every value is scale-invariant or normalised by the
    frame, so descriptors are comparable across images of different sizes.
    """
    m = np.asarray(mask, dtype=bool)
    H, W = m.shape
    area = int(m.sum())
    out = dict(area_frac=area / (H * W))
    if area == 0:
        return {**out, "n_components": 0, "largest_frac": 0.0, "elongation": 0.0,
                "thinness": 0.0, "n_loops": 0, "solidity": 0.0, "orientation": 0.0,
                "skeleton_frac": 0.0, "boundary_frac": 0.0, "kind": "empty"}

    lab = label(m, connectivity=2)
    props = sorted(regionprops(lab), key=lambda p: -p.area)
    big = props[0]
    out["n_components"] = len(props)
    out["largest_frac"] = big.area / area          # 1.0 = single blob

    # Elongation from the second moments of the largest component.
    maj = max(big.axis_major_length, 1e-6)
    mnr = max(big.axis_minor_length, 1e-6)
    out["elongation"] = float(maj / mnr)
    out["orientation"] = float(big.orientation)     # radians, -pi/2..pi/2
    out["solidity"] = float(big.solidity)           # area / convex hull area

    # Skeleton length vs area separates thin structures from solid ones.
    # A band or crack has skeleton length comparable to its area; a blob does not.
    skel = skeletonize(m)
    out["skeleton_frac"] = float(skel.sum()) / area
    out["thinness"] = float(area / max(skel.sum(), 1))   # mean half-width proxy

    # Euler number = components - holes. Loops (a necklace, a bracelet) have holes.
    out["n_loops"] = int(max(0, len(props) - euler_number(m, connectivity=2)))

    # Boundary fraction: how much of the region is edge. High = precision matters.
    from scipy.ndimage import binary_erosion
    out["boundary_frac"] = float((m & ~binary_erosion(m)).sum()) / area

    out["kind"] = classify(out)
    return out


def classify(d):
    """Coarse shape class - the label a model would predict as an auxiliary head."""
    if d["area_frac"] > 0.35:                      return "field"      # sky, global
    if d["n_loops"] >= 1 and d["thinness"] < 12:   return "band"       # necklace, collar
    if d["thinness"] < 6 and d["elongation"] > 4:  return "filament"   # crack, pole
    if d["n_components"] > 4:                      return "scattered"  # drift / diffuse
    if d["largest_frac"] > 0.8:                    return "blob"       # object
    return "mixed"


def drift_score(d):
    """How much this mask looks like global photometric drift rather than an edit.

    The exp02 pilot failure: a brightened frame yields many small components
    spread over the image. Real local edits are few, compact components.
    High score = reject the sample.
    """
    if d["area_frac"] == 0: return 1.0
    return float(min(1.0, (d["n_components"] / 12.0) * (1.0 - d["largest_frac"])))
