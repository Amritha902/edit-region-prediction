"""Stage 1 of dataset generation - grounded edit instructions.

We do not sample instructions from captions alone. The three failure modes
measured in experiments/exp01_gap are granularity, spatial language and
insertion, so the instruction set is built to exercise exactly those:

  modify  - a whole object, or a *part* of a person  (granularity)
  remove  - an object, disambiguated by position when its class repeats (spatial)
  insert  - a plausible object placed onto empty space near a host (affordance)

Grounding comes from a COCO-pretrained YOLOv8 detector, so every instruction
refers to something actually in the image, and we record the target instance's
box. That box is ground truth the delta-E mask cannot give us: it lets us later
ask "did the model pick the right instance?", not just "did it change pixels?".
"""
import random

# What can plausibly be added to / onto a given COCO class. The insert case has
# no referent object, so the edit region is empty space near the host - this
# table is the affordance prior that makes the request sensible.
# What can plausibly be added near a given COCO class, as (item, placement).
# The placement preposition is kept separate from the noun: baking it into the
# noun produced "put a game console below it on the left tv" in the first run.
AFFORD = {
    "person":      [("hat", "on"), ("pair of sunglasses", "on"), ("backpack", "on"),
                    ("scarf", "around"), ("necklace", "on")],
    "dog":         [("collar", "on"), ("bandana", "around"), ("small hat", "on")],
    "cat":         [("collar", "on"), ("bow tie", "on"), ("small hat", "on")],
    "horse":       [("saddle", "on"), ("blanket", "over")],
    "dining table":[("vase of flowers", "on"), ("cup of coffee", "on"),
                    ("book", "on"), ("laptop", "on")],
    "couch":       [("cushion", "on"), ("folded blanket", "on")],
    "chair":       [("cushion", "on"), ("jacket", "draped over")],
    "bed":         [("pillow", "on"), ("teddy bear", "on")],
    "car":         [("roof rack", "on"), ("surfboard", "on top of")],
    "truck":       [("ladder", "on top of")],
    "bicycle":     [("basket", "on"), ("water bottle", "on")],
    "motorcycle":  [("saddlebag", "on")],
    "boat":        [("flag", "on")],
    "bench":       [("cushion", "on"), ("backpack", "on")],
    "tv":          [("game console", "below")],
    "laptop":      [("coffee mug", "beside")],
    "pizza":       [("basil leaves", "on")],
    "cake":        [("candles", "on"), ("strawberries", "on")],
    "bowl":        [("spoon", "in")],
    "sink":        [("potted plant", "beside")],
    "toilet":      [("roll of paper", "beside")],
    "refrigerator":[("magnet", "on")],
    "teddy bear":  [("ribbon", "on"), ("small hat", "on")],
    "vase":        [("flowers", "in")],
    "backpack":    [("keychain", "on")],
    "suitcase":    [("luggage tag", "on")],
}
# Generic fallback so any class can host an insertion.
AFFORD_ANY = [("small potted plant", "beside"), ("ribbon", "tied to"),
              ("sticker", "on")]

# Parts, for the granularity case. Only for classes whose parts are reliably
# visible and nameable.
PARTS = {
    "person": ["jacket", "shirt", "trousers", "shoes", "hat", "hair"],
    "car":    ["wheels", "windshield", "bonnet"],
    "truck":  ["wheels", "cab"],
    "bus":    ["windows", "wheels"],
    "dog":    ["fur"],
    "cat":    ["fur"],
}

COLORS    = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink"]
MATERIALS = ["wood", "metal", "glass", "marble", "leather"]  # used as "look like X"


def _spatial(idx, boxes):
    """Position word that uniquely picks box `idx` out of same-class `boxes`.

    Returns None when the class does not repeat (no qualifier needed) or when
    no single axis separates them cleanly - we would rather emit no instruction
    than an ambiguous one.
    """
    if len(boxes) < 2:
        return None
    cx = [(b[0] + b[2]) / 2 for b in boxes]
    cy = [(b[1] + b[3]) / 2 for b in boxes]
    # horizontal separation first, it is the more natural phrasing
    if max(cx) - min(cx) > 0.15:
        order = sorted(range(len(boxes)), key=lambda i: cx[i])
        if order[0] == idx:  return "left"
        if order[-1] == idx: return "right"
        return "middle" if len(boxes) == 3 else None
    if max(cy) - min(cy) > 0.15:
        order = sorted(range(len(boxes)), key=lambda i: cy[i])
        if order[0] == idx:  return "top"
        if order[-1] == idx: return "bottom"
    return None


def _ref(cls, pos):
    return f"the {pos} {cls}" if pos else f"the {cls}"


def build(dets, rng):
    """Instructions for one image.

    dets: list of {cls, box (normalised xyxy), area_frac}
    Returns a list of dicts, each one training sample's specification.
    """
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)

    out = []
    for cls, group in by_cls.items():
        boxes = [g["box"] for g in group]
        for i, d in enumerate(group):
            # ignore specks and near-full-frame detections: delta-E on either
            # produces a mask that teaches nothing
            if not (0.01 <= d["area_frac"] <= 0.60):
                continue
            pos = _spatial(i, boxes)
            if len(group) > 1 and pos is None:
                continue                      # ambiguous - skip rather than mislabel
            ref = _ref(cls, pos)

            cand = []
            # --- modify, whole object
            cand.append(dict(kind="modify", sub="object",
                             text=f"make {ref} {rng.choice(COLORS)}"))
            # --- modify, part  (granularity case)
            if cls in PARTS:
                part = rng.choice(PARTS[cls])
                cand.append(dict(kind="modify", sub="part",
                                 text=f"make {ref}'s {part} {rng.choice(COLORS)}"))
            else:
                cand.append(dict(kind="modify", sub="material",
                                 text=f"make {ref} look like {rng.choice(MATERIALS)}"))
            # --- remove
            cand.append(dict(kind="remove", sub="object", text=f"remove {ref}"))
            # --- insert  (affordance case)
            item, prep = rng.choice(AFFORD.get(cls, AFFORD_ANY))
            article = "an" if item[0] in "aeiou" else "a"
            cand.append(dict(kind="insert", sub="affordance",
                             text=f"put {article} {item} {prep} {ref}"))

            for c in cand:
                out.append(dict(**c, target_cls=cls, target_box=d["box"],
                                target_pos=pos, n_same_class=len(group)))
    return out


def balanced_pick(cands, k, rng):
    """Pick k instructions with the three edit kinds as even as we can manage.

    Uniform sampling over candidates gave 1002 modify / 526 remove / 472 insert,
    because modify contributes two of the four candidates per object. Insert is
    the case the whole project is about, so it must not be the rarest.
    """
    byk = {}
    for c in cands:
        byk.setdefault(c["kind"], []).append(c)
    for v in byk.values():
        rng.shuffle(v)
    # Rotate which kind goes first. A fixed order with k=2 per image starves
    # the third kind entirely - the first attempt produced 1000 insert,
    # 1000 remove and zero modify.
    order = ["insert", "remove", "modify"]
    r = rng.randrange(3)
    order = order[r:] + order[:r]
    out = []
    while len(out) < k and any(byk.values()):
        progressed = False
        for kind in order:
            if len(out) >= k: break
            if byk.get(kind):
                out.append(byk[kind].pop()); progressed = True
        if not progressed: break
    return out
