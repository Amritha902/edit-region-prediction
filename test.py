#!/usr/bin/env python3
"""Smoke tests: verify the environment and the model without needing any dataset.

    python test.py

Checks, in order:
  1. the dependency stack imports and versions line up
  2. the mask-extraction logic recovers a known synthetic edit region
  3. segmentor_model builds, runs a forward pass, and completes one train step

Test 3 downloads ~340 MB of ViT-B/16 weights on first run.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PASS, FAIL = "  ok  ", " FAIL "


def test_imports() -> bool:
    print("[1] dependency stack")
    try:
        import numpy as np
        import cv2
        import tensorflow as tf
        from pycocotools import mask as _  # noqa: F401
        import vit_keras
    except ImportError as exc:
        print(f"{FAIL} import failed: {exc}")
        print("      pip install -r requirements.txt   (Python 3.10)")
        return False

    print(f"{PASS} numpy {np.__version__}, opencv {cv2.__version__}, tensorflow {tf.__version__}")

    if np.__version__.startswith("2."):
        print(f"{FAIL} numpy 2.x breaks pycocotools and TF 2.15 — pin numpy<2.0")
        return False
    if not tf.__version__.startswith("2.15"):
        print(f"{FAIL} expected tensorflow 2.15.x, got {tf.__version__} — see README 6.1")
        return False

    devices = [d.device_type for d in tf.config.list_physical_devices()]
    print(f"{PASS} devices: {', '.join(devices)}")
    return True


def test_mask_extraction() -> bool:
    """Draw known edits onto an image and check the mask logic recovers them.

    Exercises mask_extraction.edit_mask directly -- the same function the
    notebooks import -- so this cannot drift from the pipeline.
    """
    print("\n[2] mask extraction (stage 3)")
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    from mask_extraction import edit_mask, edit_mask_grayscale

    def scene():
        img = Image.new("RGB", (512, 512), (120, 140, 160))
        d = ImageDraw.Draw(img)
        d.ellipse([200, 200, 312, 312], fill=(200, 170, 140))   # head
        d.rectangle([210, 312, 302, 460], fill=(60, 80, 120))   # body
        return img

    def bgr(pil):
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    ok = True

    # (a) luminance edit: add a hat above the head.
    orig = scene()
    hat = orig.copy()
    d = ImageDraw.Draw(hat)
    d.rectangle([190, 160, 322, 200], fill=(180, 40, 40))
    d.rectangle([220, 110, 292, 160], fill=(180, 40, 40))
    expected = (132 * 40 + 72 * 50) / (512 * 512)

    frac = float((edit_mask(bgr(orig), bgr(hat)) > 0).mean())
    good = abs(frac - expected) < 0.005
    ok &= good
    print(f"{PASS if good else FAIL} luminance edit: recovered {frac:.2%} "
          f"(expected {expected:.2%})")

    # (b) chroma edit at near-constant luminance -- the case grayscale misses.
    body = scene()
    d = ImageDraw.Draw(body)
    d.rectangle([210, 312, 302, 460], fill=(200, 60, 60))       # was (60,80,120)
    expected_body = (92 * 148) / (512 * 512)

    lab_frac = float((edit_mask(bgr(orig), bgr(body)) > 0).mean())
    gray_frac = float((edit_mask_grayscale(bgr(orig), bgr(body)) > 0).mean())
    good = abs(lab_frac - expected_body) < 0.005
    ok &= good
    print(f"{PASS if good else FAIL} chroma edit:    recovered {lab_frac:.2%} "
          f"(expected {expected_body:.2%})")
    print(f"{'  note':<6} the old grayscale method recovers {gray_frac:.2%} of this same edit")

    if gray_frac >= lab_frac:
        print(f"{FAIL} LAB should strictly improve on grayscale for chroma edits")
        ok = False

    return ok


def test_segmentor() -> bool:
    print("\n[3] segmentor model (stage 4)")
    import numpy as np
    import tensorflow as tf

    try:
        from segmentor import build_segmentor_model, Losses, MASK_SIZE, NUM_POINTS_PROMPT
    except Exception as exc:
        print(f"{FAIL} could not import segmentor: {exc}")
        return False

    print("      building (downloads ViT-B/16 weights on first run)...")
    model = build_segmentor_model()
    total = model.count_params()
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"{PASS} {total:,} params ({trainable:,} trainable, {total - trainable:,} frozen)")

    if model.output_shape[1:3] != MASK_SIZE:
        print(f"{FAIL} output {model.output_shape} does not match MASK_SIZE {MASK_SIZE}")
        return False

    batch = 2
    inputs = {
        "image_input": np.random.rand(batch, 512, 512, 3).astype("float32"),
        "points_input": (np.random.rand(batch, NUM_POINTS_PROMPT, 2) * 512).astype("float32"),
        "labels_input": np.array([[1] + [0] * (NUM_POINTS_PROMPT - 1)] * batch, dtype="int32"),
    }
    target = (np.random.rand(batch, *MASK_SIZE, 1) > 0.7).astype("float32")

    pred = model.predict(inputs, verbose=0)
    lo, hi = float(pred.min()), float(pred.max())
    print(f"{PASS} forward pass {pred.shape}, range [{lo:.3f}, {hi:.3f}]")

    # Regression guard: the mask head must end in a sigmoid. dice_loss and
    # focal_loss both assume probabilities, and both fail silently on logits --
    # training still runs, the masks are just quietly wrong. See README section 8.
    if lo < 0.0 or hi > 1.0:
        print(f"{FAIL} output is outside [0,1] — the sigmoid on the final")
        print("        Conv2DTranspose in MaskDecoder.upsampling_layers is missing.")
        return False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=Losses.combined_loss)
    history = model.fit(inputs, target, epochs=1, batch_size=batch, verbose=0)
    print(f"{PASS} one train step, loss {history.history['loss'][0]:.4f}")
    return True


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if not test_imports():
        return 1
    results = [test_mask_extraction(), test_segmentor()]

    print()
    if all(results):
        print("All smoke tests passed.")
        return 0
    print(f"{sum(1 for r in results if not r)} test(s) failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
