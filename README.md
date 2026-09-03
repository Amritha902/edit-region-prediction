# Instruct-Seg-Edit

**Disentangling Localization from Editing in Instruction-Guided Image Manipulation**

Foundations of Data Science project — Vellore Institute of Technology, Chennai, School of Electronics Engineering.
Amritha S (23BEC1368) · Yugeshwaran P (23BEC1404).

---

## 1. The idea in one paragraph

Diffusion editors like InstructPix2Pix learn *where* to edit and *what* to edit inside a single denoising network. Because that spatial decision is implicit, edits leak into regions the user never asked about, and there is no way to inspect or correct the decision. Inpainting models solve the leakage — they only generate inside a mask — but they cannot work the mask out themselves; it has to be handed to them.

Instruct-Seg-Edit puts a dedicated **localization model** in that gap. Stage 1 reads the image plus the edit instruction and predicts a binary **edit mask**. The user can correct that mask. Stage 2 is a mask-conditioned diffusion model that performs the edit with the loss computed only inside the mask. Localization and editing become two separate, separately-evaluable models.

The critical distinction from referring-expression segmentation (CLIPSeg, LAVT, Grounded-SAM): those models segment *what a phrase denotes*, not *what an edit would change*. For "put a hat on the man", the phrase refers to the man, but the region that must change is the empty space above his head. That is why the masks have to be supervised from real before/after image pairs rather than borrowed from a referring-expression model.

## 2. Why the dataset has to be derived

Training Stage 1 needs the quadruple **(image, instruction, edited image, edit mask)**. No public dataset provides the edit mask. So it is manufactured:

```
PixelProse / RedCaps          →  image + original caption + VLM caption   (public)
        ↓  vision-language model (gemma-3-4b-it / qwen3-vl)
   edit instruction                                                        (derived)
        ↓  InstructPix2Pix  or  FLUX.1 Kontext via ComfyUI
   edited image                                                            (derived)
        ↓  CIE76 ΔE in L*a*b* → threshold @ 12 → morphology
   edit mask                                                               (derived)
```

**Dataset schema**

| Field | Type | Origin | Role in training |
|---|---|---|---|
| Image | RGB 512×512 | PixelProse / RedCaps | Input to Stage 1 and Stage 2 |
| Original caption | Text | PixelProse / RedCaps | Context for instruction generation |
| VLM caption | Text | Gemini 1.0 Pro Vision (in corpus) | Context for instruction generation |
| Edit instruction | Text | gemma-3-4b-it (derived) | Conditioning input to both stages |
| Edited image | RGB | InstructPix2Pix / FLUX Kontext (derived) | Target for Stage 2 |
| Edit mask | Binary | LAB ΔE + morphology (derived) | **Target for Stage 1** |

**Filtering.** Records are dropped when the pixel difference is degenerate — a near-empty mask means the editor ignored the instruction, a near-full mask means it applied a global style change instead of a local edit. Neither is a usable localization target. Connected components under 40 px are removed as generation noise.

## 3. Exploratory data analysis — results

These are the **actual numbers** from the stored outputs in `eda.ipynb`, run over both RedCaps shards, and they are reproducible from the committed notebook. §3.1 and §3.2 are safe to quote at review. **§3.3 is not** — those counts came from a since-fixed bug and must be regenerated first.

**Corpus.** 1,290,861 records, 28 columns. Zero missing values in either caption field. Beyond the fields the pipeline uses, each record carries toxicity scores, `watermark_class_score`, `aesthetic_score`, dimensions, and Reddit metadata (`author`, `subreddit`, `score`) — **none of which the pipeline currently filters on**, though `aesthetic_score` and `watermark_class_score` are obvious quality gates worth adding.

### 3.1 The two caption types are complementary, not redundant

| Metric | Original caption | VLM caption | Ratio |
|---|---|---|---|
| Mean characters | 51.9 | 493.6 | 9.5× |
| Median characters | 39.0 | 442.0 | 11.3× |
| Mean words | 9.5 | 92.1 | 9.7× |
| Median words | 7.0 | 83.0 | 11.9× |
| Mean unique words | 9.0 | 53.9 | 6.0× |

**TF-IDF cosine similarity between the two captions of the same image** (1,000-record sample): mean **0.174**, median **0.053**, std 0.221, range 0.000–0.834.

A median of 0.053 means **more than half of all image pairs share essentially no vocabulary between their two captions.** This is the single most important EDA finding, because it justifies the design decision in `generate_prompts.ipynb`:

```python
caption = f"{vlm_caption}\n{original_caption}".strip()
```

Concatenating is not redundancy — the two fields carry near-disjoint information. The original caption is the human's framing ("first caged ultraboosts. picked them up for $88"), the VLM caption is the dense visual description. An instruction generator given only one of them loses half the signal. The low-similarity examples in the notebook make this vivid: a caption reading "sunday gulch trail, custer state park, south dakota" scores 0.000 against a VLM caption describing a rock formation with a hole in it and two hikers.

### 3.2 Vocabulary

| | Count |
|---|---|
| Original vocabulary | 166,237 |
| VLM vocabulary | 151,215 |
| Common | 91,310 |
| Original only | 74,927 |
| VLM only | 59,905 |
| **Jaccard overlap** | **0.404** |

The disjoint halves are qualitatively different. Original-only tokens are Reddit noise — misspellings and elongations (`paaaaancakesssss`, `interresring`, `undooperative`). VLM-only tokens are largely **proper nouns the model read off signage and packaging** (`srixon`, `edgewoodautosales`, `marlinshipping`, `zarrelli`). Gemini is performing incidental OCR, which is a useful property: brand and text detail in the caption gives the instruction generator concrete, nameable objects to target.

### 3.3 Content distribution — and a caveat that matters

Top VLM caption words are dominated by colour and spatial terms: `white` (760,511), `red` (623,301), `black` (581,144), `green` (470,619), `small` (334,261), `large` (282,252), `sitting` (235,950), `looking` (223,760). These come from the word-frequency cell (22), which tokenizes correctly and is trustworthy — unlike the pattern-count cell below.

> **🔴 The subject / descriptor / action counts below are stale. The code is fixed; the numbers are not.**
>
> `count_patterns` in `eda.ipynb` cell 28 used `str.contains(pattern)` with no word boundary, so it was a **substring** match. Measured against real corpus text recovered from the notebook's own stored outputs:
>
> | Pattern | Also matched | Inflation on sample |
> |---|---|---|
> | `red` | blur**red**, cove**red**, lowe**red** | 4.0× |
> | `old` | f**old**ed, f**old**ing, g**old** | all matches were false |
> | `man` | perfor**man**ce, **man**y, wo**man** | all matches were false |
> | `cat` | lo**cat**ion, **cat**egory | — |
>
> **Fixed** in cell 28 — now `str.contains(r'\b' + re.escape(pattern) + r'\b', regex=True)`, with the counting semantics documented (it is a document frequency — captions *containing* the word — not a term frequency, and the plot titles now say so).
>
> **The stored outputs for that cell have been cleared, because they were produced by the buggy version.** The numbers quoted immediately below are the old, contaminated ones — they are kept only so the contamination is visible. **Re-run cell 28 once you have the parquet shards and replace them before citing anything from this subsection.** Everything else in §3.1–3.2 is unaffected: those use tokenized `re.findall(r'\b[a-zA-Z]+\b', ...)`, which was always correct.

**A finding likely to survive the correction.** `boy` and `girl` are the only two subjects *more* frequent in original captions than VLM captions (boy 19,496 → 6,246; girl 14,479 → 4,256), while `person` goes 2,712 → 110,340. Gemini appears to prefer adult and neutral terms where the human said boy or girl, and if instruction generation depends on subject nouns the derived instructions will inherit that shift. `boy` and `girl` are also the patterns least exposed to substring contamination in this set, which is why this one is worth re-checking first after the re-run — but confirm it rather than assuming it.

### 3.4 A tension worth resolving before scaling up

The instruction prompt says:

> **Exclude:** Do not mention changes to the background, lighting, contrast, saturation, or overall image style.

But by frequency, the dominant content of the VLM captions *is* colour — the top four descriptors are all colours, appearing in 0.5–0.8 M captions each. The generator is being fed a caption saturated with exactly the attribute it is forbidden to act on. The four sample instructions in the notebook do comply, but at 20,000 records this is worth measuring rather than assuming: count how many generated instructions contain a colour word. It is a cheap check and it directly guards the mask quality, because a colour-change instruction produces a diffuse global diff and therefore a degenerate mask.

### 3.5 Note on notebook provenance

`eda.ipynb` cell 9 shows the generation loop was interrupted (`KeyboardInterrupt`) at 0/1,290,861 — only four images were processed there, which is why cells 13 and 15 display exactly four results. **`eda.ipynb` is the design walkthrough, not the production run.** `generate_prompts.ipynb` is the production path and samples 20,000 records. The EDA statistics in 3.1–3.3 are over the full corpus and are unaffected.

The stored outputs also show Windows paths (`C:\Users\pc\AppData\...`), which is consistent with the backslash globbing noted in [Known gaps](#8-known-gaps).

## 4. Repository layout

| File | Stage | What it does |
|---|---|---|
| `eda.ipynb` | 0 | End-to-end walkthrough of the whole pipeline on a small sample, then the exploratory data analysis (caption lengths, word frequencies, TF-IDF caption similarity, vocabulary overlap, subject/descriptor/action counts) |
| `generate_prompts.ipynb` | 1 | **Production instruction generation.** Async, batched, resumable. Downloads images from RedCaps URLs, sends image + caption to a local OpenAI-compatible VLM endpoint, writes one `.txt` per image keyed by MD5 of the URL |
| `generate_edits.ipynb` | 2 | **Production edit generation.** Drives a local ComfyUI server over WebSocket using `flux.json`, uploads each image, injects the instruction, runs FLUX.1 Kontext, saves the edited image |
| `generate_masks.ipynb` | 3 | **Mask extraction.** Calls `mask_extraction.edit_mask` on each original/edited pair and saves the result as the ground-truth mask |
| `flux.json` | 2 | ComfyUI API-format workflow — FLUX.1 Kontext (GGUF), T5-XXL + CLIP-L dual text encoder, 10-step euler/simple, 512×512 |
| `segmentor.py` | 4 | The segmentation model — architecture, losses, and training entry point |
| `mask_extraction.py` | 3 | CIE76 ΔE extraction in L\*a\*b\* plus morphological cleanup, shared by the notebooks and `test.py` so the implementation cannot drift |
| `data_handler.py` | 4 | Batch generator: decodes RLE masks with pycocotools, samples point prompts, yields `(image, points, labels) → mask` |
| `test.py` | — | Empty placeholder |

## 5. Architecture

### ⚠️ Report / code divergence — read this first

The project report (`Instruct-Seg-Edit_Review1_23BEC1368.docx`) describes a **YOLOv8-derived** localization model: CSP backbone with C2f modules, PAN-style neck, YOLACT-style prototype + coefficient segmentation head, Distribution Focal Loss, **73,623,957 parameters**, CLIP text encoder for instruction conditioning, trained on COCO then fine-tuned on the derived dataset.

**`segmentor.py` in this repo does not implement that.** It implements a **SAM-style ViT + point-prompt architecture** trained on SA-1B, with *no text conditioning at all*. The two are different models solving adjacent problems. Section 5.1 documents what the code does; section 5.2 documents what the report specifies. Reconciling them is outstanding work — see [Known gaps](#7-known-gaps).

### 5.1 What `segmentor.py` actually implements

A three-part promptable segmenter in TensorFlow/Keras, closely following the Segment Anything design:

```
image (512×512×3) ──► ViT-B/16 encoder ──► 1024 patch tokens × 768
                       (frozen, imagenet21k+imagenet2012)   │
                                                            │ keys/values
points (10×2) ──┐                                           ▼
                ├─► PromptEncoder ──► 10 × 256 ──► 4 × [self-attn → cross-attn → FFN]
labels (10)  ───┘                                           │
                                                            ▼
                                             token[0] ──► Dense → 32×32×1
                                                       ──► 3× Conv2DTranspose
                                                       ──► resize → 256×256×1 mask
```

| Component | Detail |
|---|---|
| **Image encoder** | `vit_keras.vit.vit_b16`, frozen (`trainable=False`), output taken from `Transformer_encoder_norm`. Class token is stripped (`[:, 1:, :]`), leaving 1024 patch embeddings for a 512×512 input |
| **Prompt encoder** | Point coordinates → `Dense(256)`; labels → `Embedding(3, 256)`. Labels are shifted by +1 so padding (−1), background (0) and foreground (1) map to 0/1/2. The two embeddings are **summed** |
| **Mask decoder** | 4 blocks, each: self-attention over prompt tokens → cross-attention into image embeddings → FFN (256→1024→256), each with a residual + LayerNorm. 8 heads, key_dim 256 |
| **Mask head** | Decoder token 0 is treated as the mask token → `Dense(1024)` → reshape 32×32×1 → `Conv2DTranspose` ×3 (128, 64, 1 channels), final layer **sigmoid** → bilinear resize to 256×256 |
| **Loss** | `combined_loss = focal_loss(γ=2, α=0.25) + dice_loss` |
| **Optimizer** | Adam, lr `1e-5`, 5 epochs, batch size 4 |

**Measured** (smoke test, batch 2, random input, TF 2.15.1 on Apple M5):

```
params      : 114,058,497
trainable   :  27,623,937      (prompt encoder + mask decoder)
frozen      :  86,434,560      (ViT-B/16)
output      : (None, 256, 256, 1),  range [0.378, 0.654]
```

Note this is **114M parameters, not the 73.6M the report specifies** — further evidence the two describe different models.

**Prompt sampling** (`data_handler.py`): for each SA-1B annotation, 1 positive point is taken from the annotation's `point_coords` and 9 negative points are sampled uniformly from background pixels; coordinates are rescaled to the resized image, and slots are padded to 10 with `[0,0]` / label `−1`.

### 5.2 What the report specifies

| Category | Module | Parameters |
|---|---|---|
| Backbone | Conv/C2f/SPPF stages `model.0`–`model.8` | 30,506,183 |
| Neck | PAN-style C2f/Conv `model.9`–`model.21` | 33,781,466 |
| Head | proto 2,267,009 · dfl 16 · box 1,257,024 · cls 4,611,523 · mask-coeff 1,200,736 | 9,336,308 |
| | **Total** | **73,623,957** |

Plus: CLIP text encoder for the instruction, hybrid self/cross-attention fusion, K=32 prototype masks, CIoU+DFL box loss, BCE/Dice/focal mask loss, 1:2 detection-to-segmentation multi-task weighting.

**Evaluation protocol** — localization: IoU, Dice, precision/recall of the edit region. Editing: LPIPS inside the mask, FID, CLIP similarity to the instruction. Efficiency: latency and FPS at batch 1. Baseline: end-to-end InstructPix2Pix.

---

## 6. Setup

### 6.1 Environment

**Use Python 3.10.** Verified working on macOS 15 / Apple M5 / Python 3.10.20.

```bash
python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

The TensorFlow stack is tightly version-locked and the failure modes are all silent or cryptic. Four pins matter, each learned the hard way:

| Pin | Why |
|---|---|
| `tensorflow==2.15.1` | **Not** `tensorflow-macos`. The `tensorflow-macos` 2.15.1 wheel is a 2.2 kB placeholder that installs no code at all — `pip` reports success and `import tensorflow` then fails. Plain `tensorflow` has arm64 support built in |
| `vit-keras==0.1.2` | 0.2.0 calls `keras.ops`, a Keras 3 API. TF 2.15 ships Keras 2.15, so 0.2.0 dies with `module 'keras' has no attribute 'ops'` |
| `tensorflow-addons==0.23.0` | Hard dependency of vit-keras 0.1.2. Prints a loud end-of-life banner; harmless |
| `setuptools<81` | vit-keras imports `pkg_resources`, removed in setuptools 81. Modern venvs ship no setuptools at all, so this must be explicit |

Why not just move to Keras 3? Because `Losses` in `segmentor.py` uses `K.flatten` and `K.binary_crossentropy` from `tensorflow.keras.backend`, which Keras 3 does not provide. Migrating would mean rewriting the loss functions too.

On NVIDIA/Linux, swap `tensorflow==2.15.1` + `tensorflow-metal` for `tensorflow[and-cuda]==2.15.1` and keep the other two pins.

### 6.2 Configuration

```bash
cp .env.example .env
```

Only needed if you point the instruction generator at a hosted API instead of a local one. `.env` is gitignored.

> Note: commits `fab468e` and `7cd1354` touch API-key handling. The key in `generate_prompts.ipynb` is now the literal placeholder `"api_key"`, which is correct for a local LM Studio endpoint (it ignores auth). Supply a real key only if you switch `base_url` to a hosted provider.

## 7. Getting it running — step by step

The four stages are independent and each writes to disk, so you can stop and resume anywhere. **Stages 1–3 are what actually produce the project's dataset; stage 4 is the model.**

### Step 0 — Verify the install, create the data directories

```bash
python test.py
```

Checks the dependency versions, verifies the mask logic against a synthetic edit, and builds + trains the segmentor on random tensors. Needs no dataset. Downloads ~340 MB of ViT weights on first run. Expected output:

```
[1] dependency stack
  ok   numpy 1.26.4, opencv 4.11.0, tensorflow 2.15.1
  ok   devices: CPU, GPU
[2] mask extraction (stage 3)
  ok   luminance edit: recovered 3.47% (expected 3.39%)
  ok   chroma edit:    recovered 5.29% (expected 5.19%)
  note the old grayscale method recovers 0.00% of this same edit
[3] segmentor model (stage 4)
  ok   114,058,497 params (27,623,937 trainable, 86,434,560 frozen)
  ok   forward pass (2, 256, 256, 1), range [-0.187, 0.544]
  ok   one train step, loss 1.3670
All smoke tests passed.
```

Two warnings are expected and harmless: the TensorFlow Addons end-of-life banner, and `Resizing position embeddings from 24, 24 to 32, 32` — ViT-B/16 was pretrained at 384×384 (24×24 patches) and is being used at 512×512 (32×32 patches), so vit-keras interpolates the position embeddings.

```bash
mkdir -p data/images data/prompts data/results data/masks
```

### Step 1 — Download the source corpus (PixelProse RedCaps)

The notebooks expect `data/vlm_captions_redcaps_00.parquet` and `data/vlm_captions_redcaps_01.parquet`. These are parquet shards of caption metadata (URL + original caption + VLM caption) — **not** images. Roughly 250 MB each.

```bash
python scripts/download_data.py --shards 0 1
```

Or manually with the Hugging Face CLI:

```bash
huggingface-cli download tomg-group-umd/pixelprose --repo-type dataset --include "vlm_captions_redcaps_0[01].parquet" --local-dir data
```

The corpus is gated — accept the terms at https://huggingface.co/datasets/tomg-group-umd/pixelprose once, then `huggingface-cli login`.

**Images are not in the corpus.** PixelProse ships URLs only; the pipeline downloads each image at generation time and RedCaps URLs rot, so expect a meaningful failure rate. `generate_prompts.ipynb` enforces a 5-second download timeout and skips anything slower, and it filters out 130×60 placeholder images (the Imgur "removed" graphic).

### Step 2 — Generate edit instructions

Start a local OpenAI-compatible VLM server. With [LM Studio](https://lmstudio.ai), load a vision model and start the server on port 1234. The notebook is configured for `qwen3-vl-1b-merged`; `gemma-3-4b-it` is what the report specifies and gives better instructions.

```bash
# verify the endpoint before running the notebook
curl http://127.0.0.1:1234/v1/models
```

Then run `generate_prompts.ipynb`. It samples 20,000 records, downloads each image to `data/images/<md5>.jpg`, and writes `data/prompts/<md5>.txt`. It is **resumable** — `is_processed()` skips any hash that already has a prompt file, so re-running costs nothing for completed work.

Tune in `CONFIG` if you hit rate limits: `batch_size` (10), `max_concurrent_requests` (10), `download_delay` (0.5 s), `batch_delay` (1.0 s).

### Step 3 — Generate edited images

Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) plus [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF), and place these in `ComfyUI/models/`:

| Model | Path | Node |
|---|---|---|
| `flux1-v4-kontext-dev-mix-tq2_0.gguf` | `unet/` | LoaderGGUF |
| `t5xxl_fp32-iq4_nl.gguf` | `clip/` | DualClipLoaderGGUF |
| `clip_l_v2_fp32-f16.gguf` | `clip/` | DualClipLoaderGGUF |
| `pig_flux_vae_fp32-f16.gguf` | `vae/` | VaeGGUF |

Launch ComfyUI on `127.0.0.1:8188`, then run `generate_edits.ipynb`. It pairs each image with its prompt file by basename, uploads, patches node `5` (image), node `6` (instruction) and node `20` (random seed) in `flux.json`, and saves to `data/results/`.

**The simpler alternative** — `eda.ipynb` cell 11 uses InstructPix2Pix through `diffusers` instead, which needs no ComfyUI and no GGUF files. This is also what the report describes. Use it if you just want the pipeline working:

```python
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None
)
```

(drop `local_files_only=True` and let it download, or pre-fetch to `./models/instruct-pix2pix`).

### Step 4 — Extract the ground-truth masks

```bash
jupyter nbconvert --to notebook --execute generate_masks.ipynb
```

Reads `data/images/` and `data/results/`, writes binary masks to `data/masks/` via `mask_extraction.edit_mask`. Fast, CPU-only, no models. Tune sensitivity with `DELTA_E_THRESHOLD` in `mask_extraction.py`.

### Step 5 — Train the segmentor

⚠️ **This step is currently disconnected from steps 1–4.** `segmentor.py` reads SA-1B JSON annotations, not the masks produced above:

```python
JSON_DIR  = "data/sa-1b-000001-jsons/"
IMAGE_DIR = "data/sa-1b-000001-images/"
```

To run it as written you need one SA-1B shard from https://ai.meta.com/datasets/segment-anything-downloads/ (accept the licence; each shard is ~11 GB, ~11k images with per-image JSON). Then:

```bash
python segmentor.py
```

It builds the model, prints a summary, counts annotations for `steps_per_epoch`, trains 5 epochs and saves `segmentor_model.keras`.

To train on the *derived* dataset instead — which is the actual project goal — `data_handler.py` needs a second generator that reads `data/images/` + `data/masks/` and conditions on the instruction text rather than point prompts. That work is not done yet.

## 8. Known gaps

Honest status, so nothing is a surprise at review:

1. **Report and code implement different architectures.** The report specifies a 73.6M-parameter YOLOv8-derived model with CLIP text conditioning; `segmentor.py` is a SAM-style ViT + point-prompt model that measures **114.1M** parameters. Either the code needs rewriting to match the report, or the report needs to describe the SAM-style model.

2. **The segmentor is not instruction-conditioned.** It takes point prompts. The entire premise of the project is text→mask. There is no text encoder anywhere in `segmentor.py`.
3. **Stage 4 does not consume stages 1–3.** The derived masks in `data/masks/` are never read by the training script; it trains on SA-1B instead.
4. **Degenerate-mask filtering is not implemented.** No check for near-empty or near-full masks, though section 4.5 of the report specifies it.
5. **Stage 2 does not exist in code.** No mask-conditioned diffusion model, no masked loss. The repo produces the dataset and a segmentor; the editing half is design-only.
6. **`data_handler.py` drops the tail of every image.** The batch is only yielded when it reaches exactly `batch_size`, and leftover samples are discarded when the loop moves to the next JSON file. With `batch_size=4` this loses up to 3 annotations per image.
7. **No evaluation code.** None of IoU, Dice, LPIPS, FID or CLIP similarity from the report's evaluation protocol is implemented.

### Fixed

- **Missing sigmoid on the mask head** (`segmentor.py`). `MaskDecoder.upsampling_layers` ended in a bare `Conv2DTranspose`, so the model emitted raw logits (measured `[-0.389, 0.483]`) while both losses assume probabilities in `[0, 1]`. `dice_loss`'s intersection term went negative; `focal_loss`'s `binary_crossentropy(from_logits=False)` clipped every negative prediction to epsilon and killed the gradient there. Measured on identical targets:

  | Predictions | dice | focal |
  |---|---|---|
  | raw logits (before) | 0.9149 | 0.8069 |
  | after sigmoid | 0.6220 | **0.1192** |

  Focal loss was inflated ~6.8×. Fixed by adding `activation="sigmoid"` to the final `Conv2DTranspose`; bilinear resize afterwards preserves `[0, 1]`. Output range is now `[0.378, 0.654]` and a 12-step overfit run brings the loss down monotonically (0.7726 → 0.7360), confirming gradients flow through the sigmoid.

  This one mattered because **training ran fine either way** — loss decreased, nothing crashed. It would only have shown up as quietly bad masks.

- **Substring matching in `eda.ipynb` cell 28.** `count_patterns` used `str.contains(pattern)` with no word boundary, so `red` counted "blurred"/"covered"/"lowered" (4× inflation on a real sample), `old` counted "folded"/"gold", `man` counted "many"/"woman"/"performance", and `cat` counted "location". Now uses `\b`-anchored regex with `re.escape`. The docstring and plot titles also now state that the metric is a document frequency (captions *containing* the word), not a term frequency — the old plot titles said "Mentions", which it never measured. Stale outputs for that cell were cleared; re-run it to regenerate.

- **`segmentor.py:48` ViT layer name.** Looked up `Transformer_encoder_norm`, the vit-keras **0.2.0** name. On 0.1.2 the layer is `Transformer/encoder_norm` and the build died with `ValueError: No such layer`. Now tries both names and reports the available layers if neither is found.

- **Grayscale differencing was blind to colour-only edits.** Mask extraction converted both images to grayscale before differencing, so an edit changing hue at near-constant luminance produced almost no signal and the record was then discarded as degenerate — colour edits left the dataset silently. Replaced with CIE76 ΔE in L\*a\*b\*, now in `mask_extraction.py` and shared by both notebooks and `test.py`.

  The threshold had to be re-derived, since ΔE is not on the grayscale scale. Because the editing models regenerate the whole image, the threshold's real job is rejecting background noise, so it was chosen against simulated regeneration noise on a pair whose true edit region is 3.39%:

  | σ | gray>30 | ΔE>8 | ΔE>12 | ΔE>16 | ΔE>20 |
  |---|---|---|---|---|---|
  | 0 | 3.47% | 3.47% | 3.47% | 3.47% | 3.47% |
  | 3 | 3.47% | 3.97% | 3.48% | 3.47% | 3.47% |
  | 5 | 3.47% | **13.13%** | 4.59% | 3.53% | 3.47% |
  | 8 | 3.47% | **36.88%** | **15.41%** | 6.86% | 4.24% |

  ΔE>8 is unusable under realistic noise. **ΔE>20** matches the noise robustness of the old grayscale threshold while still catching chroma edits with a 4× margin (they measure ΔE 83–105), so it is the default in `DELTA_E_THRESHOLD`.

  Measured effect on a colour-only edit: LAB recovers **5.29%** against a 5.19% ground truth where grayscale recovers **0.00%**. `test.py` asserts LAB strictly beats grayscale on this case, so the regression is guarded.

  **Morphological cleanup** (report §4.5) is now implemented alongside it — an elliptical open then close, then removal of connected components under 40 px. This changes the threshold calculus completely, because the noise it removes is exactly what forced a conservative threshold:

  | σ | ΔE>8 raw | ΔE>8 clean | ΔE>12 raw | ΔE>12 clean | ΔE>20 clean |
  |---|---|---|---|---|---|
  | 3 | 3.96% | 3.47% | 3.47% | 3.47% | 3.47% |
  | 5 | 13.01% | 3.48% | 4.57% | 3.47% | 3.47% |
  | 8 | **36.96%** | **3.51%** | **15.52%** | **3.48%** | 3.47% |

  With cleanup every threshold holds at ~3.47% against a 3.39% ground truth even at σ=8. That bought enough headroom to lower `DELTA_E_THRESHOLD` from 20 to **12**, which catches subtler edits while staying noise-robust. The component minimum is 40 px rather than the report's 10, because ΔE is more sensitive to per-pixel noise than the grayscale average it replaced.

  One caveat remains: all of this was tuned on *synthetic* noise, not real generated pairs. Validate against a hand-labelled sample before trusting the exact numbers.

- **Windows-only paths in three notebooks.** `eda.ipynb` (cells 11, 15), `generate_edits.ipynb` (cells 2, 4) and `generate_masks.ipynb` (cells 1, 2) used backslash literals such as `glob("data\\images\\*.jpg")`. On macOS and Linux that is not a separator — it is part of the filename — so the glob returned `[]` and the loop body never ran. **Nothing raised; the notebook just reported zero work done.** All replaced with `os.path.join`. Verified against a real directory tree: all three cells now pair files and write masks correctly.

  `generate_masks.ipynb` cell 1 also compared an `os.path.join` result against a list of `glob` results, which only works when both spell separators identically. Replaced with `os.path.exists`.

  While rewriting `generate_edits.ipynb` cell 2 I also replaced its nested `for image / for prompt` scan with a dict keyed on filename stem — the original was O(images × prompts), which is 4×10⁸ comparisons at the 20,000 records the pipeline targets. Behaviour is identical; say so if you would rather keep the original loop.

- **`.gitignore` excluded every directory** via `*/`, which would have silently dropped `scripts/` from the repo. Narrowed to `data/`, `models/` and weight files.

## 9. Questions a reviewer will ask

Prepared answers, grounded in what is actually in this repo.

**"Why not just use SAM, or Grounded-SAM?"**
Because they segment what a phrase *denotes*, not what an edit would *change*. "Put a hat on the man" refers to the man; the region that must change is the empty space above his head. No referring-expression model returns that region, because it was never trained on that objective. This is the gap the derived edit masks fill, and it is the strongest single argument for the project. It is also why the highest relevance score in the survey (Grounded-SAM, 0.833) still leaves the problem open.

**"How is this different from DiffEdit? It also produces a mask automatically."**
DiffEdit's mask is a by-product of the diffusion process — computed at inference by contrasting noise predictions, never supervised, not trainable, not separately evaluable, and never shown to the user. Ours is a supervised target with ground truth, so it can be measured with IoU and Dice independently of the edit, and corrected by hand before generation runs.

**"Where does the ground truth for the mask come from? Isn't it circular to supervise on InstructPix2Pix output?"**
Partly, and you should concede this cleanly rather than defend it. The masks come from the pixel difference between original and InstructPix2Pix-edited images, so Stage 1 learns to predict *where InstructPix2Pix would edit*, and inherits its localization biases. The defence is that the pipeline still buys three things the baseline cannot: the mask becomes inspectable, correctable, and separately measurable, and the degenerate-record filter discards exactly the cases where the generator misbehaved. A stronger version of the project would validate a sample of masks against human annotation. Volunteering this before you are asked is worth more than being caught by it.

**"How do you decide which pixels changed, and why that threshold?"**
Pixels are compared with the CIE76 colour difference in L\*a\*b\* and thresholded at ΔE > 20. This replaced a grayscale difference thresholded at 30, which was blind to edits that change hue at constant luminance — a mid-tone red→blue car moves grayscale by only 26 and vanished entirely, while measuring ΔE 105 in LAB.

The threshold was chosen against simulated whole-image regeneration noise, because the editing models redraw the entire frame and the threshold's real job is rejecting that background. Without cleanup, ΔE > 8 reports 13% of the frame as edited at σ=5 against a 3.39% ground truth; ΔE > 12 with morphological cleanup stays at 3.47% and still catches chroma edits with a 4× margin. The sweep is in `mask_extraction.py` and in Known gaps → Fixed.

Concede the limit honestly: it was tuned on synthetic noise, not on real generated pairs, and validating it against a small hand-labelled sample is the obvious next experiment.

**"How many samples do you actually have?"**
1,290,861 available records; `generate_prompts.ipynb` samples 20,000. Note that the download success rate over RedCaps URLs is well below 100% — link rot plus a 5-second timeout — so the realized dataset is smaller than 20,000. **Measure and report the actual count** (`ls data/masks | wc -l`) rather than the sampled figure.

**"Your report says 73.6M parameters and YOLOv8. The code is a ViT."**
This is the most likely hard question and the repo does not paper over it — see the divergence note in section 5. Decide before the review which one is the project: rewrite `segmentor.py` to match the report, or rewrite the report to describe the SAM-style model that exists. Measured parameter count of the code as committed is 114,058,497.

**"Show me it runs."**
`python test.py` — verifies the dependency stack, recovers two known synthetic edit regions (a luminance edit and a colour-only edit) to within 0.1% of ground truth, builds the 114M-parameter model, and completes a forward pass and a training step. No dataset needed.

## 9.5 Measured baseline — the gap is not just argued, it is observed

Everything above argues from the literature that select-an-existing-mask cannot
reach the edit region. `apple-silicon/experiments/exp01_gap/` **measures** it on
this checkpoint, on an Apple M5, over the three edit types.

Two results:

1. **The inherited text conditioning was vacuous.** `segment_image` runs FastSAM
   at ultralytics' default confidence, which leaves a 1-2 mask candidate pool, so
   CLIP selection has nothing to choose between. Six unrelated prompts - including
   the nonsense string "xyzzy nonsense qwerty" - yielded two distinct masks, IoU
   1.0000 across four of them. Fixed with `conf=0.05, retina_masks=True`.

2. **Fixed, the baseline is still 0 / 7**, failing three separate ways:
   *granularity* (returns the whole person for "the jacket"), *spatial language*
   (top-two candidates tie at margin exactly +0.0000 for "the cat on the left",
   because cropped CLIP sees each region in isolation), and *structure*
   (insertions have no candidate to select - asked for empty snow on the left, it
   returned the skis).

Full table, method and stated limits: `apple-silicon/experiments/exp01_gap/FINDINGS.md`.

---

## 10. References

Key works, full list in the report: COCO [1], Mask R-CNN [2], YOLACT [3], YOLOv8 [5], SOLOv2 [6], CLIPSeg [7], LAVT [8], SAM [9], Grounded-SAM [10], LISA [11], InstructPix2Pix [12], Latent Diffusion [13], ControlNet [14], DiffEdit [15], Prompt-to-Prompt [16], PixelProse [17], Gemma 3 [18].

Relevance scored by cosine similarity over an eight-dimension capability vector — highest are Grounded-SAM (0.833), SAM (0.822), DiffEdit (0.791). None covers the full capability set: Grounded-SAM and SAM reach text-to-mask only by chaining large frozen models and perform no edit; DiffEdit performs the edit but never supervises or exposes its mask.
