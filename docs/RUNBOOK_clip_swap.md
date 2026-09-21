# Runbook — CLIP Encoder Swap (run this on your local machine)

**Why local:** the sandbox where this dataset was built blocks
`huggingface.co`, `openaipublic.azureedge.net` and `download.pytorch.org` (all
403), and has 2.8 GB of disk free. No pretrained weights can be fetched there.
Your RTX 4060 (8 GB) runs CLIP ViT-B/32 comfortably — ~350 MB of weights, a few
minutes of inference over ~2,000 images. Colab works too.

**Baseline to beat** (recorded in `TECHNICAL_REPORT_02`, identical protocol):

```
FUSION (ours)            PR-AUC 0.818 +/- 0.022   ROC-AUC 0.938
base-paper-equivalent    PR-AUC 0.772 +/- 0.023   ROC-AUC 0.919
paired difference        +0.045   p = 4.4e-06
```

---

## Step 0 — Setup

```bash
pip install open_clip_torch torch torchvision
# alternative backend, also supported:
# pip install transformers torch torchvision
```

You need the image files. If you don't have the 152 MB `out/images/` folder,
regenerate it — the seed is fixed, so it reproduces byte-identically:

```bash
git clone https://github.com/Venkatasai200628/fake_review_detection repo
cd dataset && python build_dataset.py          # ~15 min, writes ../out/
python -c "
import os, pandas as pd, forensics
df = pd.read_csv('../out/reviews_full.csv')
f = pd.DataFrame([forensics.analyse(os.path.join('../out', r.image_file))
                  for r in df.itertuples()])
f.insert(0, 'review_id', df.review_id.values)
f.to_csv('../out/forensic_features.csv', index=False)"
```

---

## Step 1 — Confirm the baseline reproduces

Before changing anything, verify you get the numbers above on your machine.

```bash
python crossval.py
```

If your PR-AUC differs from 0.818 by more than the confidence interval,
something in the environment differs and the comparison won't be valid. Sort
that out before proceeding.

---

## Step 2 — Recompute embeddings in place

```bash
python recompute_embeddings.py --dry-run    # checks image paths only
python recompute_embeddings.py
```

**Why in place, not a full regeneration:** re-running `build_dataset.py` would
redraw every transform, reviewer and timestamp, so any change in results would
be confounded by dataset churn. This script touches the `embedding` column and
nothing else, so exactly one variable changes.

What it writes:

| File | Contents |
|---|---|
| `reviews_full.csv`, `train.csv`, `holdout_realistic.csv` | `embedding` column → 512-d CLIP |
| `clip_image_embeddings.npy` | raw image vectors |
| `clip_text_embeddings.npy` | raw text vectors |
| `clip_cross_modal.csv` | per-review image·text similarity |
| `embeddings_backup_placeholder.csv` | rollback |

Rollback if needed: merge the backup CSV's `embedding` column back in.

---

## Step 3 — Retune the reuse threshold (do not skip)

```bash
python tune_thresholds.py
```

`COSINE_REUSE_THRESHOLD = 0.94` in `features.py` was tuned for the 64-d
placeholder descriptor. **It will be wrong for CLIP.** CLIP places two unrelated
photographs of the same product category at 0.8+, far higher than the
placeholder did, so 0.94 will flood the provenance module with false reuse edges
— which would also corrupt the coordination graph, since its edges are built
from the same threshold.

The script prints two candidates. Take the **1% FPR** value, not the max-Youden
value: a provenance signal feeding a fraud score should be conservative, and a
false reuse edge propagates into the graph features as well.

Edit `features.py` with the recommended value, then re-run `validate.py` and
check that check 2 (embedding cosine separation) still shows a clean margin.

---

## Step 4 — Re-run the identical protocol

```bash
python crossval.py
```

Compare to the baseline table above. Because the protocol, the folds, the seed
and the images are all unchanged, any difference is attributable to the encoder.

---

## Step 5 — The cross-modal feature (this is the interesting part)

`clip_cross_modal.csv` gives you something the pipeline **does not currently
have at all**: a score for whether the review text actually describes the
attached photograph, computed in CLIP's shared image-text space.

`features.py` already picks this file up automatically if present — no code
change needed. It becomes an extra column in the IMAGE block.

This matters because several published multimodal detectors build their entire
contribution on exactly this signal, and because it directly serves the
project's stated novelty. It should help most on `text_deception`, where the
text is generic and won't describe the specific product in the photo.

Worth running as a separate ablation so you can attribute it:

- fusion with CLIP embeddings, without cross-modal
- fusion with CLIP embeddings, with cross-modal

---

## What to expect, honestly

**Likely to improve:** provenance and reuse detection (CLIP embeddings are far
stronger than a 64-d gradient histogram), and `text_deception` recall via the
cross-modal feature.

**Unlikely to change:** the manipulation module. CLIP encodes semantics, not
compression artifacts. That module sits at ROC-AUC 0.622 and needs a different
fix (a CNN on ELA/noise maps).

**May go down:** `coordinated_reuse` is already at ~1.00, so there's no headroom
there, and a badly-tuned threshold in Step 3 could actively hurt it. If your
numbers drop after the swap, suspect the threshold before suspecting CLIP.

A modest gain is the realistic outcome. The fusion effect is coming from the
design rather than encoder strength — which is why it already held significantly
with a placeholder.

---

## Report back

Paste the `crossval.py` output and I'll write it up as Technical Report 03 with
the before/after comparison and the attribution of each change.
