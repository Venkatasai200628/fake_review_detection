# Layer 2 — balanced fake-review dataset (v6)

Built by `../code/build_dataset.py` from the Layer 1 photo pool
(`D:\sem-5 projects\computer vision\dataset`, pushed to
https://github.com/Venkatasai200628/fake_review_detection).

Full explanation: `D:\sem-5 projects\computer vision\readmes\readme1.md`.

## Files

| File | What it is | Genuine : Fake |
|---|---|---|
| `reviews_balanced.csv` | every Genuine + Fake row | exactly 50 : 50 |
| `train.csv` | rows from the 75% training root photos | exactly 50 : 50 |
| `test_balanced.csv` | rows from the 25% held-out root photos | exactly 50 : 50 |
| `holdout_realistic.csv` | test rows with Fake thinned to ~15% | ~85 : 15 |
| `needs_verification_controls.csv` | abstain-band traps (label `Needs_Verification`) | — |
| `reviews_full.csv` | all of the above in one file | — |
| `images/IMG#####.jpg` | one 1024px review photo per row | — |
| `loco_folds.json` | leave-one-category-out folds (5 folds × 3 categories) | — |
| `build_summary.json` | exact counts | — |
| `forensic_features.csv` | ELA + noise-residual features (from `run_forensics.py`) | — |
| `clip_image_embeddings.npy`, `clip_text_embeddings.npy`, `clip_cross_modal.csv` | CLIP ViT-B/32 vectors (the `embedding` column is CLIP too) | — |
| `model/fusion_rf.joblib` + `model_meta.json` | the trained fusion model (`train_model.py`) and its feature list / thresholds | — |
| `*_log.txt`, `crossval_*.csv` | outputs of every run (build, validate, cross-validation, CLIP, training) | — |

Genuine = Fake holds **inside every category of every split**, not just overall.

## Columns

| Column | Meaning | Use as a model input? |
|---|---|---|
| `review_id` | R#####, numbered after shuffling (order means nothing) | no (id) |
| `product_id` | one product per category, `P_<category>` | via graph features |
| `category` | 15 product categories | for LOCO only |
| `reviewer_id` | U#####, one namespace for all accounts | via behaviour/graph features |
| `review_text` | the review | yes |
| `rating` | 1–5 stars, always agrees with text polarity | yes |
| `timestamp` | ISO time of posting | yes |
| `image_file` | path of the review photo, relative to this folder | yes (pixels) |
| `image_id` | IMG##### | no (id) |
| `root_image_id` | which real photo this was derived from | **group for splitting — never a feature** |
| `root_file` | that photo's path inside the GitHub repo | no |
| `split` | `train` / `test` (decided by root photo) | no |
| `label` | `Genuine` / `Fake` / `Needs_Verification` | **target** |
| `is_fake` | 1 if label is Fake, else 0 | **target** |
| `fraud_type` | genuine / visual_manipulation / text_deception / coordinated_reuse / borderline_* | **NEVER — it is the answer** |
| `signal_cell` | none / image_only / text_only / both / weak_image | **NEVER — for per-type analysis** |
| `campaign_id` | CAMP### for coordinated rows | **NEVER** |
| `transform_chain` | which edits the generator applied | **NEVER** |
| `manipulation_score` | generator's ground truth for "how edited" | **NEVER (this was Leak 1)** |
| `phash` | 64-bit perceptual hash of the photo | yes |
| `embedding` | image embedding (placeholder 64-d, or 512-d CLIP after `recompute_embeddings.py`) | yes |
| `notes` | OVERLAP_* and NEGATIVE_CONTROL_* markers | no |

## Rules

1. Split by `root_image_id` only. A random row split lets the model recognise the photo and scores ~98% for nothing.
2. Never feed `fraud_type`, `signal_cell`, `campaign_id`, `transform_chain` or `manipulation_score` to a model.
3. Report PR-AUC / F1 / precision / recall on `holdout_realistic.csv` (or the 15%-thinned CV folds), not plain accuracy.
