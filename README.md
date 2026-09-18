# Fake Review Detection via Visual Forensics — image pool + generated dataset

Computer Vision course project (Semester 5), Venkata Sai.

## What is in this repository

| Path | What it is |
|---|---|
| `Bluetooth speaker/` … `suitcases/` (15 folders) | **Layer 1 — the root photo pool.** 836 real product photos, 54–63 per category. |
| `fake_review_dataset_v6/out/` | **Layer 2 — the balanced review dataset** built from those photos: 3,300 reviews, exactly 1,650 Genuine + 1,650 Fake (balanced in every category), plus 48 "Needs Verification" controls, 3,348 review images, CLIP embeddings, forensic features, and the trained model. Column guide: `fake_review_dataset_v6/out/README.md`. |
| `fake_review_dataset_v6/code/` | Code that builds, checks and evaluates the dataset (`build_dataset.py`, `validate.py`, `crossval.py`, `train_model.py`, …). |
| `prune_small_images.py` | Lists or removes photos whose long edge is under 1000px. |

## Main files

- `reviews_balanced.csv`: all 3,300 Genuine/Fake rows (50/50)
- `train.csv` (2,490 rows, 50/50) and `test_balanced.csv` (810 rows, 50/50). They are split **by root photo**, so no photo is in both.
- `holdout_realistic.csv`: 474 test rows at ~15% fake, for honest reporting

Fake rows come in three equal kinds: `visual_manipulation` (edited photo, genuine-sounding text),
`text_deception` (clean photo, deceptive text) and `coordinated_reuse` (the same photo posted by many accounts within 3 hours).

## Rebuild / re-evaluate

```
cd fake_review_dataset_v6/code
python build_dataset.py      # rebuilds out/ (same seed, same result)
python run_forensics.py
python recompute_embeddings.py
python validate.py
python crossval.py           # fusion PR-AUC 0.843 vs base-paper-equivalent 0.801 (p = 3e-08)
python train_model.py
```

The reviews are **constructed** around real photos, so the results describe a controlled test, not live platform performance.
