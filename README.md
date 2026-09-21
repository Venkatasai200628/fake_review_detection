# Fake Review Detection via Visual Forensics — image pool + generated dataset

Computer Vision course project (Semester 5), Venkata Sai.

## What is in this repository

Reorganised 21 Sep 2026: the 15 product-photo folders were moved into `source_photos/`, and the
extension, backend, papers and write-ups were brought in from outside the repo, so everything now
lives in one place.

| Path | What it is |
|---|---|
| `source_photos/` | **Layer 1 — the root photo pool.** 836 real product photos in 15 category folders (826 usable at >=1000px). |
| `fake_review_dataset_v6/out/` | **Layer 2 — the balanced review dataset** built from those photos: 3,300 reviews, exactly 1,650 Genuine + 1,650 Fake (balanced in every category), plus 48 "Needs Verification" controls, 3,348 review images, CLIP embeddings, forensic features, and the trained model. Column guide: `fake_review_dataset_v6/out/README.md`. |
| `fake_review_dataset_v6/code/` | Code that builds, checks and evaluates the dataset (`build_dataset.py`, `train_model.py`, `loco.py`, `full_metrics.py`, …). |
| `backend/` | The local FastAPI server the browser extension talks to. Loads the trained model and scores real reviews. `python -m uvicorn app:app --host 127.0.0.1 --port 8000` |
| `extension/` | The Chrome extension (Manifest V3). Load it unpacked from this folder. Supports Amazon, Flipkart and Meesho. |
| `papers/` | The 40 surveyed papers and `PAPERS_COMPARISON.md`, including the seven re-built models run on our data. |
| `readmes/` | `readme1.md` … `readme13.md`, one per work session, in plain English. |
| `docs/` | Project spec, dataset datasheet, forensics technical report, CLIP swap runbook. |
| `IMPLEMENTATION_GUIDE.md` | The master document: design, decisions, corrections and every experiment. |

## Quick start

```bash
# 1. the backend (keep it running)
cd backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000

# 2. Chrome -> chrome://extensions -> Developer mode -> Load unpacked -> the extension/ folder
# 3. open any Amazon / Flipkart / Meesho product page, or http://127.0.0.1:8000/demo
```

## Headline result

Extra Trees fusion of 41 features: **accuracy 0.932, F1 0.929, MCC 0.868, ROC-AUC 0.967**, false
alarms 2.0%. Best of seven re-built published models on the same data: 0.835 / MCC 0.709.
On a product category never seen in training: accuracy 0.889, MCC 0.789.
Numbers, caveats and how they were checked: `readmes/readme13.md` and `IMPLEMENTATION_GUIDE.md`.

## Files that are not in git

Large and regenerable, so they are gitignored:
`out/forensic_maps.npy` (rebuild with `python forensic_maps.py`), the feature cache, and
`__pycache__`. Older project versions (v1-v5, zips) are kept outside the repo in `../archive/`.

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
python forensic_maps.py       # 128x128 ELA/noise maps (not in git: 330 MB, rebuilt in ~2 min)
python cnn_forensics.py       # forensic CNN (~2.5 h on CPU)
python reuse_embedding.py     # ResNet-50 reuse fingerprints
python text_model.py          # MiniLM text score
python crossval.py            # fusion PR-AUC 0.902 vs base-paper-equivalent 0.758 (p = 2e-16)
python train_model.py         # final model: holdout PR-AUC 0.958
python verify_requirements.py  # requirements proof -> out/REQUIREMENTS_CHECK.md
python modality_ablation.py   # text-only / image-only / text+image / ... table
```

The reviews are **constructed** around real photos, so the results describe a controlled test, not live platform performance.
