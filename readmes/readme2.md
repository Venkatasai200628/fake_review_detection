# README 2: Removing the Claude co-author, pushing the v6 dataset, training the model, and the extension plan

**Owner:** Venkata Sai · **Date:** 19 September 2026
**Previous:** `readme1.md` (photo pool pushed, balanced dataset v6 built)

---

## 0. What you asked for in this prompt

1. **Remove, or don't add, the Claude contributor** on GitHub.
2. After the dataset is complete, **push everything to GitHub, including the new dataset**, and
   **put the new dataset inside the `dataset` folder** where your photos are.
3. **Move on to model training.** Explain how it will **predict new data** (not just training data),
   how we check **image and text** fakes, how **categories** are handled, and how the **Chrome extension**
   will be built and made to work.
4. Keep the readmes in a `readmes` folder in the project directory (`readme1`, `readme2`, …).
5. Explain the plan by **adding to the bottom of `IMPLEMENTATION_GUIDE.md`, removing nothing**.

All five are done. Details below.

---

## 1. The readmes folder

`D:\sem-5 projects\computer vision\readmes\` now has:
- `readme1.md`: pushing the photos + building the balanced dataset + CLIP (prompt 1)
- `readme2.md`: this file (prompt 2)
- `readme3.md`: the papers comparison (prompt 3)

Every future prompt gets the next number.

---

## 2. Removing the Claude co-author from GitHub

### What was wrong
The commit pushed in readme1 (`6543936`) ended with a line
`Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`. GitHub reads that line and shows Claude as a contributor.

### What I did
1. `git commit --amend` rewrote **only the message** of that commit, without the co-author line.
   The files inside did not change at all. The new commit id is `7eb2f02`.
2. `git push --force-with-lease=main:6543936`. A normal push is refused when history is rewritten,
   so a force-push is needed. `--force-with-lease` is the **safe** kind: it only overwrites GitHub if
   GitHub still has exactly `6543936`. If anything else had been pushed in between, it would stop instead of
   destroying it.
3. I checked GitHub afterwards: the whole history has **0** co-author lines.

### Going forward
- The new dataset commit (`6cc4abc`) has no co-author line.
- I saved this as a permanent rule in my memory for this project: **never add Claude attribution to your
  commits or pull requests.**
- Note: GitHub can take a little while to refresh the "Contributors" box on the repo page. The
  commit data itself is already clean.

---

## 3. Moving the v6 dataset into `dataset\` and pushing it

### What moved where
```
BEFORE:  D:\sem-5 projects\computer vision\fake_review_project_v6_balanced\{code, out}
AFTER:   D:\sem-5 projects\computer vision\dataset\fake_review_dataset_v6\{code, out}
```
`dataset\` is your GitHub repo, so the dataset now travels with the photos.
The Windows `mv` was blocked because a running shell had the folder open. So I **copied** it,
checked the copy was identical (`diff -r`: no differences), then removed the old one.

### One code change needed because of the move
`build_dataset.py` finds the photos with `REPO_DIR`. It was `'../../dataset'`; now it is `'../..'`
(two folders up from `code\` is the repo root). This also works if someone clones the repo under a
different folder name. I tested it: it still finds exactly **836 photos in 15 categories**.
The new `fake_review_dataset_v6` folder is **not** mistaken for a 16th product category, because it has no
photos directly inside it.

### What was pushed (commit `6cc4abc`)
| What | Size |
|---|---|
| 3,348 review images (`out/images/`) | ~351 MB |
| CSVs: `reviews_balanced`, `train`, `test_balanced`, `holdout_realistic`, `needs_verification_controls`, `reviews_full` | ~55 MB (largest file 17 MB, well under GitHub's 100 MB limit) |
| CLIP embeddings (`.npy`), forensic features, all logs and result CSVs | ~16 MB |
| Trained model `out/model/fusion_rf.joblib` + `model_meta.json` | 3.3 MB |
| All 14 Python scripts in `code/` | small |
| New repo `README.md` explaining the layout, and an updated `out/README.md` (column guide) | small |

**Not pushed** (they live outside the repo folder): `IMPLEMENTATION_GUIDE.md`, the `readmes\` folder,
`papers\`, and the old v5 folder. Tell me if you want copies of any of these in the repo too.

---

## 4. Model training (done): `train_model.py`

### What it does, simply
1. Turns every review into **39 numbers** (features): text, image, behaviour, graph.
2. Trains a **Random Forest** (500 trees) on the **train** photos only (2,490 reviews, 50/50).
3. Tests on photos it has **never seen**.
4. Saves the model to `out/model/fusion_rf.joblib`, with the feature list and thresholds in `model_meta.json`,
   so the extension backend can load exactly this model.

### Why a separate script from `crossval.py`?
`crossval.py` answers *"is fusion really better than the baseline?"* It trains 150 throw-away models across 25 folds.
`train_model.py` answers *"give me one model I can ship."* You need both.

### Results
| Test set | PR-AUC | ROC-AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| test_balanced (50% fake) | 0.952 | 0.946 | 0.852 | 0.852 | 0.850 | 0.854 |
| **holdout_realistic (15% fake)** | **0.843** | 0.950 | 0.846 | 0.610 | 0.483 | 0.826 |

- The 0.843 holdout PR-AUC **matches cross-validation (0.843)**. That agreement is a good sign that nothing is leaking.
- Precision is lower at 15% fake because there are far more genuine reviews to wrongly flag. This is normal, and
  it is why the "Needs Verification" band exists.
- **Per fraud type** (share flagged): coordinated reuse 100%, text deception 87%, edited photo 70%,
  genuine 15% (false alarms).
- **Three-way decision** (Genuine below 0.35, Needs Verification 0.35–0.65, Fake above 0.65): of 405 fake test reviews,
  294 are called Fake and 21 wrongly Genuine. Of 405 genuine reviews, 292 are called Genuine and 15 wrongly Fake.
  The rest go to amber.
- **Top features:** generic-word ratio, star rating, text length, sentence count, extreme rating,
  image reuse inside a burst, CLIP max similarity, duplicate text, burst ratio, specific-word ratio.

### Honest problem found
**3 of 12 "legit stock-photo reuse" reviews are flagged as Fake.** These are real buyers reusing the
manufacturer's photo over weeks. It is the known weakness from guide §11 item 4: the model still leans
towards "reuse = fake". It is listed as a fix-before-shipping item (guide §16 step 3).

---

## 5. The plan added to `IMPLEMENTATION_GUIDE.md` (Part 2, §15–§21)

I **only appended**. I checked that the first 817 lines are byte-for-byte identical to before (same MD5
checksum `d0101b7b…`). The new sections, in short:

| Section | Answers your question |
|---|---|
| §15 Status update | What is done now, with the v6 + CLIP + final-model numbers |
| §16 Model training | Three models (text, image, fusion). The fusion score sets the badge colour, and the text/image scores explain *why* |
| §17 **Predicting NEW data** | How a brand-new Amazon review becomes the same 39 features: download the photo → ELA/noise → compare with an **"image memory"** of photos seen before → CLIP photo-vs-text → behaviour from the page → graph from the page + memory. Rules to avoid training/serving skew. What to do with small images or missing info |
| §18 **Categories** | The model never sees the category, only evidence, so it works on any product. It is proven with leave-one-category-out (5 folds × 3 categories). New categories are handled by CLIP zero-shot (photo vs product title) |
| §19 **The Chrome extension** | Architecture diagram, file list, `manifest.json` essentials, how `content.js` finds Amazon/Flipkart review cards, the `/analyze` API, a 7-step build order where each step is testable, how to install ("Load unpacked"), privacy rules |
| §20 Updated order of work | What to do next, in order |
| §21 Literature comparison | Pointer to `papers\PAPERS_COMPARISON.md` (see readme3) |

### The three ideas to remember from Part 2
1. **Image and text are checked separately *and* together.** The text model and the image model each give a score,
   and the fusion model makes the decision. The per-type table proves each catches what the other misses.
2. **New data needs an "image memory".** Reuse can only be detected against photos seen before. The backend
   starts with our 3,348 images and adds every photo it sees.
3. **The extension is a thin shell.** Chrome reads the page and draws badges. All the thinking happens in a
   local FastAPI server on your laptop that uses the same `features.py` as training.

---

## 6. What should come next
1. Save text-only and image-only models too, and tune the decision bands (guide §16).
2. Fix the legit-stock-reuse false alarms (guide §11 item 4).
3. Write `loco.py`: the unseen-category test (guide §18).
4. Backend `pipeline.py`, with a unit test that it reproduces the training features exactly. Then `app.py`.
5. The Chrome extension (guide §19.6).
