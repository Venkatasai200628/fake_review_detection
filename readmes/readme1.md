# README 1: Pushing the photo pool to GitHub and building the balanced dataset

**Project:** Fake Review Detection via Visual Forensics (Computer Vision, Semester 5, BTech AI & DS)
**Owner:** Venkata Sai
**GitHub (photo pool):** https://github.com/Venkatasai200628/fake_review_detection
**Project folder:** `D:\sem-5 projects\computer vision`
**Date of this work:** 19 September 2026

This is the first readme in the `readmes/` folder. Each request you send gets
its own numbered readme (`readme1.md`, `readme2.md`, …), so you can follow the
whole project in order.

---

## 0. What you asked for in this prompt

1. Use the project context: `IMPLEMENTATION_GUIDE.md` and the PDF of your previous chat.
2. **Push the image dataset** (`D:\sem-5 projects\computer vision\dataset`) to GitHub.
3. **Build a new dataset from those images**, with **Fake and Genuine reviews balanced**.
4. Write a readme that explains everything in simple English: what was done, why, and
   the reasons. It goes in a `readmes` folder, numbered `readme1`, `readme2`, and so on.

All four are done. The rest of this file explains each step.

---

## 1. The project in simple words (a recap, so this file stands on its own)

### 1.1 The problem
Many product reviews on shopping sites are fake, because sellers pay people to write them.
Most detectors only read the **words** of a review. A skilled fake reviewer can
beat that by simply writing better.

### 1.2 Our idea
Many reviews include a **photo**, and almost nobody checks that photo properly. We treat
the photo as **evidence**, the way a detective treats a document, and ask three questions:

| Question | What it catches | Technique |
|---|---|---|
| **Has the photo been edited?** | pasted-in or cloned parts | ELA (Error Level Analysis) + noise-residual check |
| **Where did it come from?** | the same photo used on other accounts or products | pHash (perceptual hash) + CLIP image embeddings |
| **Who else used it?** | groups of accounts posting the same photo within a few hours | a "visual coordination graph" + Louvain communities |

We then combine those answers with the usual signals (review text, star rating, posting
time) into one decision: **Genuine**, **Fake**, or **Needs Human Verification**.

### 1.3 Why combining matters
A faker can fool one check but rarely all of them:
- Perfect text but an **edited photo**: the text-only model misses it, the image model catches it.
- A **clean real photo** but hollow text: the image model misses it, the text model catches it.
- Fusion (our model) should catch **both**. That is the main claim of the project.

### 1.4 Base paper and novelty
- **Base paper:** He et al. (2022), *Detecting Fake-Review Buyers Using Network Structure*, PNAS.
  They used review images only to ask "do this one product's photos look alike?"
- **Our novelty:** we use images as **forensic evidence across the whole site**. We check
  for edits, reuse across products and accounts, and account networks linked by shared
  photos, then fuse all of that with text and behaviour in a model that can say "not sure".
- **Why it is a Computer Vision project:** all the new work is on the image side.

### 1.5 Why we build our own dataset
No public dataset labels review photos as "edited" or "reused by a fake-review ring".
So we take **real product photos** (your GitHub pool) and **build reviews around them**.
We control exactly which reviews are honest and which are fake, and how each fake was
made. That control is what lets us prove each part works. It also means our numbers
describe a **controlled test, not live Amazon performance**, and we say so plainly.

---

## 2. Where the project was before today

(From `IMPLEMENTATION_GUIDE.md`.)

- Code v5 was in `fake_review_project_v5_clip_ready/`. It was built on the **old 197-photo pool** and produced about 1,974
  review rows, of which about **40% were Fake** (not balanced).
- Result: fusion PR-AUC **0.817** vs base-paper-equivalent **0.772** (p = 3.7e-06).
- **Pending items:**
  1. The "remove photos under 1000px" commit was stuck in a **git rebase** and was never pushed.
  2. More photos at 1000px or larger needed to be collected (target about 54 per category).
  3. The dataset needed rebuilding on the bigger, cleaner photo pool.

You had collected the new photos. Today's job was items 1 and 3, plus making the dataset balanced.

---

## 3. Step 1: What I found in your `dataset` folder

`D:\sem-5 projects\computer vision\dataset` is a **git repository** (a clone of your GitHub repo). It was in a messy state:

| Finding | Meaning |
|---|---|
| "interactive rebase in progress" | The earlier `git pull --rebase` never finished. Git was stuck halfway. |
| 421 new untracked files | Your newly collected photos, not yet committed |
| 209 deleted files | The small iStock / AVIF preview photos (below 1000px) you removed |
| 61 modified files | Photos you replaced with better versions (same file name) |
| `Handbag` → `Handbags` | You renamed that folder |
| `.gitignore` deleted, and saved in UTF-16 | PowerShell's `echo >>` writes UTF-16, which git cannot read properly |

### I checked every photo (all 836) before pushing
I opened every image with Python (Pillow) to confirm that none were broken and to measure
their resolution.

| Category | Photos | Category | Photos |
|---|---|---|---|
| Bluetooth speaker | 55 | Wireless_Earbuds | 56 |
| Handbags | 54 | Yoga mat | 56 |
| Knifes | 54 | backpack | 54 |
| Office cair | 60 | mouse | 55 |
| Phone case | 63 | skincare set | 54 |
| Shoes | 55 | suitcases | 56 |
| Sunglasses | 54 | Watches | 55 |
| Water bottle | 55 | **TOTAL** | **836** |

- 0 unreadable files, and no file over 50 MB (GitHub rejects files over 100 MB).
- **10 photos are below 1000px, all in `backpack/`** (the `istockphoto-…-612x612` and
  `photo-/premium_photo-….avif` files, 600–987px). I did **not** delete your files.
  The generator simply skips them (see §5.2). Every category reached your target of 54 photos or more.

---

## 4. Step 2: Pushing the photo pool to GitHub

### What I did
1. Rewrote `.gitignore` as normal UTF-8 text so git can read it. It ignores the scratch
   lists `_files_to_delete.txt`, `files_to_delete.txt`, and `__pycache__/`.
2. `git add -A` staged every addition, deletion and replacement.
3. Committed the change as **"Update root image pool: 836 photos across 15 categories (~54-63 each)"**.
4. `git rebase --continue` finished the stuck rebase, so `main` points at the new commit.
5. `git push origin main`.

### Why I did it this way (and not another way)
- My first plan was `git rebase --quit` and then reset the `main` branch. Claude Code's safety
  check **blocked** that, because resetting a branch can lose work. It was right to be careful.
- So I used the gentler, normal route: **make a new commit on top, then let the rebase finish itself**.
  Nothing was reset or thrown away. Your old local commit (`9ccc951`) is still in git's history (reflog).
- The new commit sits directly on top of what is already on GitHub, so the push is a plain
  **fast-forward**. No force-push was used, and no one else's work could be overwritten.

### Result
GitHub history (newest first):
```
7eb2f02  Update root image pool: 836 photos across 15 categories (~54-63 each)
e5cf3f1  Remove scratch files from dataset repo
5cd4735  Expand root image pool to ~42 photos per category
edb2c7c  Expand root image pool to ~42 photos per category
59d0deb  first commit
```
The repo now tracks 838 files: 836 photos, `.gitignore`, and `prune_small_images.py`.

*(The commit was first pushed as `6543936` with a "Co-Authored-By: Claude" line. At your request it was
reworded to `7eb2f02` without that line. See readme2 §2.)*

---

## 5. Step 3: Building the balanced dataset (v6)

### 5.1 Where it lives
```
D:\sem-5 projects\computer vision\
    dataset\                              <- Layer 1: the 836 real photos (GitHub)
    fake_review_project_v6_balanced\      <- NEW today (later MOVED to dataset\fake_review_dataset_v6\, see readme2)
        code\                             <- all Python scripts (run from here)
        out\                              <- the generated dataset
            images\                       <- 3,348 review photos (1024px JPEG, ~351 MB)
            reviews_balanced.csv          <- 3,300 rows, exactly 1,650 Genuine + 1,650 Fake
            train.csv                     <- 2,490 rows, exactly 50/50
            test_balanced.csv             <- 810 rows, exactly 50/50
            holdout_realistic.csv         <- 474 rows, ~15% Fake (for honest reporting)
            needs_verification_controls.csv  <- 48 "not sure" trap rows
            reviews_full.csv              <- everything in one file
            forensic_features.csv, loco_folds.json, build_summary.json, README.md, logs
    readmes\readme1.md                    <- this file
```
v5 was left untouched, so you can still compare against it.

### 5.2 What "balanced" means here: three levels

| Level | Genuine | Fake |
|---|---|---|
| Whole dataset | 1,650 | 1,650 |
| Train split | 1,245 | 1,245 |
| Test split | 405 | 405 |
| **Every one of the 15 categories, in every split** | **equal** | **equal** |

Each category has **220 rows (110 Genuine + 110 Fake)**. The code checks this with
`assert` statements, so the build fails if the balance is ever broken.

**Why balance per category and not just overall?** If Shoes were 80% fake and Watches 20%
fake, the model could learn "shoe = fake". Per-category balance removes that shortcut.
This follows rule 1 in your previous chat: *every category must contain both classes*.

### 5.3 The Fake half: three kinds of fake, one third each

| fraud_type | How many | What is fake | Which model should catch it | signal_cell |
|---|---|---|---|---|
| `visual_manipulation` | 555 | the **photo is edited** (copy-move, splice from another photo, or a noise patch); the text sounds real | image model | `image_only` |
| `text_deception` | 555 | **clean real photo**, but hollow and over-the-top text | text model | `text_only` |
| `coordinated_reuse` | 540 (105 campaigns of 4–6 accounts) | the **same photo** posted by many accounts within **3 hours**, with copy-paste text | both | `both` |

**Why one third each?** Your previous chat described a 2×2 grid: genuine, text-only fake,
image-only fake, and both. If every fake had both fake text and a fake photo, a text-only
model would score as well as fusion and there would be no result to show. The two
"only one modality can see it" cells are what **prove** that fusion is needed. The new
`signal_cell` column labels each row with its cell, so you can make that table directly.

### 5.4 The Genuine half, including deliberate "hard" cases
- **1,298 normal genuine reviews**: specific text naming a part, a duration and a situation,
  with realistic star ratings and times spread over about 180 days.
- **320 "lazy genuine" reviews** (`notes = OVERLAP_lazy_genuine`): real customers who write short,
  generic text like *"Good product, works fine."* Real people write like this too.
- **32 "legit stock-photo reuse" reviews** (`NEGATIVE_CONTROL_legit_stock_reuse`): many real buyers
  reuse the manufacturer's photo over weeks. The photo is reused, but they are **Genuine**. If the model
  flags these, it has wrongly learned that any reuse means fake.

On the fake side, **162 "sophisticated fakes"** (`OVERLAP_sophisticated_fake`) make up specific
details to sound real.

**Why add hard cases?** Without them, the text would be too easy (v2 scored a perfect 1.000 on
text alone, which made every comparison meaningless). They are not there to make the task
artificially hard. They remove an **artificial easiness**, because real reviews overlap like this.

### 5.5 "Needs Verification" rows are kept separate
48 rows (24 "only re-saved, nothing edited" and 24 "only 2 accounts share a photo") carry the label
`Needs_Verification`. They are the traps for the "not sure" band. They are **not part of
the 50/50 Genuine/Fake data**, because a balanced two-class dataset cannot hold a third
class. They live in `needs_verification_controls.csv` and in `reviews_full.csv`.

### 5.6 Every design decision and its reason

| Decision | Why |
|---|---|
| **Split train/test by root photo** (75% train / 25% test photos per category), decided **before** generating | If versions of the same photo are in both train and test, the model memorises the photo and scores about 98% for nothing. Deciding the split first means each split can be generated balanced, so no rows are thrown away. Result: **0 photos shared between train and test**. |
| **Every photo is used for both Genuine and Fake rows** (round-robin) | 824 of 826 photos appear in both classes, so "which photo is it?" can never predict the label. Only real evidence can. |
| **All images saved exactly the same way** (one JPEG save at quality 88, 1024px) | This avoids Leak 2 from v4, where fake images had one extra JPEG save and the "detector" just learned that. |
| **1024px output, 1000px minimum source** | Measured earlier: at 512px every edit detector is at chance (ELA 0.513). At 1024px ELA reaches 0.739. That is why the 10 small backpack photos are skipped. |
| **IDs numbered after shuffling** | In v5 all genuine rows came first (R00001–R01100), so the row number hinted at the label. Now the order means nothing. |
| **One reviewer-ID format (`U#####`) for everyone** | v5 used `C#####` for campaign accounts and `U#####` for others, which gave the answer away. Fixed. |
| **Star rating always agrees with the text** | Avoids the v1 bug of "Excellent suitcase" with 1 star, an accidental shortcut. |
| **Campaign posts inside 3 hours, genuine posts spread over 180 days** | Reuse alone is not fraud. The model has to use **reuse and timing together**, which is exactly the legit-reuse trap. |
| **Fixed random seed `20260919`**, with a separate seed for each row | Running `build_dataset.py` again gives exactly the same dataset, and the parallel rendering order cannot change anything. |
| **Parallel rendering (7 CPU workers)** | The whole dataset builds in **4.5 minutes**. |
| **v5 folder left untouched** | Old results stay reproducible for comparison. |

### 5.7 New columns (compared with v5)
`root_file` (which GitHub photo was used), `split`, `is_fake` (0/1), and `signal_cell`
(the 2×2 grid cell). The full column dictionary is in `out/README.md`, which also marks which columns
must **never** be given to a model (`fraud_type`, `signal_cell`, `campaign_id`,
`transform_chain`, `manipulation_score`). Those columns are the answer key.

---

## 6. Step 4: Checking the dataset (`validate.py`)

I extended `validate.py` with a 9th check for balance. All checks pass:

| # | Check | Result |
|---|---|---|
| 1 | pHash: copies of the same photo vs different photos | same-photo p95 = 18, different-photo p5 = 24, margin 6: **CLEAN** |
| 2 | Embedding separation | same 0.984 vs different 0.768 (placeholder; CLIP result in §8) |
| 3 | Manipulation ground truth vs threshold | clean rows ≤ 0.15, edited rows ≥ 0.65: OK |
| 4 | Negative controls | legit reuse all Genuine; recompress-only all below threshold: **OK** |
| 5 | Campaign timing | 105 campaigns, longest burst 2.95 h (target ≤ 3 h); genuine spread 180 days |
| 6 | Rating skew | fakes cluster at 5★ (1★ for negative fakes), genuine spread 1–5 |
| 7 | Text specificity | unique-text ratio: genuine 0.78, deceptive 0.31, campaign 0.14 |
| 8 | Split integrity | train 50.0%, test 50.0%, holdout 14.6% fake; **root overlap 0** |
| 9 | Balance | overall 1650 = 1650; **every category equal in train and in test** |

Full output: `out/validate_placeholder_log.txt`.

---

## 7. Step 5: Does the project still work on the new data? (cross-validation)

I ran `run_forensics.py` (a new, parallel version of the guide's one-line ELA/noise command, 3 minutes)
and then `crossval.py`. It uses the same protocol as the guide: **5 repeats × 5 folds, grouped by
root photo, test folds thinned to 15% fake, Random Forest**. This first run uses the
placeholder image embedding, so it can be compared like-for-like with v5.

### 7.1 Results (mean ± 95% CI over 25 folds)

| Configuration | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| text-only | 0.740 ±0.017 | 0.873 | 0.604 | 0.577 | 0.641 |
| image-forensic-only | 0.505 ±0.025 | 0.703 | 0.390 | 0.321 | 0.497 |
| behaviour-only | 0.401 ±0.021 | 0.767 | 0.406 | 0.279 | 0.751 |
| graph-only | 0.245 ±0.016 | 0.591 | 0.277 | 0.185 | 0.551 |
| base-paper-equivalent | 0.802 ±0.014 | 0.931 | 0.575 | 0.438 | 0.839 |
| **FUSION (ours)** | **0.821 ±0.014** | **0.938** | **0.618** | **0.484** | **0.858** |

**Paired comparison, fusion vs base paper:** +0.019 PR-AUC (95% CI +0.009 to +0.029),
t = 3.78, **p = 0.0009**, and fusion wins in **19 of 25 folds**. The gap is **statistically significant**.

### 7.2 Detection rate by type of fake (this table explains *why*)

| Configuration | visual_manipulation | text_deception | coordinated_reuse |
|---|---|---|---|
| text-only | **0.152** (misses it) | 0.792 | 1.000 |
| image-forensic-only | 0.376 | **0.175** (misses it) | 0.974 |
| base-paper-equivalent | 0.592 | 0.931 | 1.000 |
| **FUSION (ours)** | **0.707** | 0.878 | 1.000 |

**How to read it:**
- Text-only catches only **15%** of edited-photo fakes, because their text is written to sound real.
- Image-only catches only **17.5%** of hollow-text fakes, because their photo is clean.
- Fusion raises edited-photo detection from **0.592 to 0.707** over the base paper (**+19% relative**).
  That is where the overall gain comes from.
- The trade-off: fusion is a little lower on `text_deception` (0.878 vs 0.931). Report this honestly.

### 7.3 How v6 compares with v5 (honestly)

| | v5 (197 photos, 40% fake) | v6 (826 photos, 50/50) |
|---|---|---|
| Fusion PR-AUC | 0.817 ± 0.022 | **0.821 ± 0.014** |
| Base-paper PR-AUC | 0.772 | 0.802 |
| Gap | +0.045 | +0.019 |
| p-value | 3.7e-06 | 9.2e-04 |

- Fusion is about the same, with a **tighter confidence interval** (four times more photos, so the estimate is steadier).
- The **gap got smaller** because the baseline got stronger. With 50% fakes in training (not 40%), the
  behaviour features learned the rating pattern better: behaviour-only catches 78% of edited-photo fakes.
  The fusion advantage is still real and significant. It is just not as large as v5 suggested.
- These are still the **placeholder** embeddings. The CLIP version is in §8.

---

## 8. Step 6: Real CLIP image embeddings

The guide's next step (§12.1 item 4) was to replace the placeholder image descriptor with **real CLIP
embeddings** (ViT-B/32, laion2b), which were already downloaded on your laptop.

1. `recompute_embeddings.py` encoded all 3,348 images plus every review text: 7.5 min for images and 2.8 min for text,
   on CPU. I first added the new CSV files (`reviews_balanced`, `test_balanced`, `needs_verification_controls`)
   to its list, so every file got updated.
2. `tune_thresholds.py` on the new data: same-photo pairs average 0.945, different-photo pairs 0.385,
   separation +0.309 (**CLEAN**). The best threshold (max Youden's J) came out at **0.749**, exactly the value
   your guide recommended. Set in `features.py`: `COSINE_REUSE_THRESHOLD = 0.749`.
3. I ran cross-validation twice so you can see each change separately.

| Configuration (PR-AUC, 15% fake) | placeholder | CLIP | CLIP + cross-modal |
|---|---|---|---|
| text-only | 0.740 | 0.740 | 0.740 |
| image-forensic-only | 0.505 | 0.557 | 0.567 |
| base-paper-equivalent | 0.802 | 0.801 | 0.801 |
| **FUSION (ours)** | 0.821 | 0.841 | **0.843 ± 0.012** |
| gap, p-value, folds won | +0.019, p=9e-04, 19/25 | +0.040, p=6e-08, 24/25 | **+0.042, p=3e-08, 24/25** |

**What it means:** CLIP roughly **doubled** the fusion advantage, back to v5's size (+0.045), now with
a tighter confidence interval. The cross-modal "does the photo match the text?" feature adds only
+0.002, so it is not yet a strong signal (worth reporting honestly). Per type, fusion catches 71% of
edited-photo fakes vs 15% for text-only, and 86% of text fakes vs 19% for image-only.

Logs: `out/crossval_placeholder_log.txt`, `out/crossval_clip_no_crossmodal_log.txt`, `out/crossval_clip_log.txt`.

*(Later in this session, see readme2: the v6 folder was moved into `dataset\fake_review_dataset_v6\`
and pushed to GitHub, and a final model was trained.)*

---

## 9. How to re-run everything

Open PowerShell and run:
```powershell
cd "D:\sem-5 projects\computer vision\dataset\fake_review_dataset_v6\code"
python build_dataset.py          # 4-5 min, builds out\ (same result every time)
python run_forensics.py          # 3 min, ELA + noise features
python validate.py               # 9 integrity checks, all should say OK/CLEAN
python crossval.py               # 3 min, the reported numbers
python recompute_embeddings.py   # ~25 min on CPU, swaps in CLIP embeddings
python tune_thresholds.py        # pick COSINE_REUSE_THRESHOLD for CLIP, then edit features.py
python crossval.py               # numbers with CLIP
```
Libraries needed (already installed on your laptop): `pillow imagehash numpy pandas scipy scikit-learn networkx opencv-python open_clip_torch torch`.

---

## 10. Things to know / honest weaknesses

1. **10 backpack photos are below 1000px.** They are on GitHub but skipped by the generator.
   Backpack still has 44 usable photos. Replace them with ≥1000px photos when you can.
2. ~~The generated dataset is not on GitHub.~~ **Update (readme2):** at your request it was moved into
   `dataset\fake_review_dataset_v6\` and pushed to GitHub together with its code.
3. **The rating pattern is a strong clue.** Every edited-photo fake is 4–5★, while genuine reviews spread 1–5★.
   That is a design choice (fakes cluster at extremes, which is well documented), but it is why
   behaviour-only does so well. Mention it if asked.
4. **Graph-only is weak** (ROC-AUC 0.591). The coordination graph adds context but is not a detector on its own.
5. **Provenance is partly circular**: the same code makes the copies and finds them. Present it as a
   controlled check, not a headline performance number.
6. **Text is template-based and behaviour is simulated.** This is a controlled test, not real Amazon data.
7. **Never compare 0.821 with other papers' numbers** (e.g. KE-MLLM 0.943). They use different data.
   The valid comparison is our own ablation table: same data, same folds, one change at a time.

---

## 11. What should come next

1. Evaluate the **cross-modal CLIP feature** (does the photo match the text?) as its own ablation.
2. Check the two **"Needs Verification" controls** against the trained model.
3. Run the **leave-one-category-out** test using `out/loco_folds.json`. This is the generalisation claim.
4. Replace hand-made text features with **RoBERTa/BERT**. This raises the baseline, which is the honest thing to do.
5. Build the **Chrome extension** (Manifest V3 + FastAPI backend + green/amber/red badges) last.

---

## 12. Mini glossary

- **Root photo**: one of the 836 real photos. Many review images are made from each one.
- **pHash**: a 64-bit "fingerprint" of a photo. Similar photos have similar fingerprints.
- **ELA**: re-save the JPEG and see which parts change differently. Edited areas stand out.
- **CLIP**: a pretrained AI model that turns an image (or text) into 512 numbers describing its content.
- **PR-AUC**: how well the model ranks fakes above genuines. It is the best metric when fakes are rare (15%).
- **Grouped cross-validation**: testing 25 times on different photos the model never saw.
- **Negative control**: a trap row that looks suspicious but is genuine, to catch lazy shortcuts.
- **Leakage**: any way the answer sneaks into the model other than real evidence.
