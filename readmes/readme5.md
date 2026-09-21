# README 5: Removing the rating shortcut, the forensic CNN, retraining, and the working extension

**Owner:** Venkata Sai · **Date:** 19 September 2026 · **Previous:** `readme4.md`

## 0. What you asked for (in the messages after readme4)
1. "Even text-only is low. How can you say ours is better?" Then: **do steps 1 and 2**
   (remove the rating shortcut, build a stronger image module).
2. **Train it again and give the table again** (text only, text + image, …).
3. While training runs, **design and build the extension**, so a better model can be plugged in later.
4. **Show an image** of how it looks.
5. **First finish training, test on the test set and give the table, and only then check the extension.**

All done, in that order. (I briefly started the extension preview while the CNN was still training.
You stopped me, and I switched to finishing training first. I saved that as a working rule.)

---

## 1. Why text-only is "low" (your first question)
- **A third of our fakes are *designed* so no text model can catch them.** The 555 edited-photo fakes carry
  normal, genuine-sounding text. After step 1 below, they also have normal star ratings. Text-only
  therefore catches only **7%** of them. That is the point of the project, not a flaw.
- Our text block is **8 hand-counted features** (generic-word ratio, length, …), not BERT/RoBERTa.
  Upgrading it is guide §16 step 4, and it would raise text-only (and the baseline) honestly.
- I never meant "our text model beats the papers". It doesn't. What we show is that **adding image
  evidence catches fakes that text cannot**, and that is now much clearer (table below).

---

## 2. Step 1: Removing the rating shortcut
**Problem found in readme4:** every edited-photo fake had 4–5 stars and happy text, while genuine reviews
range from 1 to 5 stars. So star rating + time **alone** caught 74% of edited-photo fakes without ever
looking at the photo. That was a shortcut in the data generator, not forensic skill.

**Fix** (in `build_dataset.py`, now "v6.1"): edited-photo fakes draw their text, sentiment (positive / negative /
mixed), the 18% lazy text, and their star rating **exactly like genuine reviews**. Only the pixels differ.
The dataset was rebuilt (still exactly 1,650 / 1,650, balanced per category and split, all checks pass). CLIP,
forensics and the reuse threshold were redone. The threshold plateau is flat at 0.73–0.78; 0.749 was kept.

**Effect:** rating-only now catches 36% (down from 74%), and the no-image model catches only **4%** of edited photos.
The headline number dropped, which is correct: the shortcut had been inflating it.

---

## 3. Step 2: A CNN that looks at the photo's forensic maps
**Old image module:** 13 summary numbers per photo (average error level, and so on). It cannot see *where* an edit is.

**New:** `forensic_maps.py` turns every photo into three 128×128 maps (error level, error roughness,
noise energy). Each is computed at full resolution and then shrunk, and divided by the photo's own median.
`cnn_forensics.py` trains a small CNN (4 conv blocks, max + average pooling) to answer
"was this photo edited?" Its output becomes one new feature, `img_cnn_manip`.

**No leakage:** every CNN score is out-of-fold, grouped by root photo. For the final model, the CNN is
trained on **train photos only**, and the test photos are scored by a CNN that never saw them.

| How well it detects edited photos | ROC-AUC (unseen test photos) |
|---|---|
| Old: 13 hand-made ELA/noise numbers | 0.635 (copy-move 0.499 = chance) |
| **New: CNN on forensic maps** | **0.959** (copy-move 0.912 · splice 0.981 · noise 0.972) |

**Honest caveat (say this in the viva):** the same code that made the edits also provides the CNN's training labels,
so the CNN learns **our three edit types**. On real-world edits made with other tools it would be weaker.
Report it as a controlled result (guide §11 item 1).

---

## 4. THE TABLE: same model, different feature groups, same unseen test photos
Balanced test set (810 reviews, 50% fake). The PR-AUC column uses the realistic holdout (15% fake).
"Edited-photo caught" = share of the 135 edited-photo fakes flagged. "False alarms" = share of the 405 genuine reviews flagged.

### Before (v6.0: rating shortcut present, no CNN), from readme4
| Features | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited-photo caught | False alarms |
|---|---|---|---|---|---|---|
| Text only | 0.796 | 0.772 | 0.883 | 0.766 | 23% | 10% |
| Image only | 0.686 | 0.622 | 0.730 | 0.480 | 43% | 14% |
| Text + Image | 0.841 | 0.823 | 0.915 | 0.790 | 44% | 6% |
| No image | 0.833 | 0.837 | 0.935 | 0.846 | 62% ⚠ shortcut | 19% |
| ALL | 0.852 | 0.852 | 0.946 | 0.843 | 70% | 15% |

### After step 1 only (shortcut removed, no CNN)
| Features | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited-photo caught | False alarms |
|---|---|---|---|---|---|---|
| Text only | 0.779 | 0.742 | 0.796 | 0.692 | 7% | 8% |
| Image only | 0.694 | 0.624 | 0.746 | 0.512 | 41% | 12% |
| Text + Image | 0.815 | 0.783 | 0.868 | 0.770 | 27% | 4% |
| No image | 0.784 | 0.743 | 0.798 | 0.704 | 4% | 6% |
| ALL | 0.814 | 0.780 | 0.867 | 0.781 | 22% | 3% |

### ✅ After steps 1 + 2 (shortcut removed + CNN): the current model
| Features | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited-photo caught | Fake-text caught | False alarms |
|---|---|---|---|---|---|---|---|
| Text only | 0.779 | 0.742 | 0.796 | 0.692 | **7%** | 83% | 8% |
| Image only | 0.772 | 0.733 | 0.816 | 0.679 | **84%** | **9%** | 8% |
| Behaviour only | 0.619 | 0.613 | 0.656 | 0.377 | 36% | 59% | 37% |
| Graph only | 0.573 | 0.575 | 0.607 | 0.253 | 38% | 46% | 43% |
| **Text + Image** | 0.898 | 0.891 | 0.943 | 0.909 | 80% | 72% | 4% |
| No image (text + behaviour + graph) | 0.788 | 0.746 | 0.799 | 0.698 | 4% | 84% | 5% |
| **ALL (our fusion model)** | **0.909** | **0.903** | **0.948** | **0.918** | **81%** | 75% | **3.5%** |

**How to read it:**
- **Text only** misses edited photos (7%). **Image only** misses fake text (9%). Each is blind to what the other sees.
- **Together** they reach **0.909 accuracy**, with only 3.5% false alarms.
- **Removing the image drops accuracy from 0.909 to 0.788 and PR-AUC from 0.918 to 0.698.** The image part now clearly
  carries weight. That was not true before (readme4), and it answers your doubt.
- The CNN score is now the model's **most important feature** (13.5% of the forest's importance).

### Cross-validation (25 folds, grouped by photo, 15% fake): the numbers for the report
| | PR-AUC | ROC-AUC |
|---|---|---|
| text-only | 0.719 | 0.810 |
| image-forensic-only | 0.658 | 0.814 |
| base-paper-equivalent (text + behaviour + graph) | 0.729 | 0.817 |
| **FUSION (ours)** | **0.863 ± 0.013** | **0.933** |

**Fusion beats the base paper by +0.134 PR-AUC, p = 5e-14, and wins in all 25 of 25 folds.** (Before: +0.042.)

### Final model on the test set (`train_model.py`)
| Test set | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| test_balanced (50% fake) | 0.964 | 0.948 | 0.903 | 0.961 | 0.852 |
| **holdout_realistic (15% fake)** | **0.918** | 0.957 | 0.847 | 0.813 | 0.884 |

- **Fixed:** "legit stock-photo reuse" false alarms went from 3/12 to **0/12**.
- **Still open:** the 48 "Needs Verification" trap rows are all called Genuine. They are not flagged as fake (good),
  but the model does not abstain on them. The decision bands (0.35 / 0.65) still need tuning (guide §16 step 2).

---

## 5. The Chrome extension (built and tested)

### Files
```
D:\sem-5 projects\computer vision\
  extension\   manifest.json (Manifest V3), sites.js (where reviews are on Amazon / Flipkart / demo),
               content.js (reads reviews, draws badges), background.js (talks to backend),
               popup.html/.js (status + on/off), styles.css, icons\
  backend\     app.py (FastAPI: /health, /analyze, /demo), pipeline.py (raw review -> same 40
               features as training -> models -> decision + reasons), config.py, test_parity.py,
               requirements.txt, README.md (how to run)
```

### How a better model gets plugged in later (your question)
Yes, it will work. The backend loads whatever is in `dataset\fake_review_dataset_v6\out\model\`
(`fusion_rf.joblib`, `text_rf.joblib`, `image_rf.joblib`, `forensic_cnn.pt`, `model_meta.json`).
Train a better model, then restart the backend; the extension doesn't change. On start-up the backend
**refuses to run** if the model needs a feature it can't compute, so a mismatch can't slip through silently.

### Parity test: does the backend compute exactly what training computed?
`backend\test_parity.py` sends 12 held-out reviews through the backend as if scraped from a page.
- First run: 39/40 features identical. **`g_community_size` differed by 173.** The Louvain community detection
  depended on the order in which reviewers were added. **Fixed** in `features.py` (sorted, order-independent
  graph), the models were retrained, and now **40/40 features are identical**.

### Demo page check (14 held-out reviews the backend had never seen)
| True type | Badges shown |
|---|---|
| Genuine (5) | 5 × Genuine (**no false alarms**) |
| Coordinated reuse (3) | 3 × Likely fake ("4 matching photos from other accounts posted within 6 hours") |
| Fake text (3) | 2 × Likely fake, 1 × Needs verification |
| Edited photo (3) | 1 × Likely fake ("photo shows signs of editing"), 1 × Needs verification, 1 missed |

**The best example on the page:** a watch review with genuine-sounding text and 2 stars. The text model says 21%,
the photo model 94% (CNN 0.79), so the result is **Likely fake**. Text-only systems would miss this one.

I also fixed a wording problem found during the check. The reason text said "same photo posted by 27 accounts", but
that count includes photos that only *look similar* (CLIP). Now it only reports the real signal: matching photos
**posted within 6 hours**. For older look-alikes it says "(spread over time, can be normal)".

### How to see it yourself
Follow `backend\README.md`: start the backend, then load `extension\` in `chrome://extensions` → Load unpacked,
then open http://127.0.0.1:8000/demo. Without installing the extension, http://127.0.0.1:8000/demo?preview=1
shows the same badges.

**Not yet tested on a live Amazon page.** The Amazon selectors (`data-hook="review"` and so on) are standard, but
real pages need a check. Flipkart's selectors are marked UNTESTED in `sites.js`.

---

## 6. Honest list of what is still weak
1. The CNN learns our own three edit types (a controlled result, not real-world tamper detection).
2. Every genuine review also shows "a near-identical photo exists…", because the dataset reuses each root
   photo about 4 times. It's a dataset artifact, and harmless on real pages.
3. The "Needs Verification" band never triggers on the trap rows. The bands need tuning.
4. Text is still 8 hand-made features (RoBERTa is the next upgrade).
5. First analysis is slow (~30–60 s) because CLIP loads on first use. After that, a page takes a few seconds.

## 7. Next
1. Tune the Genuine / Needs Verification / Fake bands on validation folds.
2. Leave-one-category-out test (`loco.py`).
3. Try the extension on real Amazon product pages and fix the selectors if needed.
4. RoBERTa text features.
