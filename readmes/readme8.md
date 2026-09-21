# README 8: Why individual < combined (proof), stronger models, dataset requirements proof, final table

**Owner:** Venkata Sai · **Date:** 19 September 2026 · **Previous:** `readme7.md`

## 0. What you asked
1. "Individual models give less but combined gives more, how? I'm not OK with that."
2. "We use CLIP. Why not check other models that are good for our project and give better values?"
3. "Does the dataset satisfy all the required conditions? Prove it."
4. "After it finishes, give me the final table again."

---

## 1. Why individual models MUST score lower (a proof, not an opinion)
The test set has 405 genuine + 405 fake reviews. The fakes are 135 edited-photo, 135 fake-text and 135 coordinated.
- The 135 **edited-photo** fakes have **genuine text by design**, so a text-only model *cannot* catch them.
  The best any text model could ever do is catch the other 270 fakes with zero mistakes:
  accuracy = (405 + 270) / 810 = **0.833**. That is the ceiling.
- The same holds for image-only: the 135 **fake-text** fakes have a clean, real photo, so its ceiling is also **0.833**.
- Only the combination can go above 0.833.

| | Ceiling | Before (readme5) | Now |
|---|---|---|---|
| Text only | 0.833 | 0.779 (93%) | **0.811 (97% of ceiling)** |
| Image only | 0.833 | 0.772 (93%) | 0.765 (92%) |
| All combined | 1.000 | 0.909 | **0.943** |

So the individual models are **not weak**; they are close to the maximum possible for a single view. That gap is exactly what the project sets out to show.

---

## 2. Stronger models: what I tried, and what won
Every candidate was trained on the same training photos and tested on the same unseen test photos.
Code: `code/compare_text_models.py`, `code/compare_image_models.py`. Results: `out/model_comparison/`.

### 2.1 Text (ceiling 0.833)
| Model | Accuracy | % of ceiling | Fake-text caught | False alarms | Time |
|---|---|---|---|---|---|
| Our 8 hand-made features (before) | 0.775 | 93% | 83% | 8.1% | 2 s |
| TF-IDF + Logistic Regression | 0.820 | 98% | 100% | 4.0% | 1 s |
| **MiniLM-L6 embeddings + LogReg ✅ chosen** | 0.822 | 99% | 99% | **3.2%** | 37 s |
| MPNet-base embeddings + LogReg | 0.811 | 97% | 100% | 5.9% | 106 s |
| DistilRoBERTa fine-tuned | 0.833 | 100% | 100% | 0% | 18 min |

**Why MiniLM and not DistilRoBERTa?** DistilRoBERTa hits the ceiling, but our review text comes from templates, and a
fine-tuned transformer memorises them (even the "sophisticated" fakes reuse phrases the genuine templates never use).
MiniLM gets 99% of the ceiling with **frozen** embeddings (it memorises less), is 90 MB, needs no fine-tuning per fold,
and is fast enough for the extension. DistilRoBERTa is reported as the upper bound.
(The message "You should probably TRAIN this model on a down-stream task" is a normal Hugging Face notice printed
*before* fine-tuning. The script then fine-tuned it for 3 epochs.)

### 2.2 Photo-reuse fingerprint (is this the same photo?)
| Model | AUC vs same-category photos | Reposts caught at 0.1% false positives |
|---|---|---|
| CLIP ViT-B/32 (before) | 0.9982 | 83.0% |
| DINOv2 ViT-S/14 | 0.9992 | 92.2% |
| DINOv2 ViT-B/14 | 0.9988 | 89.8% |
| **ResNet-50 ImageNet ✅ chosen** | **0.9995** | **94.7%** |

The new threshold is 0.72 (catches 99.97% of reposts, false matches 0.15% vs CLIP's 0.39%).
**CLIP is kept** for the photo-vs-text check, because only CLIP relates images to text.

### 2.3 Edit detector on forensic maps
| Model | ROC-AUC (unseen photos) | Copy-move | Splice | Noise | Edited photos caught | Size |
|---|---|---|---|---|---|---|
| **Our small CNN ✅ kept** | **0.953** | **0.907** | 0.971 | **0.970** | **86%** | 0.2 M params |
| ResNet-18 pretrained | 0.942 | 0.904 | 0.973 | 0.940 | 80% | 11.2 M |
| EfficientNet-B0 pretrained | 0.944 | 0.883 | 0.976 | 0.959 | 76% | 4.0 M |

The ImageNet-pretrained networks were **not** better. ImageNet teaches objects, while forensic maps are about
compression and noise patterns, so our small purpose-built CNN wins and is 20–50× smaller.

---

## 3. Proof the dataset meets the requirements
`code/verify_requirements.py` checks every requirement from guide §6 and the earlier project definition on the actual files.
Full report: **`out/REQUIREMENTS_CHECK.md`**.

**Result: 39 PASS · 4 WARN · 1 FAIL · 5 INFO**

| Area | What was proven |
|---|---|
| Schema | every agreed column present; labels exactly Genuine / Fake / Needs_Verification; 0 label mismatches; every image exists |
| Balance | 1,650 = 1,650; equal in every category of train and test; fake types 33.6 / 33.6 / 32.7%; all 4 cells of the 2×2 grid present |
| Split | 0 photos shared between train and test; train 50% fake; holdout 14.6% fake with all 3 types |
| Leakage probes | row order, review_id, image_id each predict the label at AUC 0.479 (coin flip); every category exactly 50.0% fake; 822/826 photos used for both classes; the model never sees answer-key columns |
| Images | one JPEG pass with identical quality tables for every image (no compression shortcut); file size alone AUC 0.49 |
| Fake-type design | only edited-photo fakes have edited photos; their star ratings match genuine reviews (chi-square p = 0.33); campaigns 4–6 accounts within ≤ 2.9 h; 0 rating/text contradictions in 3,300 reviews; overlap cases 16.8% / 32.8% |
| Negative controls | all three present and correctly labelled; legit stock reuse flagged 0/12 |

**WARN / FAIL (honest):**
- WARN: campaign accounts post ~2 fakes each and never a genuine review (reviewer identity AUC 0.574; 0.497 without campaigns). Realistic, but simplified.
- WARN: 14 of 3,348 images are slightly below 1024px (smallest 972px).
- WARN: backpack has 44 usable photos (target 54).
- WARN + **FAIL**: of the 48 borderline trap reviews, **1 is called Fake and none go to "Needs verification"**.
  The decision thresholds (0.35 / 0.65) are still untuned (guide §16 step 2). This is the next fix.
- INFO: four mechanisms from the earlier chat (stolen seller photo labelled fake, AI-generated photos, text–image mismatch,
  incentivised wording) are **not** in the dataset. The guide's §6.3 spec replaced them with the three types above.

---

## 4. ✅ FINAL TABLE (same unseen test photos, 50% fake; last three columns = share flagged)
| Features | Accuracy | F1 | ROC-AUC | PR-AUC @15% fake | Edited-photo | Fake-text | False alarms |
|---|---|---|---|---|---|---|---|
| Text only (8 + MiniLM) | 0.811 | 0.785 | 0.838 | 0.752 | 9% | **99%** | 6.9% |
| Image only (CNN + ResNet-50 + CLIP) | 0.765 | 0.726 | 0.812 | 0.657 | **85%** | 5% | 9.1% |
| Behaviour only | 0.619 | 0.613 | 0.656 | 0.377 | 36% | 59% | 36.8% |
| Graph only | 0.554 | 0.549 | 0.595 | 0.228 | 37% | 39% | 43.5% |
| **Text + Image** | **0.944** | **0.944** | **0.977** | 0.955 | 83% | 98% | 4.7% |
| No image (text + behaviour + graph) | 0.821 | 0.787 | 0.815 | 0.747 | 2% | 97% | 2.0% |
| **ALL (our fusion model)** | **0.943** | **0.943** | **0.976** | **0.958** | 82% | 98% | 4.7% |

**Improvement over readme5:** accuracy 0.909 → **0.943**, F1 0.903 → **0.943**, PR-AUC @15% 0.918 → **0.958**.

**Honest reading:** text + image alone gives almost exactly the full result. Behaviour and graph features add
nothing on top here, because campaigns are already caught 100% by image reuse plus text. They matter in the base paper,
but in our dataset the image and text evidence carry the model.

### Cross-validation (25 folds, 15% fake), the number for the report
| | PR-AUC | ROC-AUC |
|---|---|---|
| text-only | 0.758 | 0.827 |
| image-forensic-only | 0.655 | 0.814 |
| base-paper-equivalent | 0.758 | 0.827 |
| **FUSION (ours)** | **0.902 ± 0.012** | **0.954** |

**+0.144 over the base paper, p = 2e-16, better in 25 of 25 folds.**
(The base-paper configuration also got the new MiniLM text, so this is a fair comparison against a stronger baseline.)

### Final model on the realistic holdout (15% fake)
PR-AUC **0.958** · ROC-AUC 0.980 · F1 0.857 · precision 0.776 · recall **0.957** (66 of 69 fakes caught).
Most important features: MiniLM text score (18.4%), CNN edit score (14.0%).

---

## 5. Extension
The backend now computes the MiniLM text score and the ResNet-50 reuse fingerprint with the **same code** as training.
The parity test shows **41/41 features identical**. The extension itself did not need any change.

## 6. Next
1. **Tune the decision bands** on validation folds, which fixes the 1 FAIL and makes borderline cases go to "Needs verification".
2. Leave-one-category-out test.
3. Try the extension on a live Amazon page.
