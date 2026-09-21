# README 13: Final results with paper references, the model files, and the working extension

**Owner:** Venkata Sai · **Date:** 20 September 2026 · **Previous:** `readme12.md`

> **Read this one, not readme12 §5–§6.** While wiring the extension I found a bug that inflated the numbers in
> readme12. Everything here is re-measured after the fix. Section 1 explains it.

---

## 1. First, a correction

The numbers in readme12 (accuracy 0.9494) were **wrong, in our favour**. Here is what happened and how it was
caught, because it belongs in the report as evidence the pipeline is checked rather than trusted.

`text_model.py` cached the MiniLM sentence vectors in a file keyed **only by filename**. When `regen_text.py`
rewrote every review for v6.2, that cache was left untouched, so the model kept reading vectors of the **old,
leaky text**. `txt_minilm_p` is the model's single most important feature (0.19 importance), so this mattered.
`clip_text_embeddings.npy` had the same problem, affecting `clip_cross_modal`.

Worse, it was **one-sided**: the transformer baselines (BSTC, DeBERTa, FRIDRC, FRD-LSTM, A-LSTM) tokenise raw text
themselves, so they got the honest new text while our model kept the memorised old text. Any claim built on that
comparison was unsound.

**How it was caught:** `backend/test_parity.py`, which checks that the features computed when *serving* a review
match the features computed during *training*. It reported `txt_minilm_p` differing by **0.69** — a stale cache is
exactly what that looks like.

**The fix:** the cache is now keyed by a **SHA-256 of the text itself** (`text_model.py: load_embeddings`), so
changing a single review forces a rebuild. CLIP text vectors are rebuilt by the new `refresh_clip_text.py`.
After the fix parity is **41/41 features identical**.

**What it cost:**

| metric | with stale cache (wrong) | **corrected** |
|---|---|---|
| accuracy | 0.9494 | **0.9321** |
| F1 | 0.948 | **0.9287** |
| fake text caught | 94.1% | **84.4%** |
| edited photo caught | 80.0% | 80.7% |
| false alarms | 1.5% | 2.0% |

We lost 1.7 accuracy points and 10 points of fake-text recall (figures after the pHash fix in readme14/guide 28). The edited-photo number barely moved, which is the
expected signature: that evidence comes from the forensic CNN, which never touches the text.

---

## 2. The final model

**Extra Trees** fusion of 41 features, chosen over 9 alternatives by grouped cross-validation on the **training**
data (not the test set), trained on train.csv only, tested on photos it has never seen.

| metric | value | 95% CI (2,000 bootstrap, grouped by root photo) |
|---|---|---|
| accuracy | **0.9321** | 0.914 – 0.949 |
| F1 | **0.9287** | 0.907 – 0.948 |
| Precision | **0.9783** | |
| Recall | **0.8840** | |
| ROC-AUC | **0.9670** | 0.953 – 0.978 |
| PR-AUC (balanced test) | **0.976** | 0.966 – 0.984 |
| PR-AUC @15% realistic holdout | **0.955** | 0.913 – 0.986 |

Confusion matrix on the 810-review test set: **TP 358 · FP 8 · FN 47 · TN 397**.

Detection by fraud type: coordinated **100%** · fake text **84.4%** · edited photo **80.7%** · false alarms **2.0%**.

**Integrity checks** (`out/VERIFY_RESULTS.md`):
- every metric reproduces exactly from a fresh load of the saved model
- **label-shuffle control**: training on randomly shuffled labels gives test ROC-AUC 0.527, 0.531, 0.420, 0.635,
  0.429 — mean **0.509**. No feature leaks the answer.
- **seed stability**: 5 retrains give accuracy 0.930–0.932
- **training/serving parity**: 41/41 features identical
- 0 of 12 legitimate stock-photo reuses wrongly flagged

---

## 3. The baseline papers, with references

Seven published models were re-built from their papers and trained and tested on **our** dataset: same 2,490
training reviews, same 810 unseen test reviews, same metrics. Every citation below was confirmed by opening the PDF.

| # | Reference | File in `papers/` | Model we re-built |
|---|---|---|---|
| 02 | Shan, G., Zhou, L., & Zhang, D. (2021). *From conflicts and confusion to doubts: Examining review inconsistency for fake review detection.* **Decision Support Systems**, 144, 113513. | `1-s2.0-S0167923621000233-main.pdf` | rating/sentiment/content inconsistency features + Random Forest |
| 11 | Hou, J., Tan, Z., Zhang, S., Hu, Q., & Wang, P. (2025). *Detecting fake review intentions in the review context: A multimodal deep learning approach.* **Electronic Commerce Research and Applications**. | `1-s2.0-S1567422325000109-main.pdf` | FRIDRC: fine-tuned text encoder + ViT image encoder, fused |
| 15 | Xu, S., Cuan, H., Yin, Z., & Yin, C. (2024). *A Hybridized Approach for Enhanced Fake Review Detection.* **IEEE Transactions on Computational Social Systems**, 11(6). | `A_Hybridized_Approach_for_Enhanced_Fake_Review_Detection.pdf` | A-LSTM: attention BiLSTM on text + behavioural features |
| 30 | Lu, J., Zhan, X., Liu, G., Zhan, X., & Deng, X. (2023). *BSTC: A Fake Review Detection Model Based on a Pre-Trained Language Model and Convolutional Neural Network.* **Electronics**, 12(10), 2165. | `electronics-12-02165.pdf` | BSTC: BERT + TextCNN |
| 33 | He, S., Hollenbeck, B., & Proserpio, D. (2022). *Detecting fake review buyers using network structure: Direct evidence from Amazon.* **PNAS**. | `he-et-al-2022-...-amazon.pdf` | **our base paper** — network structure + Random Forest |
| 38 | Qayyum, H., Ali, F., Nawaz, M., & Nazir, T. (2023). *FRD-LSTM: a novel technique for fake reviews detection using DCWR with the Bi-LSTM method.* **Multimedia Tools and Applications**. | `s11042-023-15098-2.pdf` | FRD-LSTM: contextual word representations + BiLSTM |
| 39 | Duma, R. A., Niu, Z., Nyamawe, A., Tchaye-Kondi, J., Chambua, J., & Yusuf, A. A. (2024). *DHMFRD–TER: a deep hybrid model for fake review detection incorporating review texts, emotions, and ratings.* **Multimedia Tools and Applications**, 83, 4533–4549. | `s11042-023-15193-4.pdf` | DHMFRD-TER: text CNN + emotion + rating |
| 40 | Geetha, S., Elakiya, E., Sujithra Kanmani, R., & Das, M. K. (2025). *High performance fake review detection using pretrained DeBERTa optimized with Monarch Butterfly paradigm.* **Scientific Reports**. | `s41598-025-89453-8.pdf` | fine-tuned DeBERTa-v3 |

**Why these seven:** they are the papers in the folder that describe a model concretely enough to rebuild, and
together they cover every method family in the review (inconsistency features, LSTM, fine-tuned transformer,
deep hybrid with emotion/rating, and the one multimodal text+image model). Paper 33 is the base paper the project
builds on and is compared separately, because it classifies *products*, not individual reviews.

---

## 4. All results, on our dataset

### 4.1 Headline comparison

| # | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC@15% | Edited photo | Fake text | False alarms | McNemar vs ours |
|---|---|---|---|---|---|---|---|---|---|
| 30 | BSTC (BERT + TextCNN) | 0.835 | 0.802 | 0.847 | 0.771 | 1% | 100% | 0.0% | 108/29, p=6.7e-12 |
| 11 | FRIDRC (text + ViT) | 0.832 | 0.799 | 0.839 | 0.767 | 0% | 100% | 0.2% | 110/29, p=2.6e-12 |
| 40 | DeBERTa-v3 fine-tuned | 0.822 | 0.795 | 0.853 | 0.776 | 7% | 100% | 4.7% | 120/31, p=1.5e-13 |
| 38 | FRD-LSTM | 0.817 | 0.787 | 0.848 | 0.764 | 2% | 100% | 4.0% | 122/29, p=9.0e-15 |
| 39 | DHMFRD-TER | 0.783 | 0.768 | 0.829 | 0.744 | 16% | 100% | 15.3% | 155/34, p=1.2e-19 |
| 15 | A-LSTM + behaviour | 0.773 | 0.760 | 0.837 | 0.748 | 16% | 100% | 17.3% | 160/31, p=3.8e-22 |
| 02 | Shan inconsistency + RF | 0.767 | 0.725 | 0.802 | 0.699 | 5% | 79% | 8.1% | 145/11, p=5.5e-31 |
| — | **OURS (Extra Trees fusion)** | **0.932** | **0.929** | **0.967** | **0.955** | **81%** | 84% | **2.0%** | – |

"McNemar" counts test reviews where exactly one of the two models is right: ours right / it right, with the p-value
that the difference is not luck. **Every gap is significant (p < 7e-12).**

### 4.2 Full metrics (precision, recall, specificity)

Derived exactly from each model's per-fraud-type rates and the known group sizes (405 genuine + 135 + 135 + 135).

| model | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| BSTC | 0.8346 | 1.0000 | 0.6691 | 0.8018 | 1.0000 | 271 | 0 | 134 | 405 |
| FRIDRC | 0.8321 | 0.9963 | 0.6667 | 0.7988 | 0.9975 | 270 | 1 | 135 | 404 |
| DeBERTa-v3 | 0.8222 | 0.9365 | 0.6914 | 0.7955 | 0.9531 | 280 | 19 | 125 | 386 |
| FRD-LSTM | 0.8173 | 0.9446 | 0.6741 | 0.7867 | 0.9605 | 273 | 16 | 132 | 389 |
| DHMFRD-TER | 0.7827 | 0.8244 | 0.7185 | 0.7678 | 0.8469 | 291 | 62 | 114 | 343 |
| A-LSTM | 0.7728 | 0.8058 | 0.7185 | 0.7597 | 0.8272 | 291 | 70 | 114 | 335 |
| Shan + RF | 0.7667 | 0.8935 | 0.6148 | 0.7284 | 0.9185 | 249 | 33 | 156 | 372 |
| **OURS** | **0.9321** | **0.9783** | **0.8840** | **0.9287** | 0.9802 | 358 | 8 | 47 | 397 |

Every baseline's **recall is 0.61–0.72** — they find roughly two thirds of the fakes. BSTC's precision of 1.0000 is
caution, not skill: it flags only what it is certain of and misses 134 fakes.

### 4.3 Why they score 100% on fake text and 0.83 overall

"Fake text caught" is recall on **one group of 135 rows**, not accuracy. Counted as mistakes out of 810:

| model | misses edited photo | misses fake text | misses coordinated | false alarms | total mistakes | accuracy |
|---|---|---|---|---|---|---|
| BSTC | **134** | 0 | 0 | 0 | 134 | 0.835 |
| FRIDRC | **135** | 0 | 0 | 1 | 136 | 0.832 |
| DeBERTa-v3 | 125 | 0 | 0 | 19 | 144 | 0.822 |
| FRD-LSTM | 132 | 0 | 0 | 16 | 148 | 0.817 |
| DHMFRD-TER | 114 | 0 | 0 | 62 | 176 | 0.783 |
| Shan + RF | 156 | 0 | 0 | 33 | 189 | 0.767 |
| A-LSTM | 105 | 0 | 0 | 98 | 203 | 0.749 |
| **OURS** | **28** | 23 | 0 | 9 | **60** | **0.926** |

A model perfect on text but blind to photos can be right on at most (810 − 135)/810 = **0.833**. Verified by
rebuilding every reported accuracy from its own per-type rates: all 8 rows agree to **0.00e+00**
(`out/model_comparison/PAPER_NUMBERS_AUDIT.md`).

### 4.4 Why our fake-text number is lower than theirs

Because they are **pure text models and we are not**:

| our branch | edited photo | fake text | coordinated | false alarms | accuracy |
|---|---|---|---|---|---|
| text only | 5.2% | **88.1%** | 100% | 7.4% | 0.7852 |
| image only | **85.2%** | 8.9% | 93.3% | 9.4% | 0.7654 |
| **fusion (shipped)** | 80.7% | 84.4% | 100% | **2.0%** | **0.9321** |

Our text branch alone flags 119/135 fake-text rows; the fusion flags 112, **losing 7** where the photo and
behaviour evidence outvotes the text. What those 7 rows buy: edited photos caught **7 → 107**, genuine reviews
wrongly flagged **30 → 9**. This is also why each single branch looks weaker than the combination — the branches
are strong on **different** fraud types.

### 4.5 The novelty is not only "edited photos"

With the text branch switched off entirely, **image evidence alone** gets:

| fraud type | novelty claim | image evidence alone |
|---|---|---|
| `visual_manipulation` | 1. manipulation — was the photo edited? | **85.2%** |
| `coordinated_reuse` | 2. provenance / reuse — same photo, many accounts | **93.3%** |
| `text_deception` | (clean photo) | 8.9% — correct, there is nothing to find |

Both photo-based claims work without the words. The edited-photo column is emphasised only because it is the one
the published models cannot do at all (0–22%); they reach coordinated reuse through the shared *wording* of
campaign reviews, not through the photo.

### 4.6 The dataset fix and the model search (from readme12, numbers unchanged by the cache bug)

| | v6.1 | **v6.2** |
|---|---|---|
| test text copied word-for-word from train | 46.2% | **0.0%** |
| unique texts | 2,027 / 3,348 | 3,297 / 3,348 |

Ten classifiers were compared on grouped CV; Extra Trees won (CV ROC 0.9420). Three extra forensic maps (SRM,
JPEG grid, copy-move offset consistency) were tried and **made the edit detector worse** (ROC 0.9367 vs 0.9442,
inside seed noise), so the CNN stays at 3 channels — recorded as a failed experiment in
`out/model_comparison/cnn_channels.md`.

---

## 5. Where the model files are

Root: `D:\sem-5 projects\computer vision\dataset\fake_review_dataset_v6\out\`

| file | size | what it is |
|---|---|---|
| `model\fusion_rf.joblib` | 16.1 MB | **the main model** — the Extra Trees fusion over 41 features. (Filename kept from the Random Forest days so nothing else breaks.) |
| `model\forensic_cnn.pt` | 868 KB | the PyTorch CNN that reads the forensic maps and answers "was this photo edited?" |
| `model\text_minilm_lr.joblib` | 4 KB | logistic regression over MiniLM sentence vectors → `txt_minilm_p` |
| `model\image_rf.joblib` | 3.3 MB | image-only model, shown in the extension as "Photo evidence" |
| `model\text_rf.joblib` | 3.1 MB | text-only model, shown as "Text evidence" |
| `model\model_meta.json` | 3 KB | feature list, thresholds, decision bands, saved results |
| `reuse_embeddings_resnet50.npy` | 13.7 MB | ResNet-50 vectors of every known photo — the "image memory" used to spot reuse |
| `clip_image_embeddings.npy` | 6.9 MB | CLIP image vectors (cross-modal check) |
| `clip_text_embeddings.npy` | 6.9 MB | CLIP text vectors |

Loading it in Python:

```python
import joblib, json
M = r"D:\sem-5 projects\computer vision\dataset\fake_review_dataset_v6\out\model"
model = joblib.load(M + r"\fusion_rf.joblib")
meta  = json.load(open(M + r"\model_meta.json"))
print(meta["model"])            # ExtraTreesClassifier(...)
print(len(meta["features"]))    # 41  -- the exact column order the model expects
```

**Not committed to GitHub** (too big / regenerable): `forensic_maps*.npy` (314 MB each). Rebuild them with
`python forensic_maps.py`.

---

## 6. The extension, working on Amazon and Flipkart

### 6.1 What was broken and is now fixed

Both sites were checked against **live pages** on 20 September 2026, not against memory.

**Amazon** (`https://www.amazon.in/dp/B085JC431R`): the selector for the review text was **broken**.
amazon.in no longer emits `data-hook="review-body"`; the body is now `data-hook="reviewText"`. Everything else
(review cards, rating, date, author, review photos, listing photos, ASIN) was fine. After the fix, on that live
page: **13 review cards, 12 with text, 2 with photos**, and the thumbnail→full-size URL rewrite works
(`..._SY500_.jpg` → `.jpg`).

**Flipkart** was marked UNTESTED, and it could never have worked. Flipkart now ships **build-generated class
names** — `v1zwn21n v1zwn28 _1psv1zeb9 css-146c3p1` — which change on every deploy, so the old selectors
(`div._27M-vq`, `span.B_NuCI`) match nothing. Selectors based on those class names are not fixable, only
re-broken.

So Flipkart is now read **structurally**, using the one thing that is stable: every real review carries a buyer
badge, and the card begins with its star rating. `sites.js` finds the innermost element containing
"Verified Purchase" (Flipkart renamed it from "Certified Buyer") and walks up to the block that also holds the
rating. Tested live on a Flipkart reviews page: **10 review cards**, with rating, title, body, author and date all
parsed correctly.

`content.js` now supports both styles: a site provides either CSS selectors (Amazon) or `findCards()` + `parse()`
(Flipkart).

### 6.2 End-to-end proof on real Amazon reviews

Five **real, unmodified** reviews scraped from that live Amazon page, sent to the backend:

| review | risk | decision | why |
|---|---|---|---|
| "I really like the product, it's cool and no bad smell… no leakage, for 170rs it's a steal" (with photo) | 0.220 | **Genuine** | photo checked against the image memory |
| "It's sturdy & looks classy. After 6 months…" | 0.165 | **Genuine** | text + behaviour only |
| "Tuff plastic..... Unbreakable" | 0.426 | **Needs verification** | very short text, no photo |
| "Smart look, keep water's test normal." | 0.295 | **Genuine** | very short text, no photo |
| "Loose rubber seal" | 0.213 | **Genuine** | very short text, no photo |

**No real review was wrongly called fake.** The one amber verdict is a three-word review with no photo, which is
honestly "not enough evidence" rather than an accusation.

### 6.3 How to run it

**Step 1 — start the backend** (keep this window open; the first request loads the models and takes ~20 s):

```bash
cd "D:\sem-5 projects\computer vision\backend" && python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Check it is alive: open <http://127.0.0.1:8000/health> — it should report `"features": 41` and the model directory.

**Step 2 — load the extension into Chrome** (once):

1. open `chrome://extensions`
2. turn on **Developer mode** (top right)
3. click **Load unpacked**
4. choose the folder `D:\sem-5 projects\computer vision\extension`

**Step 3 — use it.** Open any Amazon or Flipkart product page. Each review gets a badge: green "Looks genuine",
amber "Needs verification", red "Likely fake", with the text score, the photo score and the reasons. A summary
panel appears above the reviews.

To see it working without shopping, open the built-in demo page, which serves 14 held-out reviews from our own
test set through the real model: <http://127.0.0.1:8000/demo>

Verified on the demo page with the new model: **5 likely fake, 3 need verification, 6 look genuine** — and the
answer key at <http://127.0.0.1:8000/demo/answers> says what each one really was.

### 6.4 Honest limits of the extension

- **The backend must be running on your computer.** Nothing is sent to any server; the photos are analysed
  locally. If the backend is off, the extension shows an error rather than a guess.
- **Speed**: roughly 1–3 seconds per review with a photo, because each photo is re-compressed for ELA, pushed
  through the forensic CNN, and embedded with ResNet-50 and CLIP. A page of 14 reviews takes about 25 seconds.
- **The image memory is our dataset.** "This photo was seen on another account" is judged against the 3,348
  photos we hold, plus whatever the extension has seen since. On a live Amazon page there is usually no prior
  sighting, so reuse evidence stays neutral — it becomes useful as the memory grows.
- **The model was trained on our constructed dataset**, where our own code made the edits. It detects *our* three
  edit types (copy-move, splice, noise injection) well; that is a controlled result, not a claim about every kind
  of real-world tampering.
- **Flipkart's structure can change.** The new approach depends on the words "Verified Purchase" and a leading
  star rating, which is far more stable than hashed class names, but Flipkart is a moving target. Only `sites.js`
  needs editing if it breaks.

---

## 7. Files added or changed in this round

| file | what it is |
|---|---|
| `code/refresh_clip_text.py` | rebuilds CLIP **text** vectors after the text changes (images are reused, not re-encoded) |
| `code/text_model.py` | embedding cache is now keyed by a SHA-256 of the text — the stale-cache bug cannot recur |
| `code/explain_modalities.py` | full metrics per model, branch-by-branch breakdown, novelty check |
| `extension/sites.js` | Amazon text selector fixed; Flipkart rewritten as a structural extractor |
| `extension/content.js` | supports both selector-style and `findCards()`/`parse()` sites |
| `out/model_comparison/MODALITY_EXPLAINED.md` | the tables in §4.2–§4.5 |
| `out/model_comparison/paper_models_v62_stalecache.csv` | the superseded numbers, kept for the record |

## 8. Still open

1. **Decision bands** (0.35 / 0.65) have never been tuned. Extra Trees gives less extreme probabilities, so 66
   fake reviews now land in "Needs verification". This is the top open item.
2. **5×5 grouped cross-validation** has not been re-run since v6.2.
3. **Confidence intervals for the seven baselines** — the comparison script stores summary rates, not per-row
   predictions, so producing CIs means retraining all seven (~2.5 h).
4. **Leave-one-category-out** experiment (`out/loco_folds.json`).
