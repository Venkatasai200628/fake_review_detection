# README 12: "Their model catches 100% of fake text and ours catches 98% — but their accuracy is lower. That doesn't match."

**Owner:** Venkata Sai · **Date:** 20 September 2026 · **Previous:** `readme11.md`

> **SUPERSEDED (see `readme13.md`).** The numbers in sections 5 and 6 below were measured with a stale embedding
> cache: `text_model.py` reloaded MiniLM vectors built from the OLD text, so `txt_minilm_p` described reviews that
> no longer existed, while the transformer baselines used the new text. Corrected figures (accuracy 0.9259, not
> 0.9494) are in `readme13.md` §1-§2. The dataset fix, the model search and the failed CNN experiment in sections
> 3 and 4 are unaffected.


## 0. What you asked

Three things:

1. If the papers' models catch **100%** of fake text and ours catches only **98%**, how can their overall accuracy
   be 0.83 while ours is 0.94? "This doesn't match."
2. Don't write code that produces a nice-looking answer. Execute it properly; the values must be real.
3. Then: try better models; if the new ones are worse than what we have, stop and fix the present one; and our
   model should also beat the papers that scored well on images.

All three are done. Point 1 has a simple answer. Point 2 turned up a **real defect in our dataset**, which is now
fixed. Point 3 produced one winner and one honest failure.

---

## 1. Why 100% on fake text still gives only 0.83 accuracy

"Fake text caught" is **not** accuracy. It is recall on **one group of 135 rows**. The test set is:

| group | rows | share |
|---|---|---|
| genuine | 405 | 50.0% |
| edited photo (`visual_manipulation`) | 135 | 16.7% |
| fake text (`text_deception`) | 135 | 16.7% |
| coordinated (`coordinated_reuse`) | 135 | 16.7% |
| **total** | **810** | |

A text-only model reads words. On fake text it is perfect. On an **edited photo with an honest review written by a
real customer**, there is nothing in the words to find, so it misses all 135. 135 out of 810 is 16.7%, and
1 − 0.167 = **0.833**. That is exactly where five of the seven models landed.

Counted as mistakes instead of percentages, it is obvious (v6.1 numbers, the ones you were asking about):

| model | misses edited photo | misses fake text | misses coordinated | false alarms | total mistakes | accuracy |
|---|---|---|---|---|---|---|
| FRIDRC (text + ViT) | **135** | 0 | 0 | 0 | 135 | 0.833 |
| BSTC (BERT + TextCNN) | **135** | 0 | 0 | 0 | 135 | 0.833 |
| FRD-LSTM | 134 | 0 | 0 | 4 | 138 | 0.830 |
| DeBERTa-v3 | 131 | 0 | 0 | 9 | 140 | 0.827 |
| A-LSTM | 118 | 0 | 0 | 31 | 149 | 0.816 |
| DHMFRD-TER | 113 | 0 | 0 | 59 | 172 | 0.788 |
| Shan + RF | 130 | 12 | 0 | 31 | 173 | 0.786 |
| **OURS** | **24** | 3 | 0 | 19 | **46** | **0.943** |

So there is no contradiction. They are perfect on one sixth of the test set and blind to another sixth. We give up
3 fake-text rows and win back 111 edited-photo rows.

## 2. Proving the numbers were not shaped to look good

You said not to design code around the output. So I did not re-state the table — I **rebuilt every accuracy from
its own per-type rates** and compared it with what was reported:

```
rebuilt = [ (1 - false_alarms) x 405
          + caught_edited x 135 + caught_text x 135 + caught_coord x 135 ] / 810
```

If any number had been typed by hand or come from a broken metric, the two would disagree.

**All 8 rows, ours included, matched with a difference of +0.00e+00.** Code:
`dataset/fake_review_dataset_v6/code/verify_paper_numbers.py`, report:
`out/model_comparison/PAPER_NUMBERS_AUDIT.md`.

The 3 fake-text reviews our model missed are printed there individually, with their scores (all between 0.16 and
0.34, all five-star "used it for four months" texts).

## 3. The real problem that check uncovered

Our review text is written from templates. So I asked a second question the table could not answer: **is that 100%
real skill, or is the model just recognising sentences it already saw in training?**

Measured (`code/verify_text_difficulty.py` → `out/model_comparison/TEXT_DIFFICULTY_AUDIT.md`):

- **46.2%** of test reviews were **word-for-word copies** of a training review
  - fake text: **68.9%** · coordinated: **100%**
- A rule with no learning in it at all — *"copy the label of the most similar training review"* — already scored
  **100% on fake text and 100% on coordinated**.
- Cause: `gen_text.py` picks one **finished sentence** from a short list. `DECEPTIVE_POSITIVE` had 8 fixed
  strings, `CAMPAIGN_TEMPLATES` 5, `LAZY_GENUINE` 8. With 3,348 rows to fill, the same sentences had to be reused
  on both sides of the split.

**This inflated everybody's text score, ours included.** It is exactly the kind of thing you were right to suspect.

### The fix (v6.2)

`code/gen_text2.py` **assembles** each review from independent slots instead of choosing one whole sentence, so each
style can produce tens of thousands of different reviews. `code/regen_text.py` then rewrites **only the text
column**, because the photos, splits, IDs, reviewer accounts, timestamps and campaigns were all fine and the
forensic maps and embeddings are indexed to those exact rows.

Every row keeps its original style and its original sentiment:

- style comes from `fraud_type` + the `notes` marker that `build_dataset.py` already recorded
- sentiment is read back from the row's **existing rating**, so text and stars still agree and the v6.1 fix that
  removed the rating shortcut is preserved — **no rating was changed**
- the random seed comes from the `review_id`, so it is reproducible and order-independent
- no text may repeat, except inside one campaign, where members are *supposed* to look alike

Result:

| | before (v6.1) | **after (v6.2)** |
|---|---|---|
| test text copied word-for-word from train | **46.2%** | **0.0%** |
| — fake text | 68.9% | 0.0% |
| — coordinated | 100.0% | 0.0% |
| nearest-neighbour similarity, fake text (median) | 1.000 | 0.557 |
| unique texts in the dataset | 2,027 / 3,348 | 3,297 / 3,348 |

The old text is kept in `out/review_text_v61.csv`, so the change is reversible and the audit can be repeated.

### What the fix cost us

I expected a big drop. It was **0.1 accuracy points** (0.9432 → 0.9420 with the same Random Forest). My earlier
estimate of "−2.8 points" was too pessimistic: it came from scoring the rows whose text happened to be rare, and
those were disproportionately the awkward ones. With the text properly regenerated the model learns a real style
boundary instead of memorising sentences. The text branch on its own still reaches **ROC 0.827**, catching 98.5%
of fake text and 3% of edited photos — the signal is genuine, it just is not a look-up any more.

---

## 4. Trying better models (you asked; the rule was "keep it only if it wins")

### 4a. The classifier on top — Extra Trees wins

`code/compare_fusion_models.py` ran ten candidates on the **same 41 features, same rows, same grouped folds**. The
winner was picked on **cross-validation over the training data**, not on the test set, so the choice is not tuned
to the answer.

| model | CV ROC (grouped) | test accuracy | ROC | PR-AUC @15% | edited photo | false alarms |
|---|---|---|---|---|---|---|
| **Extra Trees 1000, leaf 2, sqrt** | **0.9553** | 0.9494 | 0.9795 | **0.9628** | 80.0% | **1.5%** |
| Extra Trees 1000, leaf 1 | 0.9553 | 0.9519 | 0.9780 | 0.9610 | 80.7% | 1.5% |
| HistGradientBoosting | 0.9539 | 0.9444 | **0.9834** | 0.9628 | **83.0%** | 4.2% |
| Soft vote ET+RF+HGB | 0.9550 | 0.9432 | 0.9801 | 0.9615 | 79.3% | 3.0% |
| Random Forest 500 (old model) | 0.9530 | 0.9420 | 0.9780 | 0.9560 | 81.5% | 4.0% |
| XGBoost deep | 0.9548 | 0.9407 | 0.9831 | 0.9582 | 83.0% | 4.9% |
| Stacking (RF+XGB+HGB → LR) | – | 0.9383 | 0.9807 | 0.9622 | 83.0% | 5.7% |
| Logistic regression | 0.9433 | 0.9259 | 0.9740 | 0.9379 | 79.3% | 5.2% |

**Extra Trees replaces the Random Forest.** Full table: `out/model_comparison/fusion_models.md`.

### 4b. A stronger image detector — tried, failed, dropped

The weak spot was copy-move detection (ROC 0.89 against 0.97 for splice). So I built three more forensic maps
(`code/forensic_maps_extra.py`): an SRM high-pass residual, a JPEG block-grid mismatch map, and a copy-move
detector based on **offset consistency** (a pasted region makes many patches agree on one translation).

It did not work:

| input channels | ROC | seed 1 / seed 2 | copy-move | splice | noise |
|---|---|---|---|---|---|
| **3 ch (current)** | **0.9442** | 0.950 / 0.906 | **0.886** | **0.971** | 0.961 |
| 6 ch (+ the three new maps) | 0.9367 | 0.919 / 0.940 | 0.858 | 0.957 | **0.976** |

Six channels are **worse**, and the gap (0.0075) is smaller than the difference between two random seeds
(0.906–0.950), so it is not even a real difference. **Why it failed:** ordinary photos are full of repeated
structure — fabric weave, tiles, grilles, bokeh — and those produce the same "many patches agree on one offset"
pattern as a copied region. The channel added noise, not evidence.

Following your rule — *if the new one is not better, stop and keep the present one* — **the CNN stays at 3
channels.** Recorded as a failed experiment rather than quietly dropped:
`out/model_comparison/cnn_channels.md`.

---

## 5. The final model (v6.2, Extra Trees, honest text)

| | old headline (RF, leaky text) | **new (ET, fixed text)** |
|---|---|---|
| PR-AUC, balanced test | 0.982 | **0.984** |
| ROC-AUC | 0.976 | **0.980** |
| F1 | 0.943 | **0.948** |
| Precision | 0.952 | **0.984** |
| Recall | 0.933 | 0.914 |
| accuracy | 0.9432 | 0.9494 |
| **false alarms** | 4.7% (19/405) | **1.5% (6/405)** |
| PR-AUC @15% realistic holdout | 0.958 | **0.963** |
| legit stock photo reuse wrongly flagged | 1 / 12 | **0 / 12** |

Detection by fraud type: coordinated **100%**, fake text **94.1%**, edited photo **80.0%**, false alarms **1.5%**.

**Checked independently** (`code/verify_results.py` → `out/VERIFY_RESULTS.md`), because a new number should not be
trusted just because it is higher:

- every metric reproduces exactly from a fresh load of the saved model
- 95% confidence interval (2,000 bootstrap resamples, grouped by root photo): accuracy **0.934–0.963**,
  ROC-AUC 0.969–0.988, PR-AUC 0.977–0.990
- **label-shuffle control**: train on randomly shuffled labels and the test ROC-AUC must fall to ~0.5. It gives
  0.550, 0.516, 0.442, 0.607, 0.418 — mean **0.506**. No feature is leaking the answer.
- **seed stability**: retrained 5 times, accuracy 0.944–0.948, ROC-AUC 0.977–0.979

So the model is better than the old one on every headline metric **and** the dataset it was measured on is now
honest. Only 6 genuine reviews out of 405 are wrongly flagged, which matters for a browser extension — a tool that
cries wolf on real reviews is useless.

**One thing to watch:** Extra Trees gives less extreme probabilities, so more rows land in the middle band. Under
the three-way decision (Genuine < 0.35, Fake > 0.65), 58 fake reviews now fall into "Needs verification" instead of
"Fake", against 23 before. That is not a wrong answer — it is the model saying "I am not sure", which is what the
band is for — but the bands were never tuned, and that is the next job.

---

## 6. The papers' models, re-run on the fixed dataset

All seven were retrained from scratch on v6.2, because their v6.1 numbers were measured on the leaky text and are
no longer valid. Same 2,490 training reviews, same 810 unseen test reviews, same metrics.
(The old table is kept at `out/model_comparison/paper_models_v61_leaky.csv`.)

| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo | Fake text | False alarms | McNemar vs ours |
|---|---|---|---|---|---|---|---|---|---|
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.835 | 0.802 | 0.847 | 0.771 | 1% | 100% | 0.0% | 107/14, p=6.5e-19 |
| 11 Hou 2025 | FRIDRC (text + ViT) | 0.832 | 0.799 | 0.839 | 0.767 | 0% | 100% | 0.2% | 109/14, p=2.1e-19 |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.822 | 0.795 | 0.853 | 0.776 | 7% | 100% | 4.7% | 119/16, p=1.2e-20 |
| 38 Qayyum 2023 | FRD-LSTM | 0.817 | 0.787 | 0.848 | 0.764 | 2% | 100% | 4.0% | 121/14, p=2.0e-22 |
| 02 Shan 2021 | inconsistency + RF | 0.791 | 0.762 | 0.852 | 0.766 | 20% | 81% | 8.6% | 140/12, p=7.8e-29 |
| 39 Duma 2024 | DHMFRD-TER | 0.783 | 0.768 | 0.829 | 0.744 | 16% | 100% | 15.3% | 155/20, p=4.6e-27 |
| 15 Xu 2024 | A-LSTM + behaviour | 0.778 | 0.766 | 0.845 | 0.773 | 19% | 100% | 17.3% | 158/19, p=1.8e-28 |
| **OURS** | **forensic fusion (Extra Trees)** | **0.949** | **0.948** | **0.980** | **0.963** | **80%** | 94% | **1.5%** | – |

**We now lead on every single column, including false alarms** — which we did not before. On v6.1 three baselines
had fewer false alarms than us (0–1.0% against our 4.7%); now ours is 1.5%, and only BSTC (0.0%) and FRIDRC (0.2%)
are lower, because they almost never say "fake" at all.

Same result counted as mistakes out of 810, which is the clearest answer to your question:

| model | misses edited photo | misses fake text | false alarms | total mistakes | accuracy |
|---|---|---|---|---|---|
| BSTC | **134** | 0 | 0 | 134 | 0.835 |
| FRIDRC | **135** | 0 | 1 | 136 | 0.832 |
| DeBERTa-v3 | 125 | 0 | 19 | 144 | 0.822 |
| FRD-LSTM | 132 | 0 | 16 | 148 | 0.817 |
| Shan + RF | 108 | 26 | 35 | 169 | 0.791 |
| DHMFRD-TER | 114 | 0 | 62 | 176 | 0.783 |
| A-LSTM | 110 | 0 | 70 | 180 | 0.778 |
| **OURS** | **27** | 8 | 6 | **41** | **0.949** |

Every reported accuracy was again rebuilt from its own per-type rates: **all 8 rows agree to +0.00e+00**.

### The fix did not flatter us

This is the important part. Removing the duplicated text made the **baselines worse and ours better**:

| model | v6.1 acc | v6.2 acc | v6.1 false alarms | v6.2 false alarms |
|---|---|---|---|---|
| A-LSTM | 0.816 | **0.778** | 7.7% | **17.3%** |
| FRD-LSTM | 0.830 | **0.817** | 1.0% | **4.0%** |
| DeBERTa-v3 | 0.827 | **0.822** | 2.2% | **4.7%** |
| DHMFRD-TER | 0.788 | 0.783 | 14.6% | 15.3% |
| BSTC | 0.833 | 0.835 | 0.0% | 0.0% |
| FRIDRC | 0.833 | 0.832 | 0.0% | 0.2% |
| Shan + RF | 0.786 | 0.791 | 7.7% | 8.6% |
| **OURS** | 0.943 | **0.949** | 4.7% | **1.5%** |

Without memorised sentences to fall back on, the text-only models start calling genuine reviews fake. They were
leaning on the duplication harder than we were. If the fix had been done to make us look good it would have worked
the other way round.

### Where we are still weakest

We miss **8 of 135** fake-text reviews (94%, not 100%). All 8 are printed with their scores in the audit file, and
every one is a *sophisticated deceptive* row — fabricated specifics written to read like a real customer, on a
clean photo:

> *"About a month of use on a camping trip. The pairing button is flawless. Very pleased with this one."*

There is nothing to find in the words and nothing to find in the pixels. This is the overlap case the dataset was
built to contain (guide: OVERLAP CASES), not a defect. Say so plainly in the report.

---

## 6b. Three follow-up questions

### 6b.1 The other models, in full (not just accuracy)

Precision, recall, F1 and specificity are derived **exactly** from each model's stored per-fraud-type rates and the
known group sizes (405 genuine, 135 + 135 + 135 fake) - the same identity that reproduced every accuracy to
0.00e+00.

| model | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| BSTC | 0.8346 | 1.0000 | 0.6691 | 0.8018 | 1.0000 | 271 | 0 | 134 | 405 |
| FRIDRC | 0.8321 | 0.9963 | 0.6667 | 0.7988 | 0.9975 | 270 | 1 | 135 | 404 |
| DeBERTa-v3 | 0.8222 | 0.9365 | 0.6914 | 0.7955 | 0.9531 | 280 | 19 | 125 | 386 |
| FRD-LSTM | 0.8173 | 0.9446 | 0.6741 | 0.7867 | 0.9605 | 273 | 16 | 132 | 389 |
| Shan + RF | 0.7914 | 0.8856 | 0.6691 | 0.7623 | 0.9136 | 271 | 35 | 134 | 370 |
| DHMFRD-TER | 0.7827 | 0.8244 | 0.7185 | 0.7678 | 0.8469 | 291 | 62 | 114 | 343 |
| A-LSTM | 0.7778 | 0.8082 | 0.7284 | 0.7662 | 0.8272 | 295 | 70 | 110 | 335 |
| **OURS** | **0.9494** | **0.9840** | **0.9136** | **0.9475** | 0.9852 | 370 | 6 | 35 | 399 |

ROC-AUC: best baseline DeBERTa 0.8529, **ours 0.9800**. PR-AUC @15%: best baseline 0.7755, **ours 0.9630**.

Look at **recall**: every baseline sits at 0.67-0.73, finding barely two thirds of the fakes. BSTC's perfect
precision (1.0000) is not skill, it is caution - it only flags what it is certain of and misses 134 fakes.

**Honest gap:** the baselines have no confidence intervals, because `compare_paper_models.py` stored summary rates
rather than per-row predictions. Producing them means retraining all seven (~2.5 h).

### 6b.2 Why they catch 100% of fake text and we catch 94%

**Because they are pure text models and we are not. Our text branch is as good as theirs - the fusion gives some
of it away on purpose.**

| our branch | edited photo | fake text | coordinated | false alarms | accuracy |
|---|---|---|---|---|---|
| text only | 4.4% | **97.8%** | 100% | 6.4% | 0.8049 |
| image only | **85.9%** | 4.4% | 96.3% | 8.1% | 0.7704 |
| **fusion (shipped)** | 80.0% | 94.1% | 100% | **1.5%** | **0.9494** |

Our text branch alone flags **132/135** fake-text rows. The fusion flags 127, **losing 6**, because on those rows
the photo and behaviour evidence says "ordinary customer" and outvotes the text.

What those 6 rows buy, on the same 810 reviews:

- edited photos caught: text branch **6** -> fusion **108**
- genuine reviews wrongly flagged: text branch **26** -> fusion **6**

Give up 6 text rows, gain 102 photo rows, remove 20 false alarms. A model scoring 100% on fake text and 0% on
edited photos is not better - it is a text model being graded on the text-only slice of the problem.

This is also the answer to the older question "how can the individual branches be weaker than the combination?"
The branches are strong on **different** fraud types, so the union covers all three.

### 6b.3 "Are we only considering edited photos? Is that the novelty I gave you?"

No. The novelty is the photo as forensic evidence in **three** ways, and all three are in the dataset and detected:

| evidence | fraud type | rows | our detection |
|---|---|---|---|
| 1. manipulation - was the photo edited? | `visual_manipulation` | 135 | 80.0% |
| 2. provenance / reuse - same photo, many accounts | `coordinated_reuse` | 135 | 100% |
| 3. text deception - clean photo, fabricated words | `text_deception` | 135 | 94.1% |

**The photo route stands on its own.** With the text branch switched off completely, image features alone catch:

| fraud type | caught by IMAGE evidence alone |
|---|---|
| `visual_manipulation` | **85.9%** |
| `coordinated_reuse` | **96.3%** |
| `text_deception` | 4.4% (correct - those photos are genuine, there is nothing to find) |

So the image side delivers on **claim 1 (manipulation) and claim 2 (reuse/provenance)** without any help from the
words.

The edited-photo column gets emphasised only because it is the one the published models **cannot do at all**
(0-20%). They also reach 100% on coordinated - but through the *words*, because coordinated reviews share talking
points, not through the photo. Where the photo is the only available evidence, they collapse and we hold at 80%.

Code: `code/explain_modalities.py`, report: `out/model_comparison/MODALITY_EXPLAINED.md`.

## 7. Files added or changed in this round

| file | what it is |
|---|---|
| `code/verify_paper_numbers.py` | rebuilds every reported accuracy from its own per-type rates |
| `code/verify_text_difficulty.py` | measures verbatim duplication and the "memorising floor" |
| `code/gen_text2.py` | the large, slot-based text generator (v6.2) |
| `code/regen_text.py` | rewrites only the text column; keeps ratings, photos, splits, campaigns |
| `code/compare_fusion_models.py` | ten classifiers on the same features; ranked by grouped CV |
| `code/forensic_maps_extra.py` | the three extra forensic maps (kept, though they did not help) |
| `code/compare_cnn_channels.py` | 3-channel vs 6-channel edit detector |
| `out/review_text_v61.csv` | the old text, so the change is reversible |
| `out/model_comparison/PAPER_NUMBERS_AUDIT.md` | the reconciliation, all differences 0 |
| `out/model_comparison/TEXT_DIFFICULTY_AUDIT.md` | the duplication evidence |
| `out/model_comparison/fusion_models.md` | Phase A table |
| `out/model_comparison/cnn_channels.md` | Phase B table (the failed experiment) |
| `code/explain_modalities.py` | full metrics for every model; branch-by-branch breakdown |
| `out/model_comparison/MODALITY_EXPLAINED.md` | the three follow-up answers in §6b |

## 8. What is still open

1. **Tune the decision bands** (0.35 / 0.65) on validation folds — more important now that Extra Trees pushes 58
   fakes into the middle band.
2. **Leave-one-category-out** test, using `out/loco_folds.json`.
3. **Extension**: still needs testing on a live Amazon page; the Flipkart selectors are untested. The backend must
   also be pointed at the new Extra Trees model.
4. The images themselves were never regenerated, so the honest caveat from earlier readmes still stands: our own
   code made the edits, so the CNN is measured against our three edit types, not against real-world tampering.
