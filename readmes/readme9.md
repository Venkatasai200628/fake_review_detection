# README 9: Verifying our results, which papers beat us, and what replaced CLIP

**Owner:** Venkata Sai · **Date:** 19 September 2026 · **Previous:** `readme8.md`

## 1. Are our results real? (independent verification)
New script `code/verify_results.py`, which trusts no stored number. Report: `out/VERIFY_RESULTS.md`.

| Check | Result |
|---|---|
| **Reproduce**: reload the saved model, rebuild every feature from scratch, recompute | all 10 reported numbers match exactly |
| **Confidence**: 2,000 bootstrap resamples, grouped by photo | balanced accuracy **0.943** (95% CI 0.926–0.959) · ROC-AUC 0.976 (0.964–0.985) · realistic PR-AUC 0.958 (0.915–0.987) |
| **Leakage**: train on randomly shuffled labels, 5 times | ROC-AUC 0.569 / 0.531 / 0.429 / 0.602 / 0.449, **mean 0.516 = coin flip** → the features do not contain the answer |
| **Stability**: retrain with 5 random seeds | accuracy 0.942–0.946 · ROC-AUC 0.975–0.977 · PR-AUC 0.959–0.961 |

Together with `REQUIREMENTS_CHECK.md` (39 PASS, 4 WARN, 1 FAIL, readme8), this is the proof the numbers are honest.

## 2. Which papers report better numbers than ours?
Full table: `papers\PAPERS_COMPARISON.md`, section "Which papers report HIGHER numbers than ours?".
Ours on a balanced test set: **accuracy 0.943 · F1 0.943 · ROC-AUC 0.976**.

**Higher on their own data (about 10 papers):** 39 DHMFRD-TER (0.988–0.994), 24 LDCP (99.2%, dataset not named),
38 FRD-LSTM (97.2%), 11 FRIDRC (F1 0.97, the only other text + image model), 09 fakeRoBERTa (96.6%), 26 BERT (96.3%),
13 RoBERTa-LSTM (96.0% on OpSpam), 40 MBO-DeBERTa (98% on OSF), 06 LLMIC (F1 95.9%), 04 (F1 0.983 on GPT-2 text).
03 is a tie (F1 94.6%).

**Lower (18 papers)**, including the **base paper He et al. (AUC 0.932, accuracy 0.860)**, BSTC (93.4%), Sun (acc 0.894),
A-LSTM (90.9%), Elmogy (87.9%), Lee (F1 0.564) and others.

**Why "higher" does not mean "better at our problem":**
1. All of them except FRIDRC are **text-only**. None checks whether a photo is edited or reused.
2. Their highest numbers come from easy data: GPT-2-generated text (OSF 40k) or 1,600 crowd-sourced reviews (Ott).
   The same models drop hard on other data: paper 40 gets 98% on OSF but **78%** on Amazon, and paper 09 gets 96.6% on its own
   data but **AUC 0.70** on human-written fakes.
3. **Their method on our data:** we fine-tuned DistilRoBERTa (the same family as BERT / RoBERTa / DeBERTa in those papers)
   on our dataset. It scores **0.833**, the maximum any text-only model can reach here, because a third of our fakes have
   perfect text. **Our fusion model scores 0.943 on the same data.**

So: about 10 papers report higher numbers **on their datasets**, but their approach cannot catch edited or reused photos,
and on our dataset it scores 11 points lower than our model.
The only paper with real, directly observed fake-review evidence (He et al.) is below ours. (Different level, though:
it scores products, we score individual reviews, so present it as a direction, not a head-to-head.)

## 3. What replaced CLIP?
CLIP was doing two jobs. One of them now uses a better model:

| Job | Before | Now | Why |
|---|---|---|---|
| **"Is this the same photo, re-posted?"** (reuse / coordination) | CLIP ViT-B/32 | **ResNet-50** (ImageNet-trained, `timm: resnet50.a1_in1k`), threshold 0.72 | catches 94.7% of reposts at 0.1% false positives vs CLIP's 83.0% (also beat DINOv2-S 92.2% and DINOv2-B 89.8%) |
| **"Does the photo match the text?"** (cross-modal) | CLIP ViT-B/32 | **CLIP ViT-B/32 (kept)** | only CLIP understands images and text in one shared space |

The other models in the final system:
- **Text:** 8 hand-made features + **MiniLM-L6** sentence embeddings (`sentence-transformers/all-MiniLM-L6-v2`) + Logistic Regression.
- **Edit detection:** our own **small CNN** on ELA / noise maps. It beat ImageNet ResNet-18 and EfficientNet-B0.
- **Final decision:** a **Random Forest** over all 41 features. It is the same classifier as the base paper, so the comparison stays fair.
