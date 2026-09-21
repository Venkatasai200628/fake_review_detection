# README 11: The papers' own models, re-built and run on our dataset

**Owner:** Venkata Sai · **Date:** 20 September 2026 · **Previous:** `readme10.md`

## 0. What you asked
"Their models on our dataset, can you do it?" — yes. readme10 compared method *families*; this one re-builds the
**specific named models** of seven papers and trains and tests each on our data: same 2,490 training reviews,
same 810 unseen test reviews, same metrics.
Code: `dataset/fake_review_dataset_v6/code/compare_paper_models.py` (total training time ~3 h on CPU).

## 1. Results
| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms | Ours right / it right (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|
| 02 Shan 2021 | inconsistency features + Random Forest | 0.786 | 0.753 | 0.813 | 0.750 | 4% | 91% | 7.7% | 148 / 21 (1e-24) |
| 15 Xu 2024 | A-LSTM + behaviour | 0.816 | 0.794 | 0.841 | 0.753 | 13% | 100% | 7.7% | 125 / 22 (1e-18) |
| 38 Qayyum 2023 | FRD-LSTM (contextual vectors + BiLSTM) | 0.830 | 0.797 | 0.839 | 0.769 | 1% | 100% | 1.0% | 113 / 21 (2e-16) |
| 39 Duma 2024 | DHMFRD-TER (text + emotion + rating) | 0.788 | 0.772 | 0.825 | 0.773 | 16% | 100% | 14.6% | 149 / 23 (9e-24) |
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.833 | 0.800 | 0.834 | 0.755 | 0% | 100% | 0.0% | 111 / 22 (2e-15) |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.827 | 0.797 | 0.841 | 0.771 | 3% | 100% | 2.2% | 116 / 22 (1e-16) |
| 11 Hou 2025 | FRIDRC (text encoder + ViT, fused) | 0.833 | 0.800 | 0.846 | 0.782 | 0% | 100% | 0.0% | 111 / 22 (2e-15) |
| **OURS** | **forensic fusion** | **0.943** | **0.943** | **0.976** | **0.958** | **82%** | 98% | 4.7% | – |

"Ours right / it right" counts the test reviews where exactly one of the two models is correct. Example, row 1: ours is
right on 148 reviews where Shan's model is wrong; it is right on 21 where ours is wrong. The McNemar p-value says the
difference is not luck.

## 2. What the numbers say
1. **All seven lose to ours, and every gap is significant** (p < 1e-14, worst case).
2. **They are text models.** They catch fake *text* nearly perfectly (91–100%) but edited *photos* almost never (0–16%).
3. **They pile up at 0.83.** Five of the seven land between 0.827 and 0.833, exactly the ceiling we computed for
   any text-only model on this dataset (readme8 §1). They are not "bad models": they are at the limit of what text can do here.
4. **The one multimodal paper (FRIDRC) is the interesting case.** With frozen embeddings + a linear classifier it caught
   36% of edited photos; **fine-tuned end-to-end it caught 0%**, because the text signal dominates training.
   A general image encoder (ViT/CLIP) describes *what is in* the photo, not *whether it was edited*. That is the gap our
   forensic CNN fills.
5. **Where ours is weaker:** BSTC, FRIDRC and FRD-LSTM have fewer false alarms (0–1.0% vs our 4.7%), because they almost
   never say "fake" unless the text is obviously fake. Report that honestly; it is the price of catching 82% of edited photos.

## 3. Honest simplifications (CPU-only re-builds)
| Paper | What differs from the original |
|---|---|
| 38 FRD-LSTM | DCWR approximated with frozen DistilBERT contextual vectors |
| 30 BSTC | SKEP sentiment-knowledge branch omitted (no English PyTorch version) |
| 40 MBO-DeBERTa | deberta-v3-small; the Monarch-Butterfly hyper-parameter search omitted |
| 15 A-LSTM | word embeddings trained from scratch (no GloVe download) |
| 39 DHMFRD-TER | emotions from a pretrained 7-emotion classifier |
| 11 FRIDRC | image encoder CLIP ViT-B/32 **frozen** (fine-tuning ViT on CPU is not feasible); no profile data |
| 02 Shan | sentiment from DistilBERT-SST2; 9 inconsistency features + our 8 base text features |

All simplifications make the baselines *slightly weaker* than the published versions, so state it plainly in the report:
*"Baselines were re-implemented from the papers' descriptions with CPU-only substitutions, listed in Table X."*
Even so, the gap (0.83 vs 0.943) is far larger than any plausible effect of these substitutions, and the
"edited photo caught" column (0–16% vs 82%) is a structural difference, not a tuning one.

## 4. Where it is written down
- `papers\PAPERS_COMPARISON.md` §1c (this table) and §1b (method families)
- `out/model_comparison/paper_models_on_our_data.csv` / `.md`
- guide §25
