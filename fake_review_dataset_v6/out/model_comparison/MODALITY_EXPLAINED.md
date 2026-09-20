# The other models, in full — and what "94% vs 100%" actually means

## Q1. Every model, all metrics (810 test reviews: 405 genuine, 135 edited photo, 135 fake text, 135 coordinated)

Precision, recall, specificity and F1 are derived exactly from each model's stored per-fraud-type rates and the known group sizes — the same identity that reproduced every reported accuracy to 0.00e+00.

| model | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| BSTC (BERT + TextCNN) | 0.8346 | 1.0000 | 0.6691 | 0.8018 | 1.0000 | 271 | 0 | 134 | 405 |
| FRIDRC (text enc. + ViT, fused) | 0.8321 | 0.9963 | 0.6667 | 0.7988 | 0.9975 | 270 | 1 | 135 | 404 |
| DeBERTa-v3 fine-tuned | 0.8222 | 0.9365 | 0.6914 | 0.7955 | 0.9531 | 280 | 19 | 125 | 386 |
| FRD-LSTM (DCWR + BiLSTM) | 0.8173 | 0.9446 | 0.6741 | 0.7867 | 0.9605 | 273 | 16 | 132 | 389 |
| Shan inconsistency + RF | 0.7914 | 0.8856 | 0.6691 | 0.7623 | 0.9136 | 271 | 35 | 134 | 370 |
| DHMFRD-TER (text + emotion + rating) | 0.7827 | 0.8244 | 0.7185 | 0.7678 | 0.8469 | 291 | 62 | 114 | 343 |
| A-LSTM + behaviour | 0.7778 | 0.8082 | 0.7284 | 0.7662 | 0.8272 | 295 | 70 | 110 | 335 |
| **OURS (Extra Trees fusion)** | 0.9494 | 0.9840 | 0.9136 | 0.9475 | 0.9852 | 370 | 6 | 35 | 399 |

| model | ROC-AUC | PR-AUC @15% holdout |
|---|---|---|
| DeBERTa-v3 fine-tuned | 0.8529 | 0.7755 |
| Shan inconsistency + RF | 0.8521 | 0.7656 |
| FRD-LSTM (DCWR + BiLSTM) | 0.8477 | 0.7643 |
| BSTC (BERT + TextCNN) | 0.8466 | 0.7710 |
| A-LSTM + behaviour | 0.8447 | 0.7729 |
| FRIDRC (text enc. + ViT, fused) | 0.8386 | 0.7665 |
| DHMFRD-TER (text + emotion + rating) | 0.8292 | 0.7437 |
| **OURS** | **0.9800** | **0.9630** |

**Note on confidence intervals:** ours has them because the model is on disk and can be re-scored 2,000 times. The seven baselines do not, because `compare_paper_models.py` stored summary rates rather than per-row predictions, and getting CIs means retraining all seven (~2.5 h on this machine). Say the word and it will be run.

## Q2. Why do they catch 100% of fake text and we catch 94%?

Because they are **pure text models** and we are not. Our own text branch is just as good as theirs; the fusion deliberately gives some of it away.

| model | edited photo | fake text | coordinated | false alarms | accuracy |
|---|---|---|---|---|---|
| text only | 4.4% | 97.8% | 100.0% | 6.4% | 0.8049 |
| image only | 85.9% | 4.4% | 96.3% | 8.1% | 0.7704 |
| FUSION (shipped) | 80.0% | 94.1% | 100.0% | 1.5% | 0.9494 |

**The trade, counted:** our text branch alone flags 132/135 fake-text rows; the fusion flags 127/135. The fusion **loses 6** of them, because the photo and behaviour evidence on those rows says "ordinary customer" and outvotes the text.

What that loss buys, on the same 810 reviews:

- edited-photo rows caught: text branch 6, fusion **108**
- genuine rows wrongly flagged: text branch 26, fusion **6**

So we give up a handful of fake-text rows and gain roughly a hundred edited-photo rows plus far fewer false alarms. A model that scores 100% on fake text and 0% on edited photos is not better — it is a text model being scored on a text-only slice of the problem.

## Q3. Are we only doing edited photos? Is that the novelty?

No. The stated novelty is treating the review PHOTO as forensic evidence in **three** ways. All three are in the dataset and all three are detected:

| evidence | fraud type it targets | rows in test | our detection |
|---|---|---|---|
| 1. manipulation (was the photo edited?) | `visual_manipulation` | 135 | 80.0% |
| 2. provenance / reuse (same photo, many accounts) | `coordinated_reuse` | 135 | 100.0% |
| 3. text deception (clean photo, fabricated words) | `text_deception` | 135 | 94.1% |

The **edited-photo column gets the attention only because it is the one the published models cannot do at all** (0-20%). Reuse and coordination they also reach 100% on — but not from the photo: coordinated reviews share talking points, so a text model catches them through the words. Our reuse and graph features reach the same rows through the evidence the novelty actually claims. Where that evidence is the ONLY route — an edited photo under an honestly written review — the published models collapse to 0-20% and ours holds at 80%.

Proof that the photo route works on its own, with the text branch switched off entirely (image features only):

| fraud type | caught by IMAGE evidence alone |
|---|---|
| `visual_manipulation` | 85.9% |
| `coordinated_reuse` | 96.3% |
| `text_deception` | 4.4% |
| `genuine` | 8.1% |

`text_deception` is low for the image branch and that is correct: those photos are genuine, there is nothing to find. `coordinated_reuse` is caught from the reused photo itself, which is novelty claim 2, not claim 1.
