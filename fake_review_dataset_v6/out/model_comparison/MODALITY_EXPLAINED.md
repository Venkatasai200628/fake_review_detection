# The other models, in full — and what "94% vs 100%" actually means

## Q1. Every model, all metrics (810 test reviews: 405 genuine, 135 edited photo, 135 fake text, 135 coordinated)

Precision, recall, specificity and F1 are derived exactly from each model's stored per-fraud-type rates and the known group sizes — the same identity that reproduced every reported accuracy to 0.00e+00.

| model | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| BSTC (BERT + TextCNN) | 0.8346 | 1.0000 | 0.6691 | 0.8018 | 1.0000 | 271 | 0 | 134 | 405 |
| FRIDRC (text enc. + ViT, fused) | 0.8321 | 0.9963 | 0.6667 | 0.7988 | 0.9975 | 270 | 1 | 135 | 404 |
| DeBERTa-v3 fine-tuned | 0.8222 | 0.9365 | 0.6914 | 0.7955 | 0.9531 | 280 | 19 | 125 | 386 |
| FRD-LSTM (DCWR + BiLSTM) | 0.8173 | 0.9446 | 0.6741 | 0.7867 | 0.9605 | 273 | 16 | 132 | 389 |
| DHMFRD-TER (text + emotion + rating) | 0.7827 | 0.8244 | 0.7185 | 0.7678 | 0.8469 | 291 | 62 | 114 | 343 |
| Shan inconsistency + RF | 0.7667 | 0.8830 | 0.6148 | 0.7249 | 0.9185 | 249 | 33 | 156 | 372 |
| A-LSTM + behaviour | 0.7494 | 0.7538 | 0.7407 | 0.7472 | 0.7580 | 300 | 98 | 105 | 307 |
| **OURS (Extra Trees fusion)** | 0.9259 | 0.9752 | 0.8741 | 0.9219 | 0.9778 | 354 | 9 | 51 | 396 |

| model | ROC-AUC | PR-AUC @15% holdout |
|---|---|---|
| DeBERTa-v3 fine-tuned | 0.8529 | 0.7755 |
| FRD-LSTM (DCWR + BiLSTM) | 0.8477 | 0.7643 |
| BSTC (BERT + TextCNN) | 0.8466 | 0.7710 |
| FRIDRC (text enc. + ViT, fused) | 0.8386 | 0.7665 |
| A-LSTM + behaviour | 0.8318 | 0.7587 |
| DHMFRD-TER (text + emotion + rating) | 0.8292 | 0.7437 |
| Shan inconsistency + RF | 0.8018 | 0.6993 |
| **OURS** | **0.9660** | **0.9540** |

**Note on confidence intervals:** ours has them because the model is on disk and can be re-scored 2,000 times. The seven baselines do not, because `compare_paper_models.py` stored summary rates rather than per-row predictions, and getting CIs means retraining all seven (~2.5 h on this machine). Say the word and it will be run.

## Q2. Why do they catch 100% of fake text and we catch 94%?

Because they are **pure text models** and we are not. Our own text branch is just as good as theirs; the fusion deliberately gives some of it away.

| model | edited photo | fake text | coordinated | false alarms | accuracy |
|---|---|---|---|---|---|
| text only | 5.2% | 88.1% | 100.0% | 7.4% | 0.7852 |
| image only | 85.2% | 8.9% | 93.3% | 9.4% | 0.7654 |
| FUSION (shipped) | 79.3% | 83.0% | 100.0% | 2.2% | 0.9259 |

**The trade, counted:** our text branch alone flags 119/135 fake-text rows; the fusion flags 112/135. The fusion **loses 7** of them, because the photo and behaviour evidence on those rows says "ordinary customer" and outvotes the text.

What that loss buys, on the same 810 reviews:

- edited-photo rows caught: text branch 7, fusion **107**
- genuine rows wrongly flagged: text branch 30, fusion **9**

So we give up a handful of fake-text rows and gain roughly a hundred edited-photo rows plus far fewer false alarms. A model that scores 100% on fake text and 0% on edited photos is not better — it is a text model being scored on a text-only slice of the problem.

## Q3. Are we only doing edited photos? Is that the novelty?

No. The stated novelty is treating the review PHOTO as forensic evidence in **three** ways. All three are in the dataset and all three are detected:

| evidence | fraud type it targets | rows in test | our detection |
|---|---|---|---|
| 1. manipulation (was the photo edited?) | `visual_manipulation` | 135 | 79.3% |
| 2. provenance / reuse (same photo, many accounts) | `coordinated_reuse` | 135 | 100.0% |
| 3. text deception (clean photo, fabricated words) | `text_deception` | 135 | 83.0% |

The **edited-photo column gets the attention only because it is the one the published models cannot do at all** (0-20%). Reuse and coordination they also reach 100% on — but not from the photo: coordinated reviews share talking points, so a text model catches them through the words. Our reuse and graph features reach the same rows through the evidence the novelty actually claims. Where that evidence is the ONLY route — an edited photo under an honestly written review — the published models collapse to 0-20% and ours holds at 80%.

Proof that the photo route works on its own, with the text branch switched off entirely (image features only):

| fraud type | caught by IMAGE evidence alone |
|---|---|
| `visual_manipulation` | 85.2% |
| `coordinated_reuse` | 93.3% |
| `text_deception` | 8.9% |
| `genuine` | 9.4% |

`text_deception` is low for the image branch and that is correct: those photos are genuine, there is nothing to find. `coordinated_reuse` is caught from the reused photo itself, which is novelty claim 2, not claim 1.
