# Audit of the paper-model table (no re-statement, everything recomputed)

Test set: **810 reviews** read from `out/test_balanced.csv`.

| group | rows | share of test set |
|---|---|---|
| genuine | 405 | 50.0% |
| visual_manipulation | 135 | 16.7% |
| text_deception | 135 | 16.7% |
| coordinated_reuse | 135 | 16.7% |

Fake rows: 405. A model that is perfect on text but blind to photos can therefore be right on at most 0.8333 of the test set (810 - 135 edited-photo rows), i.e. **83.3%**.

## 1. Does each reported accuracy follow from its own per-type rates?

`rebuilt = [(1-false_alarms)*genuine + caught_visual*135 + caught_text*135 + caught_coord*135] / 810`

| model | edited photo | fake text | coordinated | false alarms | rows right (rebuilt) | rebuilt acc | reported acc | difference |
|---|---|---|---|---|---|---|---|---|
| Shan inconsistency + RF | 20.0% | 80.7% | 100.0% | 8.6% | 641.0 | 0.7914 | 0.7914 | +0.00e+00 |
| A-LSTM + behaviour | 18.5% | 100.0% | 100.0% | 17.3% | 630.0 | 0.7778 | 0.7778 | +0.00e+00 |
| DHMFRD-TER (text + emotion + rating) | 15.6% | 100.0% | 100.0% | 15.3% | 634.0 | 0.7827 | 0.7827 | +0.00e+00 |
| FRD-LSTM (DCWR + BiLSTM) | 2.2% | 100.0% | 100.0% | 4.0% | 662.0 | 0.8173 | 0.8173 | +0.00e+00 |
| DeBERTa-v3 fine-tuned | 7.4% | 100.0% | 100.0% | 4.7% | 666.0 | 0.8222 | 0.8222 | +0.00e+00 |
| FRIDRC (text enc. + ViT, fused) | 0.0% | 100.0% | 100.0% | 0.2% | 674.0 | 0.8321 | 0.8321 | +0.00e+00 |
| BSTC (BERT + TextCNN) | 0.7% | 100.0% | 100.0% | 0.0% | 676.0 | 0.8346 | 0.8346 | +0.00e+00 |
| **OURS (recomputed now)** | 80.0% | 94.1% | 100.0% | 1.5% | 769.0 | 0.9494 | 0.9494 | +0.00e+00 |

Rows where the rebuilt accuracy disagrees with the reported accuracy by more than 1e-9: **0**.

## 2. Where each model loses its 810 rows (counts, not percentages)

| model | misses: edited photo | misses: fake text | misses: coordinated | false alarms | total mistakes | accuracy |
|---|---|---|---|---|---|---|
| Shan inconsistency + RF | 108 | 26 | 0 | 35 | 169 | 0.791 |
| A-LSTM + behaviour | 110 | 0 | 0 | 70 | 180 | 0.778 |
| DHMFRD-TER (text + emotion + rating) | 114 | 0 | 0 | 62 | 176 | 0.783 |
| FRD-LSTM (DCWR + BiLSTM) | 132 | 0 | 0 | 16 | 148 | 0.817 |
| DeBERTa-v3 fine-tuned | 125 | 0 | 0 | 19 | 144 | 0.822 |
| FRIDRC (text enc. + ViT, fused) | 135 | 0 | 0 | 1 | 136 | 0.832 |
| BSTC (BERT + TextCNN) | 134 | 0 | 0 | 0 | 134 | 0.835 |
| **OURS** | 27 | 8 | 0 | 6 | 41 | 0.949 |

## 3. The "98% not 100%" cases: the 8 fake-text reviews our model misses

Our fake-text recall is 0.9407 = 127/135. The missed rows:

| review_id | our score | rating | text |
|---|---|---|---|
| R00942 | 0.246 | 4 | Picked this up for daily morning practice about about eight months ago and the foam density is as good as day  |
| R01320 | 0.292 | 5 | Ordered it for a daily walk to work, four months ago now and the insole shows no wear at all. No regrets at al |
| R01690 | 0.313 | 5 | About a month of use on a camping trip. The pairing button is flawless. Very pleased with this one. |
| R02177 | 0.474 | 5 | Using it for driving for a month now and the lens coating has not given me a moment of trouble. Would happily  |
| R02605 | 0.196 | 5 | Bought it on a week of travel and about eight months later. The shoulder straps is as good as day one. Better  |
| R02769 | 0.472 | 5 | Had it a month, mostly on sensitive skin and the consistency is flawless. Genuinely impressed with the build. |
| R02814 | 0.369 | 5 | About half a year of use on a two week trip. The telescopic handle still looks new. Better than I expected hon |
| R03308 | 0.485 | 4 | Been using this for carrying a laptop and a lunch box for two weeks. The inner lining shows no wear at all. Do |

Our score on those rows: 0.356 mean (threshold 0.5). On the 127 fake-text rows we do catch: 0.893 mean.
