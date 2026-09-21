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
| Shan inconsistency + RF | 5.2% | 79.3% | 100.0% | 8.1% | 621.0 | 0.7667 | 0.7667 | +0.00e+00 |
| A-LSTM + behaviour | 15.6% | 100.0% | 100.0% | 17.3% | 626.0 | 0.7728 | 0.7728 | +0.00e+00 |
| DHMFRD-TER (text + emotion + rating) | 15.6% | 100.0% | 100.0% | 15.3% | 634.0 | 0.7827 | 0.7827 | +0.00e+00 |
| FRD-LSTM (DCWR + BiLSTM) | 2.2% | 100.0% | 100.0% | 4.0% | 662.0 | 0.8173 | 0.8173 | +0.00e+00 |
| DeBERTa-v3 fine-tuned | 7.4% | 100.0% | 100.0% | 4.7% | 666.0 | 0.8222 | 0.8222 | +0.00e+00 |
| FRIDRC (text enc. + ViT, fused) | 0.0% | 100.0% | 100.0% | 0.2% | 674.0 | 0.8321 | 0.8321 | +0.00e+00 |
| BSTC (BERT + TextCNN) | 0.7% | 100.0% | 100.0% | 0.0% | 676.0 | 0.8346 | 0.8346 | +0.00e+00 |
| **OURS (recomputed now)** | 80.7% | 84.4% | 100.0% | 2.0% | 755.0 | 0.9321 | 0.9321 | +0.00e+00 |

Rows where the rebuilt accuracy disagrees with the reported accuracy by more than 1e-9: **0**.

## 2. Where each model loses its 810 rows (counts, not percentages)

| model | misses: edited photo | misses: fake text | misses: coordinated | false alarms | total mistakes | accuracy |
|---|---|---|---|---|---|---|
| Shan inconsistency + RF | 128 | 28 | 0 | 33 | 189 | 0.767 |
| A-LSTM + behaviour | 114 | 0 | 0 | 70 | 184 | 0.773 |
| DHMFRD-TER (text + emotion + rating) | 114 | 0 | 0 | 62 | 176 | 0.783 |
| FRD-LSTM (DCWR + BiLSTM) | 132 | 0 | 0 | 16 | 148 | 0.817 |
| DeBERTa-v3 fine-tuned | 125 | 0 | 0 | 19 | 144 | 0.822 |
| FRIDRC (text enc. + ViT, fused) | 135 | 0 | 0 | 1 | 136 | 0.832 |
| BSTC (BERT + TextCNN) | 134 | 0 | 0 | 0 | 134 | 0.835 |
| **OURS** | 26 | 21 | 0 | 8 | 55 | 0.932 |

## 3. The "98% not 100%" cases: the 21 fake-text reviews our model misses

Our fake-text recall is 0.8444 = 114/135. The missed rows:

| review_id | our score | rating | text |
|---|---|---|---|
| R00001 | 0.363 | 5 | Using it for daily gym use for three months now and the inner coating is as good as day one. Glad I went with  |
| R00125 | 0.389 | 5 | Been using this on a long flight for about eight months. The battery is as good as day one. Better than I expe |
| R00162 | 0.165 | 5 | Using it on wet pavement for three months now and the laces shows no wear at all. Exactly what I hoped for. |
| R00383 | 0.249 | 5 | About three months of use after a waist height drop onto tile and the port cutout still looks new. Really can' |
| R00495 | 0.462 | 5 | Using it for driving for about eight months now and the nose pads is as good as day one. Does everything it pr |
| R00823 | 0.374 | 5 | Running it at a small house party for two weeks and the pairing button still performs perfectly. Exactly what  |
| R00942 | 0.111 | 4 | Picked this up for daily morning practice about about eight months ago and the foam density is as good as day  |
| R01240 | 0.428 | 5 | Bought it for daily vegetable prep and three months later and the blade edge is as good as day one. Would happ |
| R01690 | 0.213 | 5 | About a month of use on a camping trip. The pairing button is flawless. Very pleased with this one. |
| R01937 | 0.369 | 4 | Running it in a heated class for six weeks. The surface texture is flawless. Would happily buy another. |
| R02177 | 0.187 | 5 | Using it for driving for a month now and the lens coating has not given me a moment of trouble. Would happily  |
| R02434 | 0.213 | 5 | Bought it on a hardwood floor and a month later and the edge binding shows no wear at all. Would happily buy a |
| R02677 | 0.211 | 4 | Had it two weeks, mostly at the gym. The bezel is well made, everything still works. Does everything it promis |
| R02814 | 0.361 | 5 | About half a year of use on a two week trip. The telescopic handle still looks new. Better than I expected hon |
| R02840 | 0.388 | 5 | Picked this up while swimming about a month ago. The crown still performs perfectly. Would happily buy another |
| R02871 | 0.491 | 5 | Had it half a year, mostly through a dry winter and the consistency is as good as day one. Does everything it  |
| R02922 | 0.083 | 4 | About about eight months of use for monthly work travel. The spinner wheels has not given me a moment of troub |
| R03249 | 0.327 | 5 | Picked this up in a home office about a month ago. The seat foam is as good as day one. Very pleased with this |
| R03265 | 0.362 | 5 | Been using this on daily commutes for two weeks and the mic is as good as day one. Exactly what I hoped for. |
| R03308 | 0.081 | 4 | Been using this for carrying a laptop and a lunch box for two weeks. The inner lining shows no wear at all. Do |
| R03348 | 0.360 | 5 | About half a year of use on wet pavement. The insole still looks new. Better than I expected honestly. |

Our score on those rows: 0.295 mean (threshold 0.5). On the 114 fake-text rows we do catch: 0.893 mean.
