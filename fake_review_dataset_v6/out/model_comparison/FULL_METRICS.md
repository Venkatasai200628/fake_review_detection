# Every metric, not just accuracy

Same 810 test reviews, same threshold (0.5). Ranked by **MCC**, which only goes high when all four cells of the confusion matrix are good -- the standard answer to "accuracy is not everything".

| model | Accuracy | Balanced acc | MCC | Kappa | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | PR-AUC @15% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **OURS** | **0.932** | **0.932** | **0.868** | **0.864** | **0.978** | **0.884** | **0.980** | **0.929** | **0.967** | **0.976** | **0.955** |
| BSTC (BERT + TextCNN) | 0.835 | 0.835 | 0.709 | 0.669 | 1.000 | 0.669 | 1.000 | 0.802 | 0.847 | 0.896 | 0.771 |
| FRIDRC (text enc. + ViT, fused) | 0.832 | 0.832 | 0.704 | 0.664 | 0.996 | 0.667 | 0.998 | 0.799 | 0.839 | 0.892 | 0.767 |
| DeBERTa-v3 fine-tuned | 0.822 | 0.822 | 0.668 | 0.644 | 0.936 | 0.691 | 0.953 | 0.795 | 0.853 | 0.898 | 0.776 |
| FRD-LSTM (DCWR + BiLSTM) | 0.817 | 0.817 | 0.662 | 0.635 | 0.945 | 0.674 | 0.960 | 0.787 | 0.848 | 0.895 | 0.764 |
| DHMFRD-TER (text + emotion + rating) | 0.783 | 0.783 | 0.570 | 0.565 | 0.824 | 0.719 | 0.847 | 0.768 | 0.829 | 0.885 | 0.744 |
| Shan inconsistency + RF | 0.767 | 0.767 | 0.560 | 0.533 | 0.883 | 0.615 | 0.919 | 0.725 | 0.802 | 0.861 | 0.699 |
| A-LSTM + behaviour | 0.773 | 0.773 | 0.549 | 0.546 | 0.806 | 0.719 | 0.827 | 0.760 | 0.837 | 0.889 | 0.748 |

## Is the gap real? 95% CI, 2,000 bootstrap resamples grouped by root photo

| model | Accuracy [95% CI] | MCC [95% CI] | ROC-AUC [95% CI] |
|---|---|---|---|
| **OURS** | 0.932 [0.913, 0.949] | 0.868 [0.832, 0.900] | 0.967 [0.953, 0.979] |
| BSTC (BERT + TextCNN) | 0.835 [0.815, 0.853] | 0.709 [0.675, 0.740] | 0.847 [0.814, 0.876] |
| FRIDRC (text enc. + ViT, fused) | 0.832 [0.813, 0.851] | 0.704 [0.669, 0.736] | 0.839 [0.806, 0.868] |
| DeBERTa-v3 fine-tuned | 0.822 [0.799, 0.844] | 0.668 [0.620, 0.710] | 0.853 [0.821, 0.880] |
| FRD-LSTM (DCWR + BiLSTM) | 0.817 [0.794, 0.839] | 0.662 [0.615, 0.705] | 0.848 [0.818, 0.876] |
| DHMFRD-TER (text + emotion + rating) | 0.783 [0.754, 0.811] | 0.570 [0.511, 0.627] | 0.829 [0.795, 0.859] |
| Shan inconsistency + RF | 0.767 [0.743, 0.792] | 0.560 [0.506, 0.611] | 0.802 [0.765, 0.834] |
| A-LSTM + behaviour | 0.773 [0.744, 0.801] | 0.549 [0.489, 0.607] | 0.837 [0.803, 0.868] |

Non-overlapping intervals mean the difference is not sampling noise.
