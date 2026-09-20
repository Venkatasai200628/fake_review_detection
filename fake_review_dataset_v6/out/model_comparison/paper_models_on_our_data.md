| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms | Ours better on / it better on (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|
| 02 Shan 2021 | Shan inconsistency + RF | 0.791 | 0.762 | 0.852 | 0.766 | 20% | 81% | 8.6% | 140 / 12 (p=7.8e-29) |
| 15 Xu 2024 | A-LSTM + behaviour | 0.778 | 0.766 | 0.845 | 0.773 | 19% | 100% | 17.3% | 158 / 19 (p=1.8e-28) |
| 38 Qayyum 2023 | FRD-LSTM (DCWR + BiLSTM) | 0.817 | 0.787 | 0.848 | 0.764 | 2% | 100% | 4.0% | 121 / 14 (p=2.0e-22) |
| 39 Duma 2024 | DHMFRD-TER (text + emotion + rating) | 0.783 | 0.768 | 0.829 | 0.744 | 16% | 100% | 15.3% | 155 / 20 (p=4.6e-27) |
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.835 | 0.802 | 0.847 | 0.771 | 1% | 100% | 0.0% | 107 / 14 (p=6.5e-19) |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.822 | 0.795 | 0.853 | 0.776 | 7% | 100% | 4.7% | 119 / 16 (p=1.2e-20) |
| 11 Hou 2025 | FRIDRC (text enc. + ViT, fused) | 0.832 | 0.799 | 0.839 | 0.767 | 0% | 100% | 0.2% | 109 / 14 (p=2.1e-19) |
| **ours** | **forensic fusion** | **0.949** | **0.948** | **0.980** | **0.963** | **80%** | 94% | 1.5% | – |

Simplifications (CPU-only re-builds):
- 02 Shan 2021: sentiment from DistilBERT-SST2; 9 inconsistency features + 8 base text features
- 15 Xu 2024: word embeddings trained from scratch
- 38 Qayyum 2023: DCWR approximated by frozen DistilBERT contextual token vectors
- 39 Duma 2024: emotion from a pretrained English emotion classifier (7 emotions)
- 30 Lu 2023: SKEP sentiment-knowledge branch omitted (not available in English/PyTorch)
- 40 Geetha 2025: deberta-v3-small; Monarch-Butterfly hyper-parameter search omitted
- 11 Hou 2025: image encoder = CLIP ViT-B/32, frozen (CPU); text encoder DistilRoBERTa fine-tuned; profile info not available
