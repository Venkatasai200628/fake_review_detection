| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms | Ours better on / it better on (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|
| 02 Shan 2021 | Shan inconsistency + RF | 0.786 | 0.753 | 0.813 | 0.750 | 4% | 91% | 7.7% | 148 / 21 (p=1.0e-24) |
| 15 Xu 2024 | A-LSTM + behaviour | 0.816 | 0.794 | 0.841 | 0.753 | 13% | 100% | 7.7% | 125 / 22 (p=1.1e-18) |
| 38 Qayyum 2023 | FRD-LSTM (DCWR + BiLSTM) | 0.830 | 0.797 | 0.839 | 0.769 | 1% | 100% | 1.0% | 113 / 21 (p=2.0e-16) |
| 39 Duma 2024 | DHMFRD-TER (text + emotion + rating) | 0.788 | 0.772 | 0.825 | 0.773 | 16% | 100% | 14.6% | 149 / 23 (p=8.5e-24) |
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.833 | 0.800 | 0.834 | 0.755 | 0% | 100% | 0.0% | 111 / 22 (p=1.7e-15) |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.827 | 0.797 | 0.841 | 0.771 | 3% | 100% | 2.2% | 116 / 22 (p=1.3e-16) |
| 11 Hou 2025 | FRIDRC (text enc. + ViT, fused) | 0.833 | 0.800 | 0.846 | 0.782 | 0% | 100% | 0.0% | 111 / 22 (p=1.7e-15) |
| **ours** | **forensic fusion** | **0.943** | **0.943** | **0.976** | **0.958** | **82%** | 98% | 4.7% | – |

Simplifications (CPU-only re-builds):
- 02 Shan 2021: sentiment from DistilBERT-SST2; 9 inconsistency features + 8 base text features
- 15 Xu 2024: word embeddings trained from scratch
- 38 Qayyum 2023: DCWR approximated by frozen DistilBERT contextual token vectors
- 39 Duma 2024: emotion from a pretrained English emotion classifier (7 emotions)
- 30 Lu 2023: SKEP sentiment-knowledge branch omitted (not available in English/PyTorch)
- 40 Geetha 2025: deberta-v3-small; Monarch-Butterfly hyper-parameter search omitted
- 11 Hou 2025: image encoder = CLIP ViT-B/32, frozen (CPU); text encoder DistilRoBERTa fine-tuned; profile info not available
