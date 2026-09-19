| Method | Represents papers | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | Coordinated caught | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| B1 TF-IDF + linear SVM | 17, 23, 27, 29 | 0.802 | 0.775 | 0.833 | 0.763 | 4% | 100% | 100% | 7.7% |
| B2 fine-tuned transformer (DistilRoBERTa) | 09, 13, 26, 30, 38, 40 | 0.833 | 0.800 | 0.842 | 0.774 | 0% | 100% | 100% | 0.0% |
| B3 sentence embeddings + behaviour | 12, 15, 31 | 0.788 | 0.754 | 0.829 | 0.754 | 8% | 87% | 100% | 7.4% |
| B4 He et al. 2022 (base paper, with image sim.) | 33 | 0.793 | 0.751 | 0.808 | 0.716 | 4% | 83% | 100% | 4.0% |
| B5 multimodal embeddings (FRIDRC-style) | 11 | 0.851 | 0.835 | 0.898 | 0.827 | 36% | 92% | 100% | 5.7% |
| OURS forensic fusion | - | 0.943 | 0.943 | 0.976 | 0.958 | 82% | 98% | 100% | 4.7% |
