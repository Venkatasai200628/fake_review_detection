# README 10: "The comparison table will show older papers as better than ours. What do we do?"

**Owner:** Venkata Sai · **Date:** 20 September 2026 · **Previous:** `readme9.md`

## 1. The problem
If the report has one table with "paper X: 99%" next to "ours: 94.3%", a reader will conclude ours is worse.
That conclusion is wrong, because the papers measured on *different, easier* datasets. But a table like that invites it.

## 2. What we must NOT do
- Inflate our numbers (remove the hard cases, leak labels, report only the easy test set).
- Hide the papers with higher numbers.
- Put numbers from different datasets in one ranked table (guide §12.6).

## 3. What we did: the standard research fix, a fair head-to-head
Research papers compare methods by **re-implementing the other methods and running them on the same data**. So I rebuilt
the five method families behind the 40 papers and ran each one on **our** dataset: same 2,490 training reviews,
same 810 unseen test reviews, same metrics.
Code: `dataset/fake_review_dataset_v6/code/compare_baselines.py` → `out/model_comparison/baselines_on_our_data.csv`.

| Method | Represents papers | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + linear SVM | 17, 23, 27, 29 | 0.802 | 0.775 | 0.833 | 0.763 | 4% | 100% | 7.7% |
| Fine-tuned transformer (RoBERTa family) | 09, 13, 26, 30, 38, 40 | 0.833 | 0.800 | 0.842 | 0.774 | 0% | 100% | 0.0% |
| Sentence embeddings + behaviour | 12, 15, 31 | 0.788 | 0.754 | 0.829 | 0.754 | 8% | 87% | 7.4% |
| He et al. 2022 (base paper) | 33 | 0.793 | 0.751 | 0.808 | 0.716 | 4% | 83% | 4.0% |
| Text + image embeddings (FRIDRC-style) | 11 | 0.851 | 0.835 | 0.898 | 0.827 | 36% | 92% | 5.7% |
| **OURS** | – | **0.943** | **0.943** | **0.976** | **0.958** | **82%** | 98% | 4.7% |

**Ours is best on every overall metric.** Against the strongest baseline (FRIDRC-style): **+9.3 accuracy points**
(95% CI +6.8 to +11.7). McNemar test p = 2×10⁻¹²: ours is right on 97 reviews that baseline gets wrong, and wrong on only 22 that it gets right.

**Why:** look at "Edited photo caught". Text methods get 0–8%, because the text is genuine. Image *embeddings* get 36%,
because an embedding describes *what* the photo shows, not *whether it was edited*. Only our forensic CNN reaches 82%.

## 4. How to write it in the report (three tables)
| Table | What it shows | Our row? |
|---|---|---|
| **A. Related work** | each paper's dataset, modality and reported numbers. Caption: *"reported on each paper's own dataset; not directly comparable"* | no, and don't sort it |
| **B. Methods re-implemented on our dataset** | the table above | yes: best, p = 2×10⁻¹² |
| **C. Detection by fraud type** | edited-photo / fake-text / coordinated columns | yes: explains *why* |

Sentence to include (it answers the obvious examiner question before it's asked):
> "Several text-only methods report 96–99% accuracy on their own benchmarks (e.g. GPT-2-generated reviews).
> Re-implemented on our dataset, the same method family reaches at most 83.3%, because one third of our fake reviews
> carry genuine-looking text and are detectable only from the photo."

The full, updated comparison is in `papers\PAPERS_COMPARISON.md`, section **1b**.

## 5. Optional extra (if the professor asks "but how does it do on THEIR data?")
Our **text branch** could be run on the public text datasets (Ott's 1,600 hotel reviews, the OSF 40k set).
This would show that our text part is competitive on their own benchmarks too. Their datasets have **no photos**, so the image part
(our novelty) cannot run there. That is exactly why we built our own dataset. It needs the datasets downloaded
from Kaggle; ask if you want it.
