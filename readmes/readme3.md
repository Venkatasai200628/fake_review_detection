# README 3: Comparing the 40 papers with our results

**Owner:** Venkata Sai · **Date:** 19 September 2026
**Previous:** `readme2.md`

---

## 0. What you asked for

1. Go through **every paper in the `papers` folder** and pull out accuracy, precision, recall, F1 and
   any other metric.
2. For each paper, give 2–3 simple lines: **synthetic or real data? If real, which dataset? Text only,
   or images too? What exactly?**
3. Make a **table** and **store it in the `papers` folder**.
4. **Compare with our results** and **explain why ours look lower**.

The result is **`D:\sem-5 projects\computer vision\papers\PAPERS_COMPARISON.md`**.

---

## 1. How I did it

1. I extracted the text of all 40 PDFs with `pdftotext` (between 3,400 and 20,600 words each).
2. A small Python scanner printed, for each paper: title and abstract, which datasets it mentions
   (Yelp, YelpChi/NYC/Zip, Amazon, Ott, OSF, Dianping, …), whether it mentions images, sentences that
   describe the dataset, and every sentence that contains a metric with a number.
3. For about 12 papers where that was not clear, I searched the text directly (for example the base paper's
   feature table, FRIDRC's image source, and the SIPUL results).
4. I wrote each paper's **own best result** into one table and marked uncertain cells with "≈".
   Three papers do not clearly name their dataset (21, 24, and 28, which is mostly a survey), and the table says so.

**Please double-check any "≈" number in the PDF before copying it into your report.** Tables in PDFs
sometimes come out scrambled when extracted.

---

## 2. What the file contains

1. **Master table** of the 31 papers with experiments: dataset, real/crowd-sourced/AI-generated/synthetic,
   modality (text / behaviour / graph / image), model, Acc, Prec, Rec, F1, AUC.
2. **Our own row**, measured three ways (balanced test, realistic 15% holdout, cross-validation).
3. **Survey papers** (9) listed separately, because they have no experiments of their own.
4. **2–3 lines per paper** in simple English.
5. **Big-picture counts.**
6. **Why our numbers look lower** (the analysis you asked for).
7. A **ready-to-use paragraph** for your report.

---

## 3. The most important findings

### 3.1 Almost nobody uses images
- Only **2 of 31** experimental papers use review images:
  - **He et al. 2022 (our base paper):** image features alone reach only **AUC 0.592**. They were its weakest feature.
  - **FRIDRC 2025:** RoBERTa + ViT on a self-built dataset, with images used as *context*.
- **Nobody** checks whether a photo was **edited** or **reused across accounts**. That is exactly our novelty,
  and this table is now the evidence for it.

### 3.2 The very high numbers come from easy data
| Kind of dataset | Typical result | Why |
|---|---|---|
| OSF 40k (half the reviews written by GPT-2) | 88–99% accuracy | It detects *machine writing style*, not paid humans |
| Ott 1,600 (Mechanical Turk workers) | 89–96% accuracy | Small, written in one session, easy to separate |
| Real Yelp (platform-filter labels) | F1 0.56–0.90, AUC 0.74–0.88 | Real, messy, imbalanced |
| He et al. (real Amazon, direct evidence) | AUC 0.932 | Best labels, product-level |

Proof inside the folder: paper 40 gets **78% on Amazon but 98% on OSF** with the same model. Paper 09 gets
F1 0.97 on its own GPT-2 data but only **AUC 0.70** on Ott's human-written fakes.

### 3.3 Why ours looks lower (short version)
1. **Different metric and class balance.** They report accuracy on 50/50 data. We report PR-AUC at a
   realistic 15% fake. **On their terms our model gets accuracy 0.852, F1 0.852, ROC-AUC 0.946**,
   which is right beside real-data papers like Elmogy (87.9%), Wang (84.5%) and Sun (0.89 / AUC 0.95).
2. **We made the dataset hard on purpose.** Lazy genuine reviews, convincing fake text, and fakes that only
   one modality can see.
3. **Stricter split.** A photo is never in both train and test.
4. **Simpler models.** 8 hand-made text counts plus a Random Forest, trained on 2,490 rows, while the top papers use
   BERT/RoBERTa and 20k–600k rows.
5. **Images are hard for everyone.** Our image-only model (ROC-AUC 0.744) is already **above** the base paper's
   image features (0.592).

**The fair comparison is our own ablation:** fusion **0.843** vs base-paper-equivalent **0.801**, p = 3e-08,
24/25 folds. Same data, same folds, one change. Never put 0.843 next to someone's 99% as if they measure
the same thing (guide §12.6).

### 3.4 What would honestly raise our numbers
RoBERTa text features (the biggest gain), a small CNN on ELA maps, more root photos, tuned decision bands,
and reporting balanced accuracy/F1 **alongside** PR-AUC so readers can compare.

---

## 4. Also done in this prompt
- Added **§21** to the bottom of `IMPLEMENTATION_GUIDE.md`, pointing to the comparison. The first 817 lines
  are still unchanged.
- The v6 dataset push you asked for in prompt 2 finished before this work started (see readme2 §3).
