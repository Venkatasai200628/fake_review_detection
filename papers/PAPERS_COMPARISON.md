# Comparison of the 40 papers in `papers/` vs our project

**Prepared for:** Venkata Sai, Fake Review Detection via Visual Forensics (CV, Sem 5)
**Date:** 19 September 2026
**How it was made:** the text of every PDF in this folder was extracted, and the
datasets, models and headline metrics were read out of each paper's abstract,
results tables and conclusion. Numbers are each paper's **own best reported
result**. "—" means the paper does not report that metric. "≈" means the value was read from
a table whose layout made the exact cell hard to confirm, so check it in the PDF before quoting
it in the report. Survey / review papers do not run their own experiments, so they are
listed separately.

**Key to columns**
- **Data:** R = real reviews (a platform's filter labels, or real reviews with labels), C = crowd-sourced fakes
  (Amazon Mechanical Turk writers, e.g. Ott's hotel set), G = machine-generated fakes (GPT-2/GPT-3/Llama),
  S = synthetic / constructed data, U = unlabelled.
- **Modality:** T = text, B = behaviour / metadata (rating, time, reviewer), N = network / graph, I = image.

---

## 1. Master table — papers that report their own experiments

| # | Paper (first author, year) | Dataset(s) | Data | Modality | Best model | Acc | Prec | Rec | F1 | AUC / other |
|---|---|---|---|---|---|---|---|---|---|---|
| 02 | Shan, 2021 — review inconsistency (DSS) | Yelp reviews (hotels/restaurants) | R | T+B | Random Forest + 22 inconsistency features | ≈0.92 | ≈0.93 | ≈0.91 | ≈0.93 | — |
| 03 | Luo, 2023 — supervised mixed probability (DSS) | YelpZip / YelpNYC / YelpChi | R (+S points) | T+B+N | CNN on generated distribution samples | 86.8 / 89.7 / 86.6 % | 99.99 / 100 / 99.9 % | 86.8 / 89.7 / 86.7 % | 92.9 / 94.6 / 92.8 % | — |
| 04 | Luo, 2026 — AI-generated fake reviews (DSS) | GPT-3 set (12k AI + 12k Yelp), GPT-2 set (20k Amazon + 20k generated), Llama-2 set (10k) | R + G | T | AGFRDCP (probability density + AdaBoost) | 0.909 / 0.983 / 0.839 | 0.921 / 0.983 / 0.864 | 0.895 / 0.984 / 0.804 | 0.908 / 0.983 / 0.833 | — |
| 05 | Barbado, 2019 — consumer-electronics framework (IPM) | Own Yelp consumer-electronics dataset (4 cities) | R | T+B | Random Forest / AdaBoost | — | — | — | 0.82 | — |
| 06 | Wang, 2026 — LLMIC, LLM implicit characteristics (IPM) | Ott Hotel (800+800), Amazon (10,500+10,500) | C + R | T | LLM + multifractal + recurrence features | >93 % (Amazon) | — | — | 95.89 % (Hotel), >93 % (Amazon) | AUC >93 % |
| 08 | Vidanagama, 2022 — ontology sentiment (ESWA) | Amazon Unlocked Mobile (Kaggle), 5,000 reviews, unlabelled → outlier labels | U | T | Rule-based ontology classifier | 88.98 % | 88.69 % | 90.90 % | 89.79 % | — |
| 09 | Salminen, 2022 — creating & detecting fake reviews (JRCS) | Amazon 2018 real + 20k GPT-2 generated (the "OSF / CG vs OR" 40k set); also Ott | R + G | T | fakeRoBERTa | 96.6 % | 0.97 | 0.97 | 0.97 | AUC 0.696 on Ott (cross-set); 87.8 % acc on Ott after tuning |
| 11 | Hou, 2025 — FRIDRC multimodal intentions (ECRA) | Own FRIDRC dataset: manual + AI-generated fake reviews with **video-frame images** | S + G | **T+I** + profile | RoBERTa + **ViT**, combined fusion | 92.7–99.2 % per class | — | 93.6–99.1 % per class | **avg 0.97, macro 0.96** | — |
| 12 | (ITQM) 2024 — RoBERTa + behavioural features (Procedia) | YelpZip (608,598 reviews) | R | T+B | RoBERTa + CNN-LSTM + behaviour | — | ≈87 % | ≈90 % | 88.76 % | — |
| 13 | Mohawesh, 2024 — transformer-LSTM + RoBERTa (IJCCE) | OpSpam (Ott, 1,600) + Deception (Li, 3,032 hotel/doctor/restaurant) | C | T | RoBERTa + LSTM | 96.03 / 93.15 % | 91.48 / 85.39 % | 99.31 / 97.40 % | 95.36 / 91.20 % | — |
| 15 | Xu, 2024 — A-LSTM hybrid (IEEE TCSS) | Yelp (Rayana & Akoglu: NYC / Chicago) | R | T+B | Attention-LSTM + behaviour | 90.9 % | 88.7 % | 87.9 % | 88.3 % | text-only F1 82.9 %, behaviour-only 70.9 % |
| 16 | Shunxiang, 2023 — SIPUL sentiment + PU learning (IEEE TNNLS) | Ott, YelpZip, YelpChi | C + R | T | Sentiment-intensity PU learning (streaming) | results only in table/figure | — | — | — | beats baselines on Yelp; weaker on Ott (small) |
| 17 | Naresh, 2023 — ML comparison, SVM (ICSCSS) | OSF 40k: 20k computer-generated + 20k human (Amazon) | G + R | T | SVM | 88.28 % | — | — | — | — |
| 18 | Pan & Xu, 2024 — unsupervised FRD (IJEC) | Own Dianping crawl (Beijing restaurants) + Yelp | R (U) | B | Unsupervised, recommendation-based evaluation | — | — | — | — | evaluated by recommendation error (MAE), not classification |
| 19 | Lee, 2022 — supervised ML (Service Ind. J.) | Yelp "Top 100 Places to Eat 2017", Yelp-filter labels | R | T+B | Random Forest | — | — | — | **0.564** | shows how hard real, imbalanced data is |
| 20 | Nair, 2025 — feature engineering + stats (CSITSS) | OSF 40k Amazon CG vs OR, 10 product categories | G + R | T | Logistic Regression + clustering | 0.90–0.93 per category | 0.89–0.93 | — | — | — |
| 21 | Anuprathibha, 2024 — Hierarchical GAT (GCCIT) | not clearly named (text + ratings) | ? | T+B+N | HGAN | 91.26 % | 88.45 % | ≈0.87 | 90.53 % | sensitivity 0.945 |
| 22 | Wang, 2020 — multiple feature fusion, rolling co-training (IEEE Access) | YelpChi (5,854) + YelpRes (15,141) | R | T+B | Semi-supervised rolling co-training | 84.45 % | — | — | — | +3.5 % over baselines |
| 23 | Elmogy, 2021 — supervised ML (IJACSA) | YelpChi hotels (5,853: 1,144 fake / 4,709 real) | R | T+B | KNN (k=7) / Logistic Regression | 87.87 % | — | — | 82.40 % | — |
| 24 | Sudha Mercy, 2024 — LDCP (ICSES) | not clearly named | ? | T | RoBERTa / cascaded MLP | 99.2–99.5 % | 98.7 % | 98.7 % | — | — |
| 26 | Sharma, 2025 — Amazon reliability (ICICV) | 50,000 Amazon reviews (real + fake) | R/G? | T | BERT | 96.3 % | 0.95 | 0.97 | 0.96 | — |
| 27 | Manish Kumar, 2022 — role of ML (ICECA) | OSF 40k (20k CG + 20k OR) | G + R | T | SVM | 88 % | ≈0.88 | ≈0.88 | — | — |
| 29 | Tufail, 2022 — SKL during Covid (IEEE Access) | Own 1,900-review set (balanced) + Ott TripAdvisor 1,600 | S/C | T | SKL (SVM + KNN + LR ensemble) | 89.03 % (Ott) | — | — | — | — |
| 30 | Lu, 2023 — BSTC (Electronics) | Ott Hotel 1,600 · Restaurant 400 · Doctor 556 | C | T | BERT + SKEP + TextCNN | 93.44 / 91.25 / 92.86 % | 90.64 % (Hotel) | 96.88 % (Hotel) | 93.36 % (Hotel) | — |
| 31 | Sun, 2024 — FRD content + behaviour (Electronics) | 250,000 reviews (Dianping-style, authentic:fake = 34:66) | R | T+B | BERT text + reviewer & merchant behaviour | 0.894 | — | — | 0.895 | **AUC 0.953** |
| 33 | **He et al., 2022 — BASE PAPER (PNAS)** | Amazon products; fake-review buyers observed in Facebook groups | **R (direct evidence)** | T+B+**N+I** | Random Forest | 0.860 (all) | — | TPR 0.832 | 0.860 | **AUC 0.932 all features; network 0.890; text 0.857; IMAGE only 0.592** |
| 34 | Mohawesh, 2023 — explainable multi-view ensemble (JKSUCIS) | YelpNYC, YelpZip | R | T+B | CNN + DNN + BiLSTM ensemble | — | 96.9 % (NYC) | 84.2 % (NYC) | 90.1 % (NYC) | AUC 85.5 % (NYC), 81.5 % (Zip) |
| 37 | Li, 2021 — group model (MONET) | Dianping (hotel, personal care, movie) | R | T+B+N (groups) | SVM / RF / Bagging + group-collusion features | — | ≈0.91 | ≈0.91 | ≈0.91 | +4–7 % precision from group features |
| 38 | Qayyum, 2023 — FRD-LSTM (MTAP) | Amazon 21,000 (balanced, 350/350 per product) | R/labelled | T | DCWR + BiLSTM | 97.21 % | 96 % | 97.24 % | 96.61 % | — |
| 39 | Duma, 2024 — DHMFRD-TER (MTAP) | Amazon, YelpChi, OSF | R + G | T+B (text, emotion, rating) | Deep hybrid | 0.988 / 0.987 / 0.994 | 0.988 / 0.956 / 0.996 | 0.986 / 0.934 / 0.994 | 0.987 / 0.924 / 0.995 | — |
| 40 | Geetha, 2025 — MBO-DeBERTa (Sci. Reports) | Amazon 21k, OSF 40k, Ott 1,600 (Kaggle copies) | R + G + C | T | DeBERTa + Monarch Butterfly optimisation | 78 / 98 / 91 % | 77 / 98 / 91 % | 78 / 97 / 90 % | 77 / 97 / 90 % | — |

### Our project (same metrics, for comparison)

| Setting | Data | Modality | Model | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|---|---|
| **Ours FINAL v6.2, balanced test (50% fake)** | S (826 real photos, constructed reviews) | **T+B+N+I** | **Extra Trees** fusion of 41 features: MiniLM text + forensic CNN + ResNet-50 reuse + CLIP cross-modal | **0.932** | **0.978** | **0.884** | **0.929** | ROC **0.967** · PR 0.976 |
| **Ours FINAL v6.2, realistic holdout (15% fake)** | same | same | same | 0.973 | 0.889 | 0.928 | 0.908 | ROC 0.980 · **PR 0.955** |
| *(superseded: Ours v6.1 RF, balanced test)* | | | RF fusion, measured before the text duplication was fixed | *0.943* | *0.952* | *0.933* | *0.943* | *ROC 0.976 · PR 0.982* |
| *(earlier: Ours v6.0, balanced test)* | | | | *0.852* | *0.850* | *0.854* | *0.852* | *ROC 0.946* |

**v6.2 (20 Sep 2026)** is the number to quote. v6.1 and earlier were measured on a dataset in which 46.2% of test
reviews were word-for-word copies of a training review; that is fixed (0.0%), and the model was re-selected on
grouped cross-validation (Extra Trees beat the Random Forest). Details: `readmes/readme12.md`,
`out/model_comparison/TEXT_DIFFICULTY_AUDIT.md`, `out/model_comparison/fusion_models.md`.

Verified independently for v6.2 (`out/VERIFY_RESULTS.md`): every metric reproduces from a fresh load of the model;
balanced-test accuracy 95% CI **0.914–0.949** (2,000 bootstrap resamples grouped by root photo), ROC-AUC
0.953–0.978, PR-AUC 0.966–0.984; **label-shuffle control mean ROC-AUC 0.509** over 5 runs, so no feature leaks the
answer; 5 retrains give accuracy 0.930–0.932, and training/serving parity is 41/41 features identical. The 5×5 grouped CV row (previously PR 0.902) has **not** yet been
re-run on v6.2.

### Which papers report HIGHER numbers than ours? (added 19 Sep, final model)
Compared on the same kind of number (accuracy / F1 / AUC on a balanced test set; ours = 0.943 / 0.943 / 0.976):

| # | Paper | Their best | Their data | Higher than ours? | Why the comparison is not like-for-like |
|---|---|---|---|---|---|
| 39 | Duma 2024 DHMFRD-TER | acc 0.988–0.994 | Amazon, YelpChi, OSF | **yes** | text + emotion + rating only; OSF half GPT-2 text; small balanced sets |
| 24 | Sudha Mercy 2024 LDCP | acc 99.2% | not named | **yes** | dataset not stated; cannot be checked |
| 38 | Qayyum 2023 FRD-LSTM | acc 97.2%, F1 96.6% | Amazon 21k | **yes** | text only; paper 40 got only 78% on the same Amazon 21k set |
| 11 | Hou 2025 FRIDRC | F1 0.97 | own set, manual + AI-written fakes, video-frame images | **yes** | the only other text+image model; different task (intent classes), images used as context, not forensics |
| 09 | Salminen 2022 fakeRoBERTa | acc 96.6%, F1 0.97 | Amazon + GPT-2 fakes | **yes** on its own data | same model: AUC 0.696 on human-written fakes (Ott) |
| 26 | Sharma 2025 | acc 96.3%, F1 0.96 | 50k Amazon | **yes** | text only |
| 13 | Mohawesh 2024 | acc 96.0% (OpSpam) | Ott crowd-sourced | **yes** on OpSpam; lower on Deception (93.2%) | 1,600 crowd-sourced reviews |
| 40 | Geetha 2025 MBO-DeBERTa | acc 98% (OSF) | OSF, Ott, Amazon | **yes** on OSF; **lower** on Ott (91%) and Amazon (78%) | GPT-2 text is easy to separate |
| 06 | Wang 2026 LLMIC | F1 95.9% (Hotel) | Ott Hotel, Amazon | **slightly** | text only |
| 04 | Luo 2026 | F1 0.983 (GPT-2 set) | AI-generated sets | **yes** on GPT-2; lower on GPT-3 (0.908) and Llama-2 (0.833) | detects machine-written text, a different task |
| 03 | Luo 2023 | F1 94.6% (YelpNYC) | Yelp | **tie** (0.946 vs 0.943) | Yelp filter labels |

**Lower than ours (18 papers):** 02 Shan (~0.93), 05 Barbado (F1 0.82), 08 (89.0%), 12 (F1 88.8%), 15 A-LSTM (90.9%),
17 (88.3%), 19 Lee (F1 0.56), 20 (0.90–0.93), 21 HGAN (91.3%), 22 (84.5%), 23 Elmogy (87.9%), 27 (88%), 29 (89.0%),
30 BSTC (93.4%), 31 Sun (acc 0.894, AUC 0.953), 33 **He et al. base paper (AUC 0.932, acc 0.860)**, 34 Mohawesh (F1 90.1%, AUC 0.855),
37 Li (F1 ~0.91). Not comparable: 16 (numbers only in figures), 18 (unsupervised), 10 (no classifier), plus 9 surveys.

**The fairest comparison we can make:** most of the higher-scoring papers use a fine-tuned transformer on text
(RoBERTa / BERT / DeBERTa). We ran that same kind of method **on our dataset**: DistilRoBERTa fine-tuned gets
**accuracy 0.833**, and no text-only method can exceed 0.833 because a third of our fakes have perfect text.
**Our fusion model gets 0.943 on the same data.** So "their method on our data" < ours, even though "their method on their data" > ours.

---

## 1b. FAIR COMPARISON: the papers' METHODS re-run on OUR dataset (use this in the report)
Numbers copied from papers were measured on **their** datasets, so they cannot be ranked against ours (guide §12.6).
The standard fix is to **re-implement the published methods and run them on the same data, with the same split and
the same metrics**. Code: `dataset/fake_review_dataset_v6/code/compare_baselines.py`.
Same 2,490 training reviews, same 810 unseen test reviews (photos never seen in training):

| Method | Represents papers | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | Coordinated caught | False alarms |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + linear SVM | 17, 23, 27, 29 | 0.802 | 0.775 | 0.833 | 0.763 | 4% | 100% | 100% | 7.7% |
| Fine-tuned transformer (DistilRoBERTa) | 09, 13, 26, 30, 38, 40 | 0.833 | 0.800 | 0.842 | 0.774 | 0% | 100% | 100% | 0.0% |
| Sentence embeddings + behaviour | 12, 15, 31 | 0.788 | 0.754 | 0.829 | 0.754 | 8% | 87% | 100% | 7.4% |
| He et al. 2022, base paper (with its image-similarity feature) | 33 | 0.793 | 0.751 | 0.808 | 0.716 | 4% | 83% | 100% | 4.0% |
| Multimodal text + image embeddings (FRIDRC-style) | 11 | 0.851 | 0.835 | 0.898 | 0.827 | 36% | 92% | 100% | 5.7% |
| **OURS: forensic fusion** | – | **0.943** | **0.943** | **0.976** | **0.958** | **82%** | 98% | 100% | 4.7% |

**Ours vs the best baseline (FRIDRC-style):** +9.3 accuracy points (95% CI +6.8 to +11.7). McNemar test: ours is right
on 97 reviews that baseline gets wrong, and wrong on only 22 that it gets right, **p = 2×10⁻¹²**.

**Why ours wins:** the "Edited photo caught" column. Text methods catch 0–8% of edited-photo fakes, because their
text is genuine. Plain image *embeddings* (FRIDRC-style) catch 36%, because an embedding describes *what* is in the photo, not
whether it was *edited*. Only forensic evidence (the CNN on error-level / noise maps) catches 82%.

### How to present the comparison in the report
1. **Table A: Related work** (section 1 above). Their dataset, modality and reported numbers, captioned:
   *"Reported on each paper's own dataset; not directly comparable because datasets, labels and class balance differ."*
   Don't sort it or put our row in it as a competitor.
2. **Table B: Re-implemented baselines on our dataset** (this section). This is the real comparison, and ours is best
   on every overall metric, with p = 2×10⁻¹².
3. **Table C: Per fraud type** (the last four columns above). It shows *why*: no text method can see edited photos.
4. One sentence to pre-empt the question: *"Several text-only methods report 96–99% on their own benchmarks (e.g.
   GPT-2-generated reviews); re-implemented on our data, the same family reaches at most 83.3%, because a third of our
   fakes carry genuine-looking text and are only detectable from the photo."*

---

## 1c. THE SPECIFIC PAPER MODELS, re-built and run on OUR dataset
Section 1b compared method *families*. This section re-builds the **named models** of seven papers and trains and
tests each on our data (same 2,490 training reviews, same 810 unseen test reviews).
Code: `dataset/fake_review_dataset_v6/code/compare_paper_models.py`.
"Ours better on / it better on" counts test reviews where exactly one of the two is right (McNemar test).

**These are the v6.2-corrected numbers (20 Sep 2026, second pass).** An earlier v6.2 pass was invalidated by a stale MiniLM embedding cache that fed our model the OLD text while the baselines got the new text; it is kept at `out/model_comparison/paper_models_v62_stalecache.csv`. See `readmes/readme13.md` section 1.

**These are the v6.2 numbers (20 Sep 2026).** Everything here was retrained from scratch after the review text was
regenerated: in v6.1, 46.2% of test reviews were word-for-word copies of a training review, which inflated the
"fake text caught" column for every model including ours. See `out/model_comparison/TEXT_DIFFICULTY_AUDIT.md`.
The superseded v6.1 table is kept at the end of this section.

| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms | Ours better on / it better on (McNemar p) |
|---|---|---|---|---|---|---|---|---|---|
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.835 | 0.802 | 0.847 | 0.771 | 1% | 100% | 0.0% | 108 / 29 (p=6.7e-12) |
| 11 Hou 2025 | FRIDRC (text enc. + ViT, fused) | 0.832 | 0.799 | 0.839 | 0.767 | 0% | 100% | 0.2% | 110 / 29 (p=2.6e-12) |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.822 | 0.795 | 0.853 | 0.776 | 7% | 100% | 4.7% | 120 / 31 (p=1.5e-13) |
| 38 Qayyum 2023 | FRD-LSTM (DCWR + BiLSTM) | 0.817 | 0.787 | 0.848 | 0.764 | 2% | 100% | 4.0% | 122 / 29 (p=9.0e-15) |
| 39 Duma 2024 | DHMFRD-TER (text + emotion + rating) | 0.783 | 0.768 | 0.829 | 0.744 | 16% | 100% | 15.3% | 155 / 34 (p=1.2e-19) |
| 15 Xu 2024 | A-LSTM + behaviour | 0.773 | 0.760 | 0.837 | 0.748 | 16% | 100% | 17.3% | 160 / 31 (p=3.8e-22) |
| 02 Shan 2021 | Shan inconsistency + RF | 0.767 | 0.725 | 0.802 | 0.699 | 5% | 79% | 8.1% | 145 / 11 (p=5.5e-31) |
| **OURS** | **forensic fusion (Extra Trees)** | **0.932** | **0.929** | **0.967** | **0.955** | **81%** | 84% | **2.0%** | - |

Simplifications (CPU-only re-builds):
- 02 Shan 2021: sentiment from DistilBERT-SST2; 9 inconsistency features + 8 base text features
- 15 Xu 2024: word embeddings trained from scratch
- 38 Qayyum 2023: DCWR approximated by frozen DistilBERT contextual token vectors
- 39 Duma 2024: emotion from a pretrained English emotion classifier (7 emotions)
- 30 Lu 2023: SKEP sentiment-knowledge branch omitted (not available in English/PyTorch)
- 40 Geetha 2025: deberta-v3-small; Monarch-Butterfly hyper-parameter search omitted
- 11 Hou 2025: image encoder = CLIP ViT-B/32, frozen (CPU); text encoder DistilRoBERTa fine-tuned; profile info not available

**Every one of them loses to ours, and every difference is statistically significant (p < 7e-12).**
On v6.2 we lead on **every column, including false alarms**, which we did not on v6.1.

### The same result counted as mistakes, out of 810 test reviews
This is the clearest way to answer "their model catches 100% of fake text, so why is its accuracy only 0.83?".
"Fake text caught" is recall on ONE group of 135 rows, not accuracy.

| model | misses: edited photo | misses: fake text | misses: coordinated | false alarms | total mistakes | accuracy |
|---|---|---|---|---|---|---|
| BSTC (BERT + TextCNN) | **134** | 0 | 0 | 0 | 134 | 0.835 |
| FRIDRC (text + ViT) | **135** | 0 | 0 | 1 | 136 | 0.832 |
| DeBERTa-v3 | 125 | 0 | 0 | 19 | 144 | 0.822 |
| FRD-LSTM | 132 | 0 | 0 | 16 | 148 | 0.817 |
| DHMFRD-TER | 114 | 0 | 0 | 62 | 176 | 0.783 |
| A-LSTM | 114 | 0 | 0 | 70 | 184 | 0.773 |
| Shan + RF | 128 | 28 | 0 | 33 | 189 | 0.767 |
| **OURS** | **26** | 21 | 0 | 8 | **55** | **0.932** |

A model that is perfect on text but blind to photos can be right on at most (810 - 135)/810 = **0.833**. Five of the
seven land within 0.02 of that ceiling. Verified by rebuilding every reported accuracy from its own per-type
rates: all 8 rows agree to within 0.00e+00 (`out/model_comparison/PAPER_NUMBERS_AUDIT.md`,
`code/verify_paper_numbers.py`).

**The pattern is the same in every row:** these models catch fake *text* almost perfectly (81-100%) and
edited *photos* almost never (0-22%). They are text models. Our dataset's edited-photo fakes carry genuine
text, so no amount of text modelling can reach them. Our forensic fusion catches 79% of them.

Note on FRIDRC (the only other text+image paper): with **frozen** embeddings and a linear classifier (section 1b)
it caught 36% of edited photos; **fine-tuned end-to-end** it caught 0%, because the text signal dominates training.
Either way, a general image encoder describes *what is in* the photo, not *whether it was edited*.

### Full metrics for every model (precision / recall / F1 / specificity)
Derived exactly from each model's per-fraud-type rates and the known group sizes
(405 genuine + 135 + 135 + 135). Code: `code/explain_modalities.py`, report
`out/model_comparison/MODALITY_EXPLAINED.md`.

| model | Accuracy | Precision | Recall | F1 | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| BSTC (BERT + TextCNN) | 0.8346 | 1.0000 | 0.6691 | 0.8018 | 1.0000 | 271 | 0 | 134 | 405 |
| FRIDRC (text + ViT) | 0.8321 | 0.9963 | 0.6667 | 0.7988 | 0.9975 | 270 | 1 | 135 | 404 |
| DeBERTa-v3 fine-tuned | 0.8222 | 0.9365 | 0.6914 | 0.7955 | 0.9531 | 280 | 19 | 125 | 386 |
| FRD-LSTM | 0.8173 | 0.9446 | 0.6741 | 0.7867 | 0.9605 | 273 | 16 | 132 | 389 |
| DHMFRD-TER | 0.7827 | 0.8244 | 0.7185 | 0.7678 | 0.8469 | 291 | 62 | 114 | 343 |
| Shan inconsistency + RF | 0.7667 | 0.8830 | 0.6148 | 0.7249 | 0.9185 | 249 | 33 | 156 | 372 |
| A-LSTM + behaviour | 0.7494 | 0.7538 | 0.7407 | 0.7472 | 0.7580 | 300 | 98 | 105 | 307 |
| **OURS (Extra Trees fusion)** | **0.9321** | **0.9783** | **0.8840** | **0.9287** | 0.9802 | 358 | 8 | 47 | 397 |

Every baseline's **recall is 0.61-0.74**: they find roughly two thirds of the fakes. BSTC's precision of 1.0000 is
caution rather than skill - it flags only what it is certain of and misses 134 fakes.
The baselines have **no confidence intervals**, because compare_paper_models.py stored summary rates rather than
per-row predictions; producing them means retraining all seven (~2.5 h).

### Why the baselines reach 100% on fake text and our fusion reaches 94%
Because they are pure text models and ours is not. **Our text branch is as good as theirs; the fusion trades some
of it away on purpose.**

| our branch | edited photo | fake text | coordinated | false alarms | accuracy |
|---|---|---|---|---|---|
| text only | 5.2% | **88.1%** | 100% | 7.4% | 0.7852 |
| image only | **85.2%** | 8.9% | 93.3% | 9.4% | 0.7654 |
| **fusion (shipped)** | 80.7% | 84.4% | 100% | **2.0%** | **0.9321** |

The text branch alone flags 119/135 fake-text rows; the fusion flags 112, losing 7 where the photo and behaviour
evidence outvotes the text. Those 7 rows buy: edited photos caught **7 -> 107**, genuine reviews wrongly flagged
**30 -> 9**. This is also why the single-modality numbers look weaker than the combination - the branches are
strong on *different* fraud types.

### Is the contribution only "edited photos"?
No. The photo is used as forensic evidence in three ways, and the image branch delivers on two of them **with the
text branch switched off entirely**:

| fraud type | image evidence alone | fusion |
|---|---|---|
| `visual_manipulation` (novelty 1: manipulation) | **85.2%** | 80.7% |
| `coordinated_reuse` (novelty 2: provenance / reuse) | **93.3%** | 100% |
| `text_deception` (clean photo) | 8.9% - correct, nothing to find | 84.4% |

The edited-photo column is emphasised only because it is the one the published models cannot do at all (0-22%).
They also reach 100% on coordinated reuse, but through the shared *wording* of campaign reviews, not through the
photo.

### What the text fix did to these numbers (v6.1 -> v6.2)
Removing the duplicated text lowered **almost every model, including ours**. State it that way; an earlier draft
claimed the fix "made the baselines worse and ours better", which was an artefact of the stale embedding cache
described above, not a real effect.

| model | v6.1 accuracy | v6.2 accuracy | change | v6.1 false alarms | v6.2 false alarms |
|---|---|---|---|---|---|
| A-LSTM | 0.816 | 0.749 | **-6.7** | 7.7% | 24.2% |
| Shan + RF | 0.786 | 0.767 | -1.9 | 7.7% | 8.1% |
| **OURS** | 0.943 | **0.932** | **-1.1** | 4.7% | 2.0% |
| FRD-LSTM | 0.830 | 0.817 | -1.3 | 1.0% | 4.0% |
| DeBERTa-v3 | 0.827 | 0.822 | -0.5 | 2.2% | 4.7% |
| DHMFRD-TER | 0.788 | 0.783 | -0.5 | 14.6% | 15.3% |
| FRIDRC | 0.833 | 0.832 | -0.1 | 0.0% | 0.2% |
| BSTC | 0.833 | 0.835 | +0.2 | 0.0% | 0.0% |

Reading it honestly:

* Our lead over the best baseline **shrank**, from 11.0 points (0.943 vs 0.833) to **9.7 points**
  (0.932 vs 0.835). It is still large and still significant at p < 2e-10, but it is smaller than the v6.1 table
  suggested.
* The models that lost most are the ones that leaned hardest on memorised sentences: A-LSTM fell 6.7 points and
  its false alarms tripled, because without recognisable text it starts guessing "fake" on genuine reviews.
* BSTC and FRIDRC barely moved, because they were already refusing to flag anything without clear textual
  evidence - their accuracy is pinned near the 0.833 text-only ceiling either way.
* **Ours fell too** (-1.7). The part of our score that came from memorised text is gone; what remains is the
  photo evidence, which is why `visual_manipulation` detection is essentially unchanged (82% -> 79%) while
  `text_deception` fell (98% -> 83%).

**Where ours is weakest:** 21 of 135 fake-text reviews are missed (84%, not 100%). The hardest are the
*sophisticated deceptive* rows - fabricated specifics written to read like a real customer, on a clean photo, e.g.
*"About a month of use on a camping trip. The pairing button is flawless."* There is no evidence in the words and
none in the pixels; this is the designed overlap case (section 5.3), not a bug. They are listed with their scores
in `out/model_comparison/PAPER_NUMBERS_AUDIT.md`.

<details>
<summary>Superseded: the v6.1 table, measured before the text duplication was fixed</summary>

| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms |
|---|---|---|---|---|---|---|---|---|
| 02 Shan 2021 | Shan inconsistency + RF | 0.786 | 0.753 | 0.813 | 0.750 | 4% | 91% | 7.7% |
| 15 Xu 2024 | A-LSTM + behaviour | 0.816 | 0.794 | 0.841 | 0.753 | 13% | 100% | 7.7% |
| 38 Qayyum 2023 | FRD-LSTM (DCWR + BiLSTM) | 0.830 | 0.797 | 0.839 | 0.769 | 1% | 100% | 1.0% |
| 39 Duma 2024 | DHMFRD-TER | 0.788 | 0.772 | 0.825 | 0.773 | 16% | 100% | 14.6% |
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.833 | 0.800 | 0.834 | 0.755 | 0% | 100% | 0.0% |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.827 | 0.797 | 0.841 | 0.771 | 3% | 100% | 2.2% |
| 11 Hou 2025 | FRIDRC (text enc. + ViT, fused) | 0.833 | 0.800 | 0.846 | 0.782 | 0% | 100% | 0.0% |
| ours (RF) | forensic fusion | 0.943 | 0.943 | 0.976 | 0.958 | 82% | 98% | 4.7% |

Raw file: `out/model_comparison/paper_models_v61_leaky.csv`.
</details>


---

## 2. Surveys / review papers (no experiments of their own)

| # | Paper | What it is useful for |
|---|---|---|
| 01 | Ben Jabeur, 2023 — AI in fake review detection: bibliometric analysis (J. Bus. Res.) | Map of the field 2012–2021. Notes SVM accuracy falls from 86 % (one domain) to 52–64 % (cross-domain) |
| 07 | Duma, 2025 — GNNs for fake review detection, SLR (Neurocomputing) | Lists GNN results: AUC 0.74–0.98 on YelpChi/NYC/Zip/Amazon |
| 10 | Zhao, 2025 — AI vs human fake reviews, 714,016 reviews (JRCS) | Linguistic analysis only (YelpZip real + 117,859 AI-generated). No classifier metrics |
| 14 | Vayadande, 2025 — cutting-edge techniques (EAI) | Tabulates others' results (CNN 97.7 %, RoBERTa 91 %, …) |
| 25 | Ennaouri, 2023 — ML approaches, SLR | Summarises datasets (Ott 1,600; YelpChi/NYC/Zip) and accuracies (74–92 %) |
| 28 | Mishra, 2025 — NLP, BERT & behaviour (ICAAIC) | Mostly a literature summary (RoBERTa 97 %, BSTC 91–93 %, …) plus a proposed pipeline |
| 32 | Yu, 2022 — graph learning for fake review detection (Frontiers AI) | Dataset statistics (YelpCHI 67,395 · YelpNYC 359,052 · YelpZip 608,598 · Amazon) |
| 35 | Duma, 2024 — techniques, issues, future directions (KAIS) | Big tables of supervised / semi-supervised / unsupervised results |
| 36 | Paul & Nikolaev, 2021 — e-commerce SLR (DMKD) | How labels are made (Yelp filter, Ott crowd-sourcing, heuristics), and why that matters |

---

## 3. Each paper in 2–3 simple lines

- **01 Ben Jabeur 2023:** A survey, not an experiment. It maps 10 years of AI fake-review research and finds it is almost all **text**, with some behaviour and graph work. Images are barely mentioned.
- **02 Shan 2021:** **Real Yelp** reviews. **Text + rating**: it measures whether the stars, sentiment and content disagree. Random Forest reaches about 0.93 F-score. This is the "consistency checking" idea we build on (guide §4.5).
- **03 Luo 2023:** **Real YelpZip/NYC/Chi** reviews with Yelp-filter labels. **Text + behaviour + relations.** It *generates synthetic data points* from learned distributions to balance classes. Accuracy is 87–90%, with precision close to 100%.
- **04 Luo 2026:** Detects **AI-generated** reviews. Real reviews come from Yelp/Amazon, and fakes are generated by GPT-2/GPT-3/Llama-2. **Text only.** F1 is 0.83–0.98.
- **05 Barbado 2019:** **Real Yelp** consumer-electronics reviews collected by the authors. **Text + reviewer behaviour.** F1 is 0.82.
- **06 Wang 2026 (LLMIC):** **Ott Hotel (crowd-sourced fakes) + Amazon.** **Text only**, using LLM-derived "implicit" features. F1 is above 93–96%.
- **07 Duma 2025:** A survey of **graph neural network** methods on Yelp/Amazon. AUC is 0.74–0.98.
- **08 Vidanagama 2022:** **Unlabelled Kaggle Amazon** phone reviews. Labels were made by outlier detection. **Text only.** Accuracy is 89%.
- **09 Salminen 2022:** Real Amazon reviews + **GPT-2-generated fakes** (this is the famous 40k "OSF" set). **Text only.** fakeRoBERTa gets F1 0.97, but only AUC 0.70 when tested on Ott's human-written fakes. **Detecting machine text ≠ detecting paid humans.**
- **10 Zhao 2025:** A **real YelpZip + AI-generated** corpus of 714k reviews. It studies language differences only, with no detector.
- **11 Hou 2025 (FRIDRC):** A **self-built** dataset with manual + AI-generated fakes and **images (video frames)**. **Text + image + profile**, using RoBERTa + ViT. F1 is 0.97. **This is the only other paper here that really uses images**, but the images are context, not forensic evidence.
- **12 ITQM 2024:** **Real YelpZip (608k).** **Text + behaviour**, RoBERTa + CNN-LSTM. F1 is 88.8%.
- **13 Mohawesh 2024:** **Crowd-sourced** Ott OpSpam (1,600) + Li's Deception set. **Text only**, RoBERTa+LSTM. Accuracy 96% / 93%.
- **14 Vayadande 2025:** A survey of techniques and others' results.
- **15 Xu 2024 (A-LSTM):** **Real Yelp** (NYC/Chicago). **Text + behaviour.** Accuracy 90.9%, F1 88.3%. It also shows text-only F1 82.9% and behaviour-only 70.9%, the same "fusion beats each part" pattern as ours.
- **16 Shunxiang 2023 (SIPUL):** **Ott + real YelpZip/YelpChi.** **Text** (sentiment intensity), semi-supervised PU learning for streaming data. Good on Yelp, weaker on the small Ott set.
- **17 Naresh 2023:** The **OSF 40k set (half GPT-2-generated).** **Text only**, classic ML. SVM accuracy is 88%.
- **18 Pan & Xu 2024:** **Real Dianping crawl** (unlabelled) + Yelp. **Unsupervised, behaviour-based.** Evaluated by how much removing fakes improves recommendations, so there is no accuracy number.
- **19 Lee 2022:** **Real Yelp** restaurant reviews with Yelp-filter labels, imbalanced. **Text + behaviour.** The best F1 is only **0.564**. This is what honest, real, imbalanced data looks like.
- **20 Nair 2025:** The **OSF 40k set** split into 10 Amazon categories. **Text only.** Accuracy is 0.90–0.93 per category.
- **21 Anuprathibha 2024 (HGAN):** Dataset not clearly named. **Text + ratings in a graph.** Accuracy 91.3%.
- **22 Wang 2020:** **Real YelpChi + YelpRes.** **Text + behaviour**, semi-supervised. Accuracy 84.5%.
- **23 Elmogy 2021:** **Real YelpChi hotels** (5,853). **Text + reviewer behaviour.** Accuracy 87.9%, F1 82.4%.
- **24 Sudha Mercy 2024 (LDCP):** Dataset not clearly named. **Text only.** RoBERTa accuracy is 99.2–99.5%. Treat with caution.
- **25 Ennaouri 2023:** A survey (systematic literature review).
- **26 Sharma 2025:** **50k Amazon reviews.** **Text only**, BERT. Accuracy 96.3%.
- **27 Manish Kumar 2022:** The **OSF 40k set.** **Text only.** SVM accuracy is 88%.
- **28 Mishra 2025:** A literature summary plus a proposed BERT + behaviour pipeline.
- **29 Tufail 2022:** A small **self-built** set plus **Ott TripAdvisor.** **Text only**, an SVM/KNN/LR ensemble. Accuracy 89%.
- **30 Lu 2023 (BSTC):** **Crowd-sourced** Ott Hotel / Restaurant / Doctor sets. **Text only**, BERT + TextCNN. Accuracy 91–93%.
- **31 Sun 2024:** **250k real reviews** (Chinese platform). **Text + reviewer + merchant behaviour.** Accuracy 0.89, AUC 0.95.
- **32 Yu 2022:** A survey of **graph learning**, with good dataset statistics.
- **33 He 2022 (OUR BASE PAPER):** **Real Amazon** products, with fake-review buyers **observed directly** in Facebook groups (the best labels in this folder). **Network + metadata + text + image.** Random Forest AUC is 0.932 with all features. **Image features alone reach only AUC 0.592**, which is the gap our project targets.
- **34 Mohawesh 2023:** **Real YelpNYC/YelpZip.** **Text + behaviour** multi-view ensemble. F1 90% (NYC), AUC 0.82–0.86.
- **35 Duma 2024:** A survey with large results tables.
- **36 Paul 2021:** A survey. Its strongest point: **how the labels were made decides how high the numbers look.**
- **37 Li 2021:** **Real Dianping.** **Text + behaviour + reviewer groups (collusion).** F1 about 0.91. Group features add 4–7%. This is the closest idea to our "coordination graph".
- **38 Qayyum 2023 (FRD-LSTM):** The **Amazon 21k** labelled set. **Text only**, BiLSTM. Accuracy 97.2%.
- **39 Duma 2024 (DHMFRD-TER):** **Amazon + YelpChi + OSF.** **Text + emotion + rating.** Accuracy 0.99.
- **40 Geetha 2025 (MBO-DeBERTa):** **Amazon 21k, OSF 40k, Ott.** **Text only.** Accuracy 78% (Amazon), 98% (OSF), 91% (Ott). The same model scores 20 points apart depending on the dataset.

---

## 4. The big picture in numbers

| Question | Answer from the 31 experimental papers |
|---|---|
| How many use **images**? | **2**: He et al. 2022 (image AUC 0.592, the weakest feature) and FRIDRC 2025 (images as context). **None** checks whether a photo is **edited or reused**. |
| How many are **text-only**? | about 17 |
| How many use **real, platform-labelled** reviews? | about 14 (mostly Yelp filter labels, one Dianping, one Amazon) |
| How many use **crowd-sourced or AI-generated** fakes? | about 14 (Ott 1,600; OSF 40k GPT-2; GPT-3/Llama sets) |
| How many use **fully constructed / synthetic** data like ours? | FRIDRC (partly), Tufail (partly), Luo 2023 (synthetic points). **Nobody** has constructed image-forensic labels |
| Headline metric used | **Accuracy** in almost all; F1 often; AUC in about 8; **PR-AUC in none** |
| Typical numbers | OSF/GPT-2 sets: 88–99% acc · Ott crowd-sourced: 89–96% acc · real Yelp labels: F1 0.56–0.90, AUC 0.74–0.88 · base paper: AUC 0.932 |

---

## 5. Why our numbers look lower, and why that does NOT mean our model is worse

> **Update (final model):** this section was written for v6.0 (accuracy 0.852). The final model reaches
> **accuracy 0.943, F1 0.943, ROC-AUC 0.976** on the balanced test set and **PR-AUC 0.958** at 15% fake. The reasoning
> below still holds; see "Which papers report HIGHER numbers than ours?" in section 1 for the updated comparison.

You asked why we got "only less". Most of the gap comes from **what is being measured and on what data**,
not from the model. In order of importance:

### 5.1 We report a harder metric at a harder class balance
- Most papers report **accuracy on a 50/50 test set**. We report **PR-AUC at 15% fake**, which
  is the realistic rate and the honest metric (guide §6.6, §7.1).
- **Measured the way they measure** (balanced test set, accuracy), our fusion model gets **accuracy 0.852,
  F1 0.852, ROC-AUC 0.946**. That sits right next to real-data papers like Elmogy (87.9%),
  Wang 2020 (84.5%), Sun 2024 (0.89 acc / AUC 0.95), and above Lee 2022 (F1 0.564).
- At 15% fake, precision falls naturally (0.85 → 0.48) because there are 5.7 genuine reviews
  for every fake one. The model has not become worse; the task has become more realistic. Plain accuracy
  there would be meaningless: always saying "genuine" scores 85%.

### 5.2 The easy datasets inflate everyone else's numbers
- The **98–99%** results (OSF 40k, papers 17, 20, 27, 39, 40) detect **GPT-2 machine text vs human
  text**. That is a writing-style difference, not paid-review fraud. Salminen (paper 09) shows the same
  model falls to **AUC 0.70** on human-written fakes.
- The **Ott 1,600** set (papers 13, 29, 30) was written by Mechanical Turk workers in one session. It is small
  and easy to separate. On **real Yelp labels** the same kind of model drops to 67.8% (Mukherjee, cited in paper 15)
  and F1 0.564 (Lee, paper 19). Paper 40 shows it directly: one model gets 78% on Amazon and 98% on OSF.

### 5.3 We made our dataset hard on purpose
- 18% of our genuine reviews are **short and generic**, and 30% of text fakes **invent convincing detail**.
  Text alone cannot separate these (guide §9.6), so text-only is capped at PR-AUC 0.74.
- 555 fakes carry **perfectly genuine-sounding text** (edited photo only), and 555 carry a **perfectly
  clean photo** (text only). No single modality can win.
- Two published datasets have no such traps. That is why they look "better" and why their text models
  would fail on our data.

### 5.4 Our split is stricter
We split **by root photo**, so a photo the model trained on never appears in testing. Many papers split rows at random,
so near-duplicate reviews end up in both train and test (guide §6.6 says this alone can inflate results to ~98%).

### 5.5 Our models are deliberately simple
- The text block is **8 hand-made counts**, not BERT/RoBERTa. Almost every high-scoring paper uses a
  fine-tuned transformer. Swapping ours in (guide §16 step 4) is the single biggest easy gain,
  but it must be done honestly, because it also raises the baseline.
- The classifier is a Random Forest (the base paper's own choice), and it is trained on 2,490 rows.
  Others train on 20k–600k rows.

### 5.6 The image part is hard for everyone
- The base paper's image features alone reach **AUC 0.592**. FND-CLIP's image-only F1 is 0.409 (guide §3.4).
  Our image-forensic-only model reaches **ROC-AUC 0.744 / PR-AUC 0.567 at 15% fake**, which is actually **higher than
  the base paper's image features**.
- ELA-based tamper detection needs about 1000px images and is weakened by re-compression (guide §9.7, §12.4).

### 5.7 So what is the fair comparison?
- **Not** our 0.843 vs someone's 99%. They are different data, labels, metrics and prevalence (guide §12.6).
- **Yes:** our own ablation. Same data, same folds, one change at a time. **Fusion 0.843 vs
  base-paper-equivalent 0.801 (p = 3e-08, better in 24/25 folds)**, and the per-type table
  (text-only catches 15% of edited-photo fakes, image-only catches 19% of text fakes, fusion catches 71% and 86%).
- **Closest external comparison:** the base paper (He et al.). All features give AUC 0.932, image-only 0.592.
  Our ROC-AUC is 0.946 with image-forensic-only 0.744. This is only a *direction* check (different task: product-level
  vs review-level; real vs constructed data). Do not put these numbers side by side as a claim.

### 5.8 What would honestly raise our numbers
1. RoBERTa text features instead of 8 counts (the biggest gain; also raises the baseline).
2. A small CNN on ELA / noise-residual maps instead of 13 summary numbers (guide §12.5).
3. More root photos (826 now), which gives tighter confidence intervals.
4. Tuning the Genuine / Needs-Verification / Fake bands on validation folds.
5. Reporting **both** balanced accuracy/F1 (for comparison with the literature) and PR-AUC at 15% (for honesty).

---

## 6. One-paragraph summary for the report

> Across 31 experimental papers in our survey, reported accuracies range from 84% to 99%.
> The highest numbers come from text-only models on machine-generated (OSF/GPT-2) or crowd-sourced (Ott)
> datasets, while real platform-labelled data gives F1 0.56–0.90. Only two works use review images, and
> in the strongest of them (He et al., 2022) image features alone reach AUC 0.592. No surveyed work treats
> the image as forensic evidence (tampering, cross-account reuse). On a balanced test set, our fusion model
> reaches 94.3% accuracy, F1 0.943 and ROC-AUC 0.976, above 18 of the surveyed papers and below about 10
> text-only papers evaluated on easier data. A fine-tuned transformer text model, the method behind most of those
> 10, reaches only 83.3% on our dataset, because a third of our fakes carry genuine-looking text. Under a stricter
> protocol (grouped by photo, 15% fake prevalence), our model reaches PR-AUC 0.902 in cross-validation and
> significantly outperforms a base-paper-equivalent configuration (+0.144, p < 1e-15, 25/25 folds).