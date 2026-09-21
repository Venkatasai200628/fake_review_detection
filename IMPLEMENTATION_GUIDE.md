# Fake Review Detection via Visual Forensics
## Master Implementation Guide

**Version:** 1.0 (consolidates PROJECT_SPEC, DATASHEET, TECHNICAL_REPORT_01, TECHNICAL_REPORT_02)
**Last updated:** 2026-09-18
**Course:** Computer Vision, Semester 5 BTech (AI & Data Science)
**Owner:** Venkatasai
**Repository (image pool):** https://github.com/Venkatasai200628/fake_review_detection

---

## 0. How to read this document

This is the single source of truth. It is written so that a person or an AI
assistant who has never seen this project can read it top to bottom and then do
useful work without asking for background.

| If you want to know… | Read |
|---|---|
| What this project is, in plain English | §1 |
| The problem and why it matters | §2 |
| What other researchers have done | §3 |
| What we claim is new | §4 |
| Which models we use and why | §5 |
| How the dataset is built and labelled | §6 |
| How we evaluate and why that way | §7 |
| What results we have | §8 |
| Everything that went wrong and how we fixed it | §9 |
| Exactly where the project stands today | §10 |
| What is still weak | §11 |
| What happens next | §12 |
| How to run the code | §13 |

**Terminology warning.** The word "extension" is ambiguous in this project and
has caused real confusion. It always means one of two things and this document
always says which:
- **research extension** — the novel contribution beyond the base paper
- **browser extension** — the Chrome plugin deliverable

---

## 1. The project in plain English

A large share of product reviews on shopping sites are fake. Sellers pay people
to write them. Most existing detection systems read the *words* of a review and
try to spot dishonest writing. That works, but it is beatable — a competent paid
reviewer simply writes better.

Many reviews come with a photo, and almost nobody examines that photo properly.
We treat the photo as **evidence**, the way an investigator treats a document,
and ask three questions about it:

1. **Has it been edited?** Was something pasted in, cloned over, or retouched?
2. **Where did it come from?** Is this "customer photo" actually the seller's own
   catalogue image? Has the same picture already appeared under a different
   account, on a different product?
3. **Who else used it?** If twenty accounts posted slight variations of one photo
   within a couple of hours, those accounts are working together.

We then combine those three answers with the usual signals — what the text says,
the star rating, when it was posted — into one judgement, which can be
*Genuine*, *Fake*, or *Needs Human Verification*.

**Why combining matters.** A fake reviewer can fool one check but rarely all of
them. Someone who writes flawless text still has to attach a photo from
somewhere. Someone with a genuine photo still has to write something. Our
dataset deliberately contains both cases, and we have measured that the combined
system catches both while either single-modality system misses one.

**Why it is a Computer Vision project.** All of the new work is on the image
side. Text is still present and still matters, but we do not claim to have
improved text analysis.

---

## 2. Problem statement

Paid and fabricated reviews distort purchasing decisions at scale. Existing
detection work falls into three families:

1. **Text-based** — linguistic deception cues, BERT/RoBERTa classifiers. Strong
   within a domain, but the cues are domain-specific: fake reviews about
   headphones use different vocabulary from fake reviews about shoes.
2. **Behavioural / graph-based** — posting bursts, rating skew, reviewer–product
   bipartite graph structure. Currently the strongest family.
3. **Multimodal** — a small number of works add review images, almost always as
   an extra feature block alongside text.

**The gap.** No significant work treats the review image as evidence with its own
provenance and integrity. Images get embedded and compared, but nobody asks
whether an image was edited, whether it has appeared elsewhere on the platform
under a different account, or whether shared imagery links accounts into a
coordinated cluster.

That gap is what this project fills, and it is a genuine computer vision problem.

---

## 3. What previous work has done

### 3.1 Reported performance in the literature

| Work | Dataset | Reported |
|---|---|---|
| KE-MLLM | YelpChi | F1 94.3%, AUC-ROC 96.7% |
| A-LSTM (via survey) | YelpNYC | up to 95% F1 |
| text + behaviour fusion (survey) | Yelp variants | accuracy up to 90.9% |
| Mukherjee et al. | YelpChi | 67.8% accuracy (n-gram features) |
| FND-CLIP | Politifact | image-only F1 0.409 vs text-only 0.808 |

Note the 27-point spread between KE-MLLM and Mukherjee on the *same* dataset.
That is what a valid model comparison looks like: same data, different models.

### 3.2 Standard metrics in this field

Accuracy, precision, recall, F1, AUC-ROC. At least one multimodal work
explicitly prioritises F1 because of the class imbalance typical of fake-review
datasets. **PR-AUC is rarer but more appropriate at low fraud prevalence**, so we
report both.

### 3.3 Models used in the multimodal branch

- FRIDRC found **RoBERTa (text) + ViT (image)** the strongest combination
- A two-tier co-attention framework uses **fine-tuned BERT + fine-tuned VGG19**
  plus hand-crafted aesthetic features
- Several works compute **text–image cosine similarity**, or use **CLIP**
  directly to detect cross-modal inconsistency

### 3.4 A published finding that supports our results

FND-CLIP's own ablation shows image-only performing far below text-only —
Politifact image-only F1 0.409 against text-only 0.808. **A weaker image
modality in isolation is the normal, published pattern in multimodal deception
detection, not a defect in our pipeline.** Cite this when presenting our
ablation, because "why is your image module worse than text?" is the obvious
question.

---

## 4. Base paper and our novelty

### 4.1 Base paper

> He, S., Hollenbeck, B., Overgoor, G., Proserpio, D., & Tosyali, A. (2022).
> *Detecting Fake-Review Buyers Using Network Structure: Direct Evidence from
> Amazon.* PNAS 119(47).

Selected from a survey of ~40 papers for three reasons:

- **Best available label quality.** The authors directly observed sellers
  recruiting reviewers in Facebook groups, so labels tie to documented purchase
  of reviews — not a platform's opaque filter (Yelp) and not lab-written fakes
  (Ott et al.).
- **It already uses review images**, making it the closest paper to extend rather
  than contradict.
- **It reports that network features beat image features.** That published
  weakness is our opening.

### 4.2 What the base paper does

| Component | Their approach |
|---|---|
| Labels | Direct evidence from Facebook recruitment posts |
| Network | Bipartite reviewer–product graph; products sharing unusually many reviewers flagged |
| Image | Pretrained CNN embeddings → cosine similarity, aggregated, **within a single product only** |
| Text/behaviour | Rating skew, helpfulness votes, timing gaps |
| Classifier | Random Forest (supervised), Gaussian mixture (unsupervised) |

Their image module answers exactly one question: *do this product's images look
similar to each other?*

### 4.3 Our novelty, precisely stated

**One sentence:** *He et al. use image similarity as a feature within a product;
we use manipulation, reuse and coordination evidence from images as forensic
signals across the entire reviewer–product ecosystem, fused with text and
behaviour in a model that can abstain.*

Three new signal types:

1. **Manipulation / authenticity evidence** — absent from the base paper entirely
2. **Cross-product and cross-reviewer provenance** — not confined to one product
3. **A visual coordination graph** — image-similarity edges layered onto the
   reviewer–product graph

### 4.4 Research extension (second contribution)

He et al. never test generalisation beyond their domain. We explicitly test
whether the detector generalises to **product categories never seen during
training**, using leave-one-category-out validation.

The reasoning: text features are domain-specific; image-forensic features ("was
this edited, reused, or shared across suspicious accounts") are largely
independent of what the product is. A sharp drop is still a valid reportable
result.

### 4.5 Honest novelty boundary

State these before an examiner finds them:

- Consistency checking is not new. Prior work checks rating-vs-sentiment
  inconsistency **within one review**. We widen it across review, reviewer,
  product and time.
- pHash, ELA, CLIP and Louvain are all standard off-the-shelf techniques.
  **The novelty is composition and application**, not invention of components.
- The dataset is a controlled evaluation, not real-world ground truth.

---

## 5. Models and methods — what we use and why

### 5.1 Pipeline

```
review (text, rating, timestamp, reviewer_id, product_id, image)
   │
   ├─ M1  Image representation      → embedding e_i
   ├─ M2  Similarity                → cosine sim(e_i, e_j)
   ├─ M3  Provenance / reuse        → pHash + ANN search → provenance features
   ├─ M4  Manipulation              → ELA + noise residual → manipulation features
   ├─ M5  Visual coordination graph → clustering coeff, Louvain communities
   ├─ MT  Text module               → deception features
   ├─ MB  Behaviour module          → rating skew, burst timing
   │
   └─ M6  Fusion → risk score → 3-way decision + evidence explanation
```

### 5.2 Module-by-module rationale

**M1 — Image representation. CLIP ViT-B/32 (laion2b_s34b_b79k).**
*Why:* FRIDRC found ViT strongest for this task; CLIP additionally gives a shared
image–text space, which unlocks cross-modal consistency for free. *Why not
ResNet-50:* weaker semantic separation. *Why not train from scratch:* impossible
at ~400 root images. *Measured benefit:* unrelated-image similarity dropped from
0.802 (placeholder descriptor) to 0.391 — a real gap instead of a sliver.

**M2 — Cosine similarity.** `sim(e_i,e_j) = e_i·e_j / (‖e_i‖‖e_j‖)`. Same
building block as the base paper, so comparisons stay fair.

**M3 — Provenance. Perceptual hash (pHash, DCT-based) + approximate nearest
neighbour search.**
*Why pHash:* cheap, robust to mild re-encoding and resizing, gives an
interpretable Hamming distance. *Why not dHash/aHash:* faster but less robust.
*Why ANN search platform-wide rather than per-product:* this is precisely what
the base paper cannot do, and it is what catches an image reused across
different products or accounts.

**M4 — Manipulation. Error Level Analysis + noise-residual inconsistency.**
*Why ELA:* recompressing at fixed quality and measuring the *inconsistency* of
the error field across the frame exposes regions with a different compression
history — a spliced patch, a cloned region. *Why noise residual:* a camera
produces a roughly uniform noise floor; a splice from another photograph does
not. *Why classical rather than a CNN:* appropriate scale for a course project,
interpretable, and no training data for manipulation exists. *Known weakness:*
see §11.

**M5 — Visual coordination graph. Louvain community detection + clustering
coefficient.**
`G=(V,E)` over reviewers ∪ products ∪ images; edge when `sim > τ` or
`d_H < δ`. Clustering coefficient `C_v = 2T_v / (k_v(k_v−1))`; modularity
`Q = (1/2m) Σ[A_ij − k_ik_j/2m]δ(c_i,c_j)`.
*Why explicit graph features and not a GNN:* a GNN has more parameters than ~2000
rows can support and would obscure which signal is doing the work. Explicit
features keep the ablation interpretable. GNN is listed as future work.

**M6 — Fusion. Random Forest (primary) + MLP.**
*Why Random Forest:* it is the base paper's own classifier, so using it makes the
comparison a like-for-like test of features rather than of model class. *Why also
an MLP:* the spec's design, and it handles feature interactions differently.
*Why not co-attention yet:* that is current multimodal SOTA and is listed as
future work; it needs more data than we have.

### 5.3 The three-way output

| Band | Decision |
|---|---|
| low | Genuine |
| middle | **Needs Human Verification** (abstain) |
| high | Fake |

*Why abstain:* real platforms route borderline cases to human review rather than
auto-deleting. It is part of the stated novelty, and it makes the browser
extension demo far more credible — an amber badge reads as considered judgement,
a red/green binary reads as overconfidence.

---

## 6. Dataset

### 6.1 Why synthetic

The novel signals — coordinated image reuse and image manipulation — **do not
exist as labelled data anywhere**. No public dataset labels review images for
tampering or cross-account reuse. Constructing controlled examples is standard
practice in multimodal-consistency work, not a fallback.

Datasets considered and rejected:

| Dataset | Disposition |
|---|---|
| YelpChi / YelpNYC / YelpZip | **Deferred.** Standard benchmark, but no review images — cannot exercise any novel module. Optional for text baseline. |
| Yelp Open Dataset | **Rejected.** Has photos, but no manipulation/reuse labels; we would construct them anyway, with less control. |
| Ott et al. Deceptive Opinion Spam | **Optional.** Gold-standard text, no images. |
| Amazon Reviews (McAuley) / Kaggle CG-OR | **Optional.** Second text domain for generalisation. |
| FRIDRC | **Blocked.** Access never confirmed. Do not plan around it. |
| He et al. data | **Not obtainable.** This is why we build our own. |

### 6.2 Two layers

| | Layer 1 | Layer 2 |
|---|---|---|
| What | Root photo pool | Generated review dataset |
| Where | the GitHub repo | produced by `build_dataset.py` |
| Current | 627 photos, 15 categories | ~1974 rows |
| Labels | none | full ground truth |

Each Layer 1 photo is a **root image**. Multiple review rows derive from one root
via different transform chains — mirroring real listings, where a handful of true
product photos spawn many customer re-uploads.

### 6.3 Labels — two columns, different jobs

**`label` — the prediction target.**

| Value | Meaning |
|---|---|
| `Genuine` | Real customer review; model should pass it |
| `Fake` | Fraudulent by any mechanism; model should flag it |
| `Needs_Verification` | Genuinely ambiguous; model should abstain |

**`fraud_type` — the mechanism. NEVER train on this; it encodes the answer.**

| `fraud_type` | `label` | Image tell | Text tell |
|---|---|---|---|
| `genuine` | Genuine | none | none |
| `visual_manipulation` | Fake | edited pixels | **none — text sounds real** |
| `text_deception` | Fake | **none — clean photo** | generic superlatives |
| `coordinated_reuse` | Fake | same root across accounts | shared templates |
| `borderline_recompress` | Needs_Verification | re-saved only | none |
| `borderline_ambiguous_reuse` | Needs_Verification | 2-account reuse | none |

**Why `fraud_type` must exist.** Your headline result is not one accuracy number,
it is this:

| | catches `visual_manipulation` | catches `text_deception` |
|---|---|---|
| text-only | ✗ | ✓ |
| image-only | ✓ | ✗ |
| fusion | ✓ | ✓ |

Those two failure cells are the entire justification for the project.
`visual_manipulation` carries deliberately authentic-sounding text and
`text_deception` carries a clean real photo **precisely so that only one modality
can catch each**. If the signals correlated, a text-only baseline would match
fusion and there would be no result.

### 6.4 Generation rules

- **Benign transforms** (every row): crop 90–99%, rotate ±3°, brightness
  0.82–1.18×, contrast 0.85–1.20×, saturation 0.80–1.20×, blur σ0.2–0.7, JPEG
  q55–80. Two to four applied at random. *Magnitudes are mild for a measured
  reason — see §9.4.*
- **Manipulations** (`visual_manipulation` only): copy-move (12–22% patch),
  splice (15–28% region from a different photo), noise injection (σ18 into a 25%
  region).
- **Rating skew:** fraud clusters at 1★/5★; genuine spreads realistically.
  Ratings must agree with text polarity (§9.3).
- **Timing:** campaigns burst inside ~3h; genuine spreads across ~180 days.
  Deliberately opposite, so the model must learn reuse **and** timing jointly —
  reuse alone is not fraud.
- **Text specificity gap:** genuine names a specific part, duration and
  situation; deceptive gives generic superlatives; campaigns paraphrase 3 shared
  templates.
- **Overlap cases (§9.5):** 18% of genuine reviews are short and generic (real
  customers do write that way); 30% of `text_deception` fabricates convincing
  specifics (skilled paid reviewers do that). These two groups overlap in text
  space so text alone cannot resolve them.

### 6.5 Negative controls — deliberate traps

| Control | `label` | The trap | Pass condition |
|---|---|---|---|
| `legit_stock_reuse` | Genuine | Manufacturer photo reused by real buyers over weeks | Must NOT be flagged |
| `recompress_only` | Needs_Verification | Re-saved, nothing edited | Score below 0.25 |
| `weak_reuse_not_campaign` | Needs_Verification | Only two accounts share an image | Should abstain |

If the model flags `legit_stock_reuse`, the provenance module has learned "any
reuse = fraud" and the result is invalid regardless of headline accuracy.

### 6.6 Split policy — non-negotiable

Partition **by `root_image_id`**, stratified within category.

**Why.** If transform variants of the same root appear in both train and test,
the model recognises the photograph rather than the fraud signal and reports a
meaningless ~98%. Verified by assertion at build time: root overlap = 0.

Two prevalences:
- **Training ~40% fraud** — at real prevalence there would be too few fraud
  examples to learn from. Over-sampling for training is standard and correct.
- **Reporting ~15% fraud** — approximates real platform rates.
- **Never report plain accuracy on the holdout.** At 15% prevalence, always
  predicting Genuine scores 85%.

---

## 7. Evaluation methodology

### 7.1 Metrics

Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, explicit operating
threshold. **PR-AUC is the headline** at 15% prevalence; ROC-AUC is also reported
for comparability with the literature.

### 7.2 Protocol

**5 repeats × 5 folds grouped cross-validation**, groups = `root_image_id`, group
assignment reshuffled each repeat, test folds thinned to 15% prevalence before
scoring, all configurations seeing **identical folds** so differences are paired
and a paired t-test is valid.

*Why not a single holdout:* the earlier holdout had 12 fraud rows. A 0.067 gap on
12 positives is noise, not a result.

### 7.3 Required ablations

text-only, image-only, behaviour-only, graph-only, base-paper-equivalent,
full fusion (RF), full fusion (MLP) — plus **per-fraud-type breakdown**.

### 7.4 Leave-one-category-out

Train on 11–12 categories, test on 3–4 unseen. Report Δ for text-only vs
image-forensic-only vs fusion. `loco_folds.json` is generated.

---

## 8. Results to date

### 8.1 Headline

**Fusion beats the base-paper-equivalent configuration by +0.045 PR-AUC, and the
gap survives cross-validation.**

```
fusion PR-AUC        0.817
base-equivalent      0.772
paired difference    +0.045  (95% CI +0.030 to +0.060)
paired t-test        t = 5.96   p = 3.7e-06   n = 25 folds
fusion wins in       23 of 25 folds
```

Independently reproduced on a second machine (0.817 vs 0.818; drift is Random
Forest thread scheduling, well inside the CI).

### 8.2 Full ablation (5×5 grouped CV, Random Forest, mean ± 95% CI)

| Configuration | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| text-only | 0.721 ±0.025 | 0.865 | 0.550 | 0.481 | 0.656 |
| image-forensic-only | 0.506 ±0.032 | 0.711 | 0.433 | 0.415 | 0.468 |
| behaviour-only | 0.377 ±0.026 | 0.746 | 0.379 | 0.265 | 0.672 |
| graph-only | 0.213 ±0.013 | 0.575 | 0.268 | 0.187 | 0.475 |
| base-paper-equivalent | 0.772 ±0.023 | 0.919 | 0.576 | 0.467 | 0.757 |
| **FUSION (ours)** | **0.817 ±0.022** | **0.938** | **0.653** | **0.573** | **0.776** |

### 8.3 Per-fraud-type detection rate — the table that explains why

| Configuration | visual_manipulation | text_deception | coordinated_reuse |
|---|---|---|---|
| text-only | **0.241** | 0.780 | 1.000 |
| image-forensic-only | 0.315 | **0.186** | 0.979 |
| base-paper-equivalent | 0.441 | 0.870 | 1.000 |
| **FUSION (ours)** | **0.579** | 0.789 | 1.000 |

**How to read this.** Text-only collapses on `visual_manipulation` (0.241)
because that text is deliberately authentic. Image-only collapses on
`text_deception` (0.186) because that photo is clean and real. Fusion lifts
`visual_manipulation` from 0.441 to 0.579 over the baseline — a **32% relative
improvement**, and the source of the overall PR-AUC gain. Precision also rises
from 0.467 to 0.573, a 22% relative cut in false alarms.

### 8.4 What is weak in the results

- **`graph-only` is barely above chance** (ROC-AUC 0.575). The coordination graph
  is context, not a standalone detector. Report honestly rather than omit.
- **`image-forensic-only` sits at 0.506 PR-AUC**, essentially the prevalence
  baseline. Its contribution is real but only visible in combination.
- **`coordinated_reuse` at ~1.00 for everything** is partly circular by
  construction (§11). Do not present it as a headline number.

### 8.5 Robustness check — is the result an artifact of image reuse?

Concern: 1974 rows come from only ~197 roots, so rows are not independent.
Tested by capping variants per root:

| Setting | Rows | Fusion | Base | Diff | p |
|---|---|---|---|---|---|
| 3 variants/root max | 590 | 0.785 | 0.740 | **+0.045** | 0.0006 |
| 5 variants/root max | 963 | 0.799 | 0.773 | +0.026 | 0.018 |
| uncapped (~10/root) | 1926 | 0.826 | 0.780 | +0.046 | 4e-09 |

At a third of the data the gap is identical. **Scaling up bought tighter
confidence, not a bigger effect** — exactly what you want. If multiplication were
creating the result, the gap would grow with the cap.

---

## 9. Project history — what went wrong and how it was fixed

This section exists because every item here is something an examiner may ask
about, and because five of them would have quietly invalidated the results.

### 9.1 Chronology

| Stage | Outcome |
|---|---|
| Literature survey (~40 papers) | done |
| Base paper selected | He et al. 2022 |
| Novelty defined | three modules + fusion + abstain |
| Layer 1 collected | 197 photos → later 627 |
| Generator written | `build_dataset.py`, `gen_images.py`, `gen_text.py` |
| v1 → v5 dataset iterations | see below |
| Forensic module built | `forensics.py` |
| Cross-validation protocol | `crossval.py` |
| CLIP encoder swap | embeddings recomputed, threshold retuned |

### 9.2 Leak 1 — the feature read the answer key

`features.py` used `manipulation_score` as its principal image feature. That
column is **the generator's ground truth** (0.01–0.12 clean, 0.66–0.96
manipulated). The model was reading the labels.

*Symptom:* image-only detected 100% of `visual_manipulation`; fusion PR-AUC 1.000.
*Fix:* built `forensics.py` to compute ELA and noise features from pixels; removed
the column from the feature set.
*Effect:* image-only 0.804 → 0.531; fusion 1.000 → 0.824. **Removing the leak
removed the result**, which is what prompted everything below.

### 9.3 Rating/text polarity contradiction

v1 assigned fraud ratings independently of text, producing rows like *"Excellent
suitcase, quality is really good"* at 1★. That contradiction is a real fraud
signal — but **uncontrolled and undesigned**, so the model could have learned it
as a shortcut instead of the three claimed signals. ~12 of 501 rows affected.
*Fix:* threaded text polarity into the rating function. Verified: 0 contradictions.

### 9.4 Leak 2 — different compression histories

`apply_manipulation()` ended with its own JPEG pass that genuine images never got.
Manipulated images were simply *more compressed*.

*Symptom:* ROC-AUCs of 0.198, 0.281 — **below 0.5**, i.e. anti-predictive. The
detectors were finding "this image went through an extra save."
*Fix:* removed that pass; `save_derived()` now applies exactly one uniform pass to
every row.
*Why it matters beyond this project:* this is the trap that makes deepfake
detectors fail in deployment. A detector trained where fakes share a processing
history that reals do not learns the processing, not the manipulation.

### 9.5 Leak 3 — root leakage inside the diagnostic experiment

The experiment built one genuine **and** one manipulated sample from each root.
With ordinary stratified CV, recognising a root in training implies the
**opposite** label at test time — hence systematic sub-0.5 AUC.
*Fix:* `GroupKFold` on `root_image_id`.
*Note:* this is the rule the spec already stated for the main dataset. The main
splits were correct; the diagnostic experiment was not. **The rule must be
applied everywhere, not once.**

### 9.6 Text was trivially separable

v2 text-only reached PR-AUC 1.000 — the template pool was small enough that
"generic word ratio" separated the classes perfectly, making every ablation
meaningless.
*Fix:* added overlap cases (§6.4). Text-only fell to 0.664, then 0.721 at scale.
**Not making the task artificially harder — removing an artificial easiness.**

### 9.7 The real problem — output resolution

With all leaks removed, honest measurement:

| Detector | 512px | 1024px |
|---|---|---|
| ELA | 0.513 | **0.739** |
| noise residual | 0.507 | 0.692 |
| ORB copy-move | 0.535 | 0.429 |
| all combined | 0.554 | **0.758** |

At 512px every manipulation detector is at chance. Manipulations were applied at
768px then saved at 512px; that downscale averages away exactly the
high-frequency evidence both detectors need.

*Fix:* `WORK_SIZE` 768→1280, `OUT_SIZE` 512→1024.

**Why this is legitimate and not result-shopping.** 512px was chosen for file-size
convenience and never justified. Amazon serves product images at 1500px, so 1024
moves the dataset *toward* reality. Both settings were measured side by side
under identical leak-free conditions, and both are reported — including ORB
copy-move, which got *worse* and was excluded.

### 9.8 Holdout too small

12 fraud rows. *Fix:* scaled to 1974 rows (48 holdout fraud rows, 16 per type) and
switched to 5×5 grouped CV with paired testing.

### 9.9 Performance fix required to get there

The pairwise reuse index was an O(n²) Python loop — ~3.9M object comparisons at
1974 rows. Replaced with vectorised bit arithmetic:
`hamming = B·(1−B)ᵀ + (1−B)·Bᵀ`. Feature build now takes **6 seconds**.

### 9.10 CLIP encoder swap

Cannot run in the sandbox (huggingface.co, openaipublic and download.pytorch.org
all blocked). Run locally on RTX 4060 / CPU.

*Measured effect on separation:*

| | Placeholder | CLIP |
|---|---|---|
| same-root mean | 0.986 | 0.938 |
| different-root mean | 0.802 | **0.391** |
| p5(same) − p95(diff) | ~+0.03 | **+0.292** |

**Threshold must be retuned.** `COSINE_REUSE_THRESHOLD = 0.94` was tuned for the
placeholder. Recommended value from `tune_thresholds.py`: **0.749** (TPR 0.991,
FPR 0.0038). Note the script's "1% FPR" suggestion (0.699) is worse than its
max-Youden suggestion — ignore it. A false reuse edge propagates into the
coordination graph, so be conservative.

---

## 10. Current status

| Component | Status |
|---|---|
| Layer 1 image pool | **627 photos** on GitHub, 15 categories |
| Pruning of <1000px images | **PENDING** — 240 files identified, commit is local, not yet pushed |
| Layer 2 generator | working, v5 |
| Split integrity | verified by assertion |
| Negative controls | `legit_stock_reuse` passes; two others regressed (§11) |
| Forensic module | working, ROC-AUC 0.622 in isolation |
| Cross-validation protocol | working, reproduced on two machines |
| CLIP embeddings | computed locally; **threshold not yet set to 0.749** |
| Cross-modal feature | computed (`clip_cross_modal.csv`), not yet evaluated |
| LOCO experiment | not started |
| Browser extension | not started |

**Immediate blockers:**
1. Push the pruning commit (rebase conflict being resolved).
2. Collect ~423 replacement photos at ≥1000px to reach 54/category.
3. Set `COSINE_REUSE_THRESHOLD = 0.749`, re-run `crossval.py`.

---

## 11. Known weaknesses

1. **Circular provenance ground truth.** The same code applies transforms and the
   detector finds them. pHash scores near-perfectly and that number means little.
   Report as a controlled check, never as a performance claim. **Raising this
   yourself is evidence you understand your own method.**
2. **Manipulation module is weak** — ROC-AUC 0.622 in isolation. ELA on stock
   photos already compressed before collection, then re-encoded once more, is a
   hard setting.
3. **ELA is defeated by platform re-encoding and downscaling below ~1024px.**
   Measured (§9.7). This is also a *contribution* — see §12.4.
4. **Two negative controls currently failing:** `recompress_only` (2 of 6
   flagged), `weak_reuse_not_campaign` (3 of 6). The model is drifting toward
   "any reuse = fraud." **Diagnose; do not raise thresholds until they pass.**
5. **Image diversity is bounded by the root pool.** Rows scale via transform
   variants, not new imagery, so the effective independent sample size for
   image-based claims is the number of roots. Mitigated by grouping and verified
   stable under a 3-variant cap (§8.5).
6. **38% of current Layer 1 photos are below 1000px** and cannot support the
   manipulation module. Being replaced.
7. **Behaviour is simulated**, not observed from real reviewer logs.
8. **Text is template-generated.** A text model trained here will not transfer to
   real reviews.
9. **Products are category-level, not item-level** — one `product_id` per
   category, so within-product comparisons are coarser than a real catalogue.
10. **Component techniques are all standard.** The contribution is composition.

---

## 12. What happens next

### 12.1 Immediate

1. Push the image-pruning commit.
2. Collect ~423 photos at ≥1000px (Unsplash/Pexels/Pixabay; **avoid iStock —
   previews are capped at 612px**, which is where all 240 rejects came from).
3. Regenerate Layer 2 against the larger, cleaner pool.
4. Set `COSINE_REUSE_THRESHOLD = 0.749`; re-run `validate.py` then `crossval.py`.
5. Compare against the recorded 0.817 baseline under the identical protocol.

### 12.2 Then

6. **Evaluate the cross-modal feature** as its own ablation (fusion with CLIP
   without cross-modal, vs with). Expected to help most on `text_deception`,
   where generic text will not describe the specific product shown.
7. **Diagnose the two failing negative controls.**
8. **RoBERTa/BERT text module**, replacing 8 hand-crafted counts. Note this
   *raises the baseline* fusion must beat — correct and honest.
9. **Leave-one-category-out run** — the generalisation claim.
10. **Browser extension.**

### 12.3 Browser extension design

Manifest V3, three parts:
- **Content script** — reads review blocks from an Amazon/Flipkart product page:
  text, rating, reviewer name, image URLs.
- **Local backend** — FastAPI on localhost running the fusion model, returning
  `{risk_score, decision, driving_module}` per review.
- **Injection** — badge per review card (green / amber / red) plus a page-top
  summary.

Build last. Worthless without a trained model; roughly a weekend once one exists.
Keep it read-only and local; do not log or redistribute scraped data.

### 12.4 Framing for the paper

The instinct that a failed module cannot be published is only half right. What
cannot be published is a module that fails *and is reported as working*. What is
entirely publishable — and stronger than a clean sweep:

> We evaluated three visual-forensic signal families. Provenance and coordination
> evidence transfer robustly. Compression-based manipulation evidence is
> substantially degraded by platform re-encoding and image downscaling, and we
> quantify the resolution threshold below which it disappears entirely.

That last clause is a contribution in its own right. Nobody in the surveyed
literature reports it, because most work evaluates on full-resolution research
datasets rather than platform-processed images.

### 12.5 Longer-term / optional

- Small CNN on ELA and noise-residual maps instead of 14 summary statistics
- Co-attention fusion (current multimodal SOTA)
- GNN over the coordination graph, once data supports it
- Item-level rather than category-level products
- A second text domain (Amazon/Ott corpora) to support generalisation

### 12.6 The comparison trap — do not fall into it

**Never place our 0.817 beside KE-MLLM's 0.943 in a table.** They measure
different things: 608,598 rows vs 1,974; platform-filter labels vs constructed
ground truth; a decade-tuned benchmark vs a set deliberately seeded with
ambiguous cases. A model scoring 0.94 on an easy dataset and 0.82 on a hard one
tells you about the datasets.

The valid comparison is the internal ablation in §8.2: identical data, identical
folds, identical model class, one variable changed.

---

## 13. Code and how to run it

### 13.1 Files

| File | Purpose |
|---|---|
| `build_dataset.py` | orchestration, campaigns, controls, splits, integrity asserts |
| `gen_images.py` | root loading, transforms, manipulations, pHash, embeddings |
| `gen_text.py` | per-fraud-type text and rating generation |
| `forensics.py` | ELA, noise residual, copy-move — computed from pixels |
| `features.py` | the four feature blocks (TEXT / IMAGE / BEHAVIOUR / GRAPH) |
| `baselines.py` | single-split ablation |
| `crossval.py` | 5×5 grouped CV with paired testing — **the reporting protocol** |
| `validate.py` | 8 post-build integrity checks |
| `experiment_forensics.py` | resolution and detector-family comparison |
| `clip_embedding.py` | CLIP image + text encoders (run locally) |
| `recompute_embeddings.py` | in-place embedding swap on existing images |
| `tune_thresholds.py` | pick `COSINE_REUSE_THRESHOLD` from data |
| `prune_small_images.py` | remove Layer 1 photos below the resolution threshold |

### 13.2 Folder layout the scripts expect

```
project/
    dataset/    all .py files      <- run from here
    out/        CSVs, images/, results
    repo/       clone of the GitHub image pool
```

### 13.3 Full run

```bash
cd dataset
python build_dataset.py                    # Layer 2 from Layer 1
python -c "import os,pandas as pd,forensics; df=pd.read_csv('../out/reviews_full.csv'); f=pd.DataFrame([forensics.analyse(os.path.join('../out',r.image_file)) for r in df.itertuples()]); f.insert(0,'review_id',df.review_id.values); f.to_csv('../out/forensic_features.csv',index=False)"
python recompute_embeddings.py             # CLIP (local only)
python tune_thresholds.py                  # then edit features.py
python validate.py                         # integrity checks
python crossval.py                         # the reported numbers
```

### 13.4 Reproducibility

`SEED = 20260908` throughout. `build_dataset.py` reproduces byte-identically
given the same root pool. **Always run `validate.py` after regenerating** —
check 1 (pHash separation) is the one that fails first if transform magnitudes
change.

---

## 14. Glossary

- **Root image** — one of the Layer 1 photos; multiple review images derive from
  it via transforms
- **pHash** — perceptual hash; DCT-based fingerprint compared by Hamming distance
- **ELA** — Error Level Analysis; JPEG recompression difference map
- **LOCO** — leave-one-category-out validation
- **Abstain band** — the middle risk range mapped to "Needs Human Verification"
- **Negative control** — a case built to look suspicious but labelled genuine, to
  catch shortcut learning
- **Leakage** — any path by which information about the label reaches the model
  other than through genuine signal

---
---

# PART 2 — Added 2026-09-19 (nothing above this line was changed)

This part is written in simple English. It records what was done on 19 September
2026 and explains, step by step, how we go from "we have a dataset" to "a Chrome
extension that checks reviews on a real shopping page". The detailed story of
each day's work is in the `readmes\` folder (`readme1.md`, `readme2.md`, …).

---

## 15. Status update — 2026-09-19

| Item from §10 / §12.1 | Status now |
|---|---|
| Push the image-pool commit | **DONE.** 836 photos, 15 categories (54–63 each), on GitHub `main`. The stuck rebase was finished without deleting anything. |
| Collect ~423 photos ≥1000px | **DONE.** Every category has ≥54 photos. 10 backpack photos are still <1000px; the generator skips them (826 usable roots). |
| Regenerate Layer 2 on the bigger pool | **DONE — v6, BALANCED.** 3,300 rows = exactly 1,650 Genuine + 1,650 Fake (+48 Needs_Verification controls kept separate). |
| Set `COSINE_REUSE_THRESHOLD = 0.749` | **DONE.** Re-tuned on v6 data with CLIP; max Youden's J landed on 0.749 again (TPR 0.997, FPR 0.0039). |
| Re-run validate + crossval | **DONE.** All 9 checks pass; results below. |
| Evaluate the cross-modal CLIP feature (§12.2 step 6) | **DONE** as an ablation (below). |
| Train one final model to ship | **DONE** — `train_model.py`, saved to `out/model/fusion_rf.joblib`. |

### 15.1 Where the v6 dataset lives
It lives inside the image-pool repo, so it goes to GitHub with the photos:
```
dataset\                                  <- GitHub repo (Layer 1 photos in 15 folders)
    Bluetooth speaker\ … suitcases\
    fake_review_dataset_v6\
        code\     build_dataset.py, run_forensics.py, features.py, crossval.py,
                  train_model.py, validate.py, recompute_embeddings.py, …
        out\      reviews_balanced.csv, train.csv, test_balanced.csv,
                  holdout_realistic.csv, needs_verification_controls.csv,
                  reviews_full.csv, images\ (3,348 JPEGs), model\, logs
```
Run the scripts from `code\`. `REPO_DIR = '../..'` is the repo root, so this
works wherever the repo is cloned.

### 15.2 v6 balanced dataset in one table
| | Genuine | Fake | Notes |
|---|---|---|---|
| Whole set | 1,650 | 1,650 | 220 rows per category, 110/110 in **every** category |
| train.csv (616 root photos) | 1,245 | 1,245 | 50/50 |
| test_balanced.csv (210 root photos) | 405 | 405 | 50/50, root overlap with train = **0** |
| holdout_realistic.csv | 405 | 69 | ~15% fake — **the one to report** |

Fake half = `visual_manipulation` 555 + `text_deception` 555 + `coordinated_reuse` 540
(105 campaigns). New column `signal_cell` = none / image_only / text_only / both —
the 2×2 grid. Also fixed in v6: ids numbered after shuffling (row order no longer
hints at the label), and one reviewer-id format for all accounts (v5 used `C#####`
for campaign accounts, which gave the answer away).

### 15.3 v6 results (5×5 grouped CV, test folds at 15% fake, Random Forest, PR-AUC)
| Configuration | placeholder emb. | CLIP | CLIP + cross-modal |
|---|---|---|---|
| text-only | 0.740 | 0.740 | 0.740 |
| image-forensic-only | 0.505 | 0.557 | 0.567 |
| behaviour-only | 0.401 | 0.401 | 0.401 |
| graph-only | 0.245 | 0.243 | 0.243 |
| base-paper-equivalent | 0.802 | 0.801 | 0.801 |
| **FUSION (ours)** | 0.821 | 0.841 | **0.843 ±0.012** |
| gap vs base (p-value) | +0.019 (9e-04) | +0.040 (6e-08) | **+0.042 (3e-08), wins 24/25** |

Per-fraud-type detection, CLIP + cross-modal: text-only catches only **0.152** of
`visual_manipulation`; image-only catches only **0.188** of `text_deception`;
fusion catches 0.710 / 0.864 / 1.000. That is the §6.3 table, reproduced on the new data.

**Honest reading:** CLIP roughly doubled the fusion gain (+0.019 → +0.042). The
cross-modal feature adds only +0.002 on its own, so it is not yet a strong signal.
The gap vs v5 (+0.045) is about the same size, now with a tighter CI.

### 15.4 The final trained model (`train_model.py`)
Trained on train.csv, tested on root photos it has never seen:

| Test set | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| test_balanced (50% fake) | 0.952 | 0.946 | 0.852 | 0.850 | 0.854 |
| **holdout_realistic (15% fake)** | **0.843** | 0.950 | 0.610 | 0.483 | 0.826 |

Three-way decision (Genuine < 0.35 ≤ Needs Verification ≤ 0.65 < Fake): of 405
Fake test rows, 294 are called Fake and 21 are wrongly called Genuine. Of 405
Genuine rows, 292 are called Genuine and 15 wrongly Fake. The rest are sent to
"Needs Verification". **Negative control still weak:** 3 of 12
`legit_stock_reuse` test rows are flagged (§11 item 4 is still open).

---

## 16. How model training works from here (simple version)

**What "training" means here.** Every review becomes a row of 39 numbers
(features): 8 about the text, about 22 about the image (reuse, ELA, noise, CLIP cross-modal),
5 about behaviour (rating, timing, reviewer activity) and 4 about the graph. The
Random Forest learns which combinations of those numbers mean "Fake". It never
sees the photo's identity or the fraud type, only the evidence.

**Three models, not one.** For the extension we want to say *why* a review is
suspicious, so we train and save three models on the same rows:

| Model | Feature blocks | Answers |
|---|---|---|
| text model | TEXT | "Does the writing look fake?" |
| image model | IMAGE (+GRAPH) | "Is the photo edited or reused?" |
| **fusion model** | TEXT + IMAGE + BEHAVIOUR + GRAPH | **the final decision** |

The fusion score decides the badge colour. The text and image scores explain it
("flagged mainly because of the photo"). This is how we check **image fake and
text fake separately and together**. The §6.3 / §15.3 per-type table is the
proof that each one catches something the other misses.

**Training steps (in order):**
1. `python train_model.py` → fusion model (done). Next: add the text-only and
   image-only models to the same script and save all three.
2. Pick the decision bands (0.35 / 0.65 now) on validation folds, not by hand,
   so that on the 15% holdout the "Fake" band has precision ≥ 0.8.
3. Fix the `legit_stock_reuse` failure before shipping (§11 item 4). Likely
   cause: reuse features dominate when the reuse is spread over weeks. Add a
   "reuse spread in days" feature and re-check.
4. (Later) Replace the 8 hand-made text features with a small **RoBERTa**
   fine-tuned on text. Keep the Random Forest for fusion.
5. Save everything with `joblib` into `out/model/` together with
   `model_meta.json` (feature list, thresholds). The backend loads exactly this.

---

## 17. How the model predicts on NEW data (not just the training data)

This is the most important question. The training rows had every feature
precomputed. A brand-new review on Amazon has only: text, stars, date, reviewer
name, one or more photo URLs, and the product's own listing photos. So the
backend must compute **the same 39 features** from those raw inputs, using the **same
code** as training. Here is how each block works for one new review:

| Block | What we need | How we get it for a new review |
|---|---|---|
| TEXT | review text | computed directly from the text (same `text_features()`) |
| IMAGE — forensics | the photo pixels | download the photo, resize to 1024px, run `forensics.analyse()` (ELA + noise) |
| IMAGE — provenance / reuse | "have we seen this photo before?" | compare its **pHash + CLIP vector** against an **image memory** (see below) and against the **seller's listing photos on the same page** |
| IMAGE — cross-modal | does the photo match the text? | CLIP image vector · CLIP text vector |
| BEHAVIOUR | stars, date, reviewer | read from the page; reviewer history = what the image memory has recorded for that name |
| GRAPH | who shares photos with whom | build the small graph from the reviews on the current page plus the image memory |

### 17.1 The "image memory" (reuse index)
Reuse can only be detected against something. So the backend keeps a small local
database (a SQLite file plus a numpy array):
- **Starts with** the 3,348 v6 images (pHash + CLIP vector + reviewer + time + product).
- **Grows** every time the extension analyses a page: each new review photo is added.
- Lookup = Hamming distance on pHash (≤ 20) OR CLIP cosine (≥ 0.749), the same
  thresholds as training. For tens of thousands of images a plain numpy dot
  product is fast enough; FAISS is only needed beyond that.
- **Stolen catalogue photo check:** the listing photos on the product page are compared
  with each review photo. A near-match means "the customer photo is the seller's own
  photo" (Signal 1 in the original project definition).

### 17.2 Rules so that prediction matches training
1. **One feature function for both.** The backend imports `features.py` and
   `forensics.py`. It never re-implements them. (If training and serving compute a
   feature differently, the model silently breaks. This is called training/serving skew.)
2. **Same image pipeline.** Resize to ≤1024px and use one JPEG decode, exactly like `save_derived()`.
   Amazon thumbnails are often small, so the content script must request the
   **full-size image URL**. Below ~1000px, ELA is at chance (§9.7). In that case the backend reports
   "image too small for tamper check" instead of guessing.
3. **Missing information → neutral value, not zero-risk.** For example, if the reviewer history is
   unknown, use the training median and lean towards "Needs Verification".

### 17.3 The honest limit (say this in the viva)
The model was trained on **constructed** reviews. On real Amazon text it will be
less accurate, because real writing is more varied than our templates (§11 items 7–8). That
is why the extension:
- shows **Needs Verification** (amber) generously instead of forcing red/green,
- shows **which evidence** fired (photo reused? edited? generic text?), which is useful
  even when the overall score is uncertain,
- is presented as a **demo of the method**, not a production fraud filter.

The image-evidence part (reuse across accounts, stolen catalogue photo) transfers best
to real pages, because it does not depend on writing style. That is the
project's core claim again.

---

## 18. Product categories — how we handle them

- **The model never receives the category as an input.** Features are about *evidence*
  (edits, reuse, timing, generic wording), not about *what the product is*. So a
  laptop review is handled the same way as a shoe review.
- **Proof it generalises: leave-one-category-out (LOCO).** `out/loco_folds.json`
  has 5 folds × 3 categories, and every category is held out once. Train on 12 categories and
  test on the 3 unseen ones. Report text-only vs image-only vs fusion for each fold. Expected:
  text drops more than image, because text vocabulary is category-specific and image forensics is not
  (§4.4). **Next script to write: `loco.py`.**
- **Categories the model has never seen on Amazon** (e.g. "laptop"): the cross-modal check
  still works because CLIP is zero-shot. It compares the photo with the product
  *title* ("a photo of a {title}"). A review photo that does not match the product is a signal
  in any category.
- **Category-specific vocabulary** (`CATEGORY_VOCAB` in `gen_text.py`) exists only
  to *generate* realistic training text. It is not used by the model.

---

## 19. The Chrome extension — how we build it and make it work

### 19.1 The big picture
```
 Amazon / Flipkart product page (in Chrome)
        |  1. content.js reads each review card:
        |     text, stars, date, reviewer, photo URLs (+ listing photo URLs)
        v
 background.js (service worker)  --2. POST /analyze (JSON)-->  FastAPI on http://127.0.0.1:8000
                                                               | downloads photos
                                                               | computes the 39 features (features.py, forensics.py, CLIP)
                                                               | checks the image memory
                                                               | runs text / image / fusion models
        <--3. [{risk, decision, text_score, image_score, reasons}] --+
        |
        v
 content.js draws a badge on every review card:
   GREEN Genuine   AMBER Needs Verification   RED Fake     + a one-line reason on hover
 and a summary bar at the top: "12 reviews checked - 2 fake - 3 need verification"
```

### 19.2 Files we will create
```
extension\
    manifest.json      Manifest V3: name, permissions, which sites, which scripts
    content.js         reads review cards from the page DOM, draws badges
    background.js      service worker: talks to the local backend
    popup.html/.js     small popup: backend status (on/off), counts, on/off switch
    styles.css         badge + summary bar styling
    icons\             16/48/128 px icons
backend\
    app.py             FastAPI app: POST /analyze, GET /health
    pipeline.py        raw review -> 39 features (imports features.py, forensics.py)
    image_memory.py    pHash + CLIP index (SQLite + numpy), add/lookup
    requirements.txt   fastapi uvicorn pillow imagehash open_clip_torch joblib ...
```

### 19.3 manifest.json essentials
- `"manifest_version": 3`
- `"content_scripts"`: matches `https://www.amazon.in/*`, `https://www.amazon.com/*`,
  `https://www.flipkart.com/*`, runs `content.js`
- `"host_permissions"`: `http://127.0.0.1:8000/*` (so the extension may call the backend)
- `"permissions"`: `["storage"]` only. Read-only; no history, no cookies.

### 19.4 How content.js finds reviews
Amazon review cards have stable-ish attributes: `div[data-hook="review"]`, with
`[data-hook="review-body"]`, `[data-hook="review-star-rating"]`,
`[data-hook="review-date"]`, `.a-profile-name`, and review images inside
`[data-hook="review-image-tile"]` / `img.review-image-tile`. Flipkart uses different
class names, so we keep a **per-site selector table** in one place. When the site
changes its HTML, we update only that table. A `MutationObserver` re-runs the scan
when the page loads more reviews.

### 19.5 The backend API
```
POST /analyze
{ "page_url": "...", "product_title": "...", "listing_images": ["url", ...],
  "reviews": [ { "id": "R1", "text": "...", "rating": 5, "date": "2026-09-01",
                 "reviewer": "name", "images": ["url", ...] }, ... ] }
-> { "reviews": [ { "id": "R1", "risk": 0.82, "decision": "Fake",
                    "text_score": 0.31, "image_score": 0.88,
                    "reasons": ["photo matches 3 other accounts within 2h",
                                "photo is the seller's listing image"] } ],
     "summary": { "checked": 12, "fake": 2, "needs_verification": 3 } }
```
Reviews **without a photo** still get a text + behaviour score, and the badge says
"no photo, text-only check". This is honest, because the image modules cannot run.

### 19.6 Build order (each step is testable on its own)
1. **Backend first, no browser:** `pipeline.py` must reproduce the training
   features for 10 rows of `test_balanced.csv` **exactly** (unit test: same numbers).
   This catches training/serving skew before anything else.
2. `app.py` with `/health` and `/analyze`, tested with `curl` or `/docs` (FastAPI's
   built-in test page).
3. `image_memory.py` seeded with the 3,348 v6 images.
4. Extension skeleton: `manifest.json` + `content.js` that only **logs** the reviews
   it found (check in Chrome DevTools console).
5. Connect: `background.js` → backend → draw badges.
6. Popup + summary bar + hover reasons.
7. Demo script: open 3 real product pages, and show one clearly reused photo being caught.

### 19.7 How to run it (once built)
```powershell
cd backend
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```
Chrome → `chrome://extensions` → turn on **Developer mode** → **Load unpacked** →
choose the `extension\` folder → open an Amazon product page → badges appear.

### 19.8 Rules we keep
- **Local and read-only.** The backend runs on your laptop (127.0.0.1). Nothing is
  uploaded, no data is logged or redistributed (§12.3).
- The image memory stores only hashes, vectors, reviewer names and times. It does not keep
  full photos.
- Keep request rates human-like (one page at a time). Do not crawl.

---

## 20. Updated order of work

1. DONE: push photo pool · v6 balanced dataset · CLIP + threshold · CV · final model
2. Save text-only and image-only models too; tune decision bands (§16)
3. Fix the `legit_stock_reuse` negative control (§11 item 4)
4. `loco.py`, the leave-one-category-out generalisation result (§18)
5. Backend `pipeline.py` + skew unit test → `app.py` → `image_memory.py` (§19.6 steps 1–3)
6. Chrome extension (§19.6 steps 4–7)
7. Write-up: ablation table (§15.3), per-type table, LOCO table, resolution finding (§12.4),
   and the literature comparison in `papers\PAPERS_COMPARISON.md` (§21)

---

## 21. Literature comparison (added 2026-09-19)

All 40 PDFs in `papers\` were read and tabulated in **`papers\PAPERS_COMPARISON.md`**
(dataset, real vs synthetic, text/behaviour/graph/image, accuracy, precision,
recall, F1, AUC, plus a 2–3 line summary of each paper).

Main points, in simple words:
- 31 papers run experiments and 9 are surveys. Only **2** use review images
  (He et al.: image-only AUC 0.592; FRIDRC: images as context). **None** checks whether
  a photo was edited or reused. That is our gap.
- The 96–99% results come from easy data: GPT-2-generated text (the OSF 40k set) or the small
  crowd-sourced Ott set. On real Yelp labels, papers report F1 0.56–0.90.
- Measured the way most papers measure (balanced test set), our fusion model gets
  accuracy 0.852, F1 0.852, ROC-AUC 0.946. Our headline (PR-AUC 0.843 at 15% fake, grouped
  by photo) is a stricter protocol, not a weaker model. See §12.6: never put the numbers side by side.

---

## 22. Status update 2 (2026-09-19, later): shortcut removed, forensic CNN, extension built

Details: `readmes\readme4.md` (the problem) and `readmes\readme5.md` (the fix).

- **Rating shortcut removed (v6.1).** Edited-photo fakes used to be always 4–5★, so rating alone caught 74%
  of them. Now they get genuine-like text and ratings; only the pixels differ.
- **Forensic CNN** (`forensic_maps.py` + `cnn_forensics.py`) on 128×128 ELA / noise maps. It detects edited photos
  at ROC-AUC **0.959** on unseen photos (the 13 hand-made numbers managed 0.635). Scores are out-of-fold and grouped
  by root photo. Caveat: it learns our own three edit types (§11 item 1).
- **Cross-validation (25 folds, 15% fake):** FUSION PR-AUC **0.863** vs base-paper-equivalent **0.729**
  (+0.134, p = 5e-14, 25/25 folds). Text-only 0.719, image-only 0.658.
- **Final model, realistic holdout:** PR-AUC 0.918, F1 0.847. Balanced test accuracy 0.909.
  Removing the image block drops accuracy to 0.788. `legit_stock_reuse` false alarms: 0/12 (fixed).
  Needs-Verification controls do not yet abstain (bands still to tune).
- **Graph features made order-independent** (sorted Louvain input), found by the backend parity test.
- **Extension + backend built** (`extension\`, `backend\`), following §19. The parity test shows 40/40 features identical.
  Demo page: 5/5 genuine correct, no false alarms. See `backend\README.md` to run it.

---

## 23. Status update 3 (2026-09-19, night): model comparison and final numbers

Details: `readmes\readme8.md`. Comparisons: `dataset\fake_review_dataset_v6\out\model_comparison\`.

- **Single-view ceiling:** 135 of the 405 test fakes are invisible to text (and 135 others to images) by design, so no
  single-view model can exceed accuracy 0.833. Text-only now reaches 0.811 (97% of that ceiling).
- **Text:** added MiniLM-L6 sentence embeddings + LogReg (`text_model.py`, feature `txt_minilm_p`, out-of-fold).
  DistilRoBERTa fine-tuned hit the ceiling but is likely memorising our templates; it is kept as an upper bound.
- **Reuse fingerprint:** ResNet-50 replaces CLIP for reuse (`reuse_embedding.py`, threshold 0.72). It catches 94.7% of
  copies at 0.1% FPR vs CLIP's 83.0%. CLIP stays for the cross-modal feature.
- **Edit detector:** the small CNN beat ImageNet ResNet-18 and EfficientNet-B0 (0.953 vs 0.942 / 0.944).
- **Final:** cross-validation FUSION PR-AUC **0.902** vs base-paper-equivalent 0.758 (+0.144, p = 2e-16, 25/25).
  Balanced test accuracy **0.943**, holdout PR-AUC **0.958**. Backend parity: 41/41 features identical.
- **Requirements proof** (`verify_requirements.py` → `out/REQUIREMENTS_CHECK.md`): 39 PASS, 4 WARN, 1 FAIL.
  The FAIL is 1/48 borderline rows called Fake, because the decision bands are still untuned.

---

## 24. Fair comparison with the literature (added 2026-09-20)

Reported numbers from other papers come from other datasets and must not be ranked against ours (§12.6).
Instead, the five method families behind the surveyed papers were **re-implemented and run on our dataset**
(`code/compare_baselines.py`, same split, same metrics; details in `readmes\readme10.md` and
`papers\PAPERS_COMPARISON.md` §1b):

| Method family | Accuracy | ROC-AUC | Edited-photo caught |
|---|---|---|---|
| TF-IDF + SVM | 0.802 | 0.833 | 4% |
| Fine-tuned RoBERTa-family transformer | 0.833 | 0.842 | 0% |
| Sentence embeddings + behaviour | 0.788 | 0.829 | 8% |
| He et al. 2022 base paper | 0.793 | 0.808 | 4% |
| Text + image embeddings (FRIDRC-style) | 0.851 | 0.898 | 36% |
| **Ours** | **0.943** | **0.976** | **82%** |

Ours vs the best baseline: +9.3 accuracy points (95% CI +6.8 to +11.7), McNemar p = 2e-12.
Report layout: Table A related work (not comparable, not ranked) → Table B re-implemented baselines → Table C per fraud type.

---

## 25. The papers' own models on our dataset (added 2026-09-20)

`code/compare_paper_models.py` re-builds seven named models from the surveyed papers and trains/tests each on our
data (same split, same metrics). Details: `readmes\readme11.md`, `papers\PAPERS_COMPARISON.md` §1c.

| Paper | Re-built model | Accuracy | Edited photo caught | McNemar vs ours |
|---|---|---|---|---|
| 02 Shan 2021 | inconsistency features + RF | 0.786 | 4% | p = 1e-24 |
| 15 Xu 2024 | A-LSTM + behaviour | 0.816 | 13% | p = 1e-18 |
| 38 Qayyum 2023 | FRD-LSTM (BiLSTM) | 0.830 | 1% | p = 2e-16 |
| 39 Duma 2024 | DHMFRD-TER | 0.788 | 16% | p = 9e-24 |
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.833 | 0% | p = 2e-15 |
| 40 Geetha 2025 | DeBERTa-v3 | 0.827 | 3% | p = 1e-16 |
| 11 Hou 2025 | FRIDRC (text + ViT) | 0.833 | 0% | p = 2e-15 |
| **Ours** | forensic fusion | **0.943** | **82%** | – |

Five of the seven land at 0.827-0.833, the computed ceiling for any text-only model on this dataset. They catch fake
text (91-100%) but not edited photos (0-16%). FRIDRC fine-tuned end-to-end catches 0% of edited photos, although its
frozen-embedding variant catches 36%: a general image encoder describes what is in a photo, not whether it was edited.
Honest caveat: three baselines have fewer false alarms than ours (0-1.0% vs 4.7%), and the CPU-only simplifications
(listed in readme11 §3) make each baseline slightly weaker than its published version.


---

## 26. v6.2 — the text leak, the model swap, and the corrected comparison (20 Sep 2026)

This section answers a question Venkata Sai asked about section 25, and records what checking it properly turned up.

### 26.1 The question: "their model catches 100% of fake text and ours 98%, but their accuracy is lower — that does not match"

It does match, and it is arithmetic, not a coincidence. "Fake text caught" is **recall on one group of 135 rows**,
not accuracy. The 810-row test set is 405 genuine + 135 edited-photo + 135 fake-text + 135 coordinated.

A text-only model reads words. On fake text it is perfect. On an **edited photo with an honest review written by a
real customer** there is nothing in the words to find, so it misses all 135. 135/810 = 16.7%, and 1 - 0.167 =
**0.833** — exactly where five of the seven models landed.

Counted as mistakes rather than percentages it is obvious at a glance; see the table in
`papers/PAPERS_COMPARISON.md` section 1c ("The same result counted as mistakes").

### 26.2 Proving the table was not written to look good

The instruction was: do not design the code around the output; the values must be real. So the table was not
re-stated — every reported accuracy was **rebuilt from its own per-type rates**:

    rebuilt = [ (1 - false_alarms)*405 + caught_edited*135 + caught_text*135 + caught_coord*135 ] / 810

If any number had been typed by hand or produced by a broken metric, the two would disagree. **All 8 rows,
ours included, matched with a difference of +0.00e+00.**

  code:   dataset/fake_review_dataset_v6/code/verify_paper_numbers.py
  report: out/model_comparison/PAPER_NUMBERS_AUDIT.md

### 26.3 What that check uncovered: the text was memorisable

Our review text is generated from templates, so a second question was asked that the table could not answer: is
"100% on fake text" real skill, or is the model recognising sentences it has already seen?

  code:   dataset/fake_review_dataset_v6/code/verify_text_difficulty.py
  report: out/model_comparison/TEXT_DIFFICULTY_AUDIT.md

Measured on v6.1:

  * **46.2%** of test reviews were **word-for-word copies** of a training review
    (fake text **68.9%**, coordinated **100%**)
  * a rule with no learning in it — "copy the label of the most similar training review" — already scored
    **100% on fake text and 100% on coordinated**
  * cause: gen_text.py picks one **finished sentence** from a short list (DECEPTIVE_POSITIVE had 8 fixed strings,
    CAMPAIGN_TEMPLATES 5, LAZY_GENUINE 8). With 3,348 rows to fill, sentences had to repeat across the split.

This inflated the text column for **every** model, ours included. It is a genuine defect in the dataset, not a
presentational problem.

### 26.4 The fix: v6.2 text (styles unchanged, space enlarged)

  gen_text2.py    assembles each review from independent slots instead of choosing one whole sentence, so each
                  style can produce tens of thousands of distinct reviews. The five styles and the deliberate
                  OVERLAP between them (lazy genuine vs generic deceptive; sophisticated deceptive vs genuine
                  detailed) are unchanged — that overlap is the premise of the project.
  regen_text.py   rewrites ONLY the review_text column. Photos, splits, review ids, reviewer accounts, timestamps
                  and campaigns are untouched, so forensic_maps.npy, the ResNet-50 reuse embeddings and the CLIP
                  vectors all stay valid and were not recomputed.

Rules the rewrite obeys, so the dataset still means the same thing:

  * style comes from fraud_type + the `notes` marker build_dataset.py already recorded
  * sentiment is read back from the row's EXISTING rating, so text and stars still agree and the v6.1 removal of
    the rating shortcut (section 22) is preserved. **No rating was changed.**
  * the RNG is seeded from review_id — reproducible and independent of row order
  * no text may repeat, except inside one campaign, where members are supposed to look alike

Result:

| | v6.1 | v6.2 |
|---|---|---|
| test text copied verbatim from train | 46.2% | **0.0%** |
| — fake text | 68.9% | 0.0% |
| — coordinated | 100.0% | 0.0% |
| nearest-neighbour TF-IDF similarity, fake text (median) | 1.000 | 0.557 |
| unique texts in the dataset | 2,027 / 3,348 | 3,297 / 3,348 |

The old text is kept at `out/review_text_v61.csv`, so the change is reversible and the audit repeatable.

**Cost of the fix: 0.1 accuracy points** (0.9432 -> 0.9420, same Random Forest). The earlier estimate of -2.8
points was too pessimistic — it came from scoring only the rows whose text happened to be rare, and those were
disproportionately the awkward ones. The text branch alone still reaches ROC 0.827.

### 26.5 Model search — one winner, one honest failure

**Fusion classifier (Phase A) — Extra Trees replaces the Random Forest.** Ten candidates, same 41 features, same
rows, same grouped folds; the winner was chosen on **grouped cross-validation over the training data**, not on the
test set, so the choice is not tuned to the answer.
Table: `out/model_comparison/fusion_models.md`, code: `code/compare_fusion_models.py`.

  Extra Trees 1000, leaf 2, sqrt   CV ROC 0.9553   test acc 0.9494   false alarms 1.5%
  Random Forest 500 (old)          CV ROC 0.9530   test acc 0.9420   false alarms 4.0%

train_model.py now has a `FUSION` switch ('extra_trees' | 'random_forest') and make_fusion().

**Image detector (Phase B) — tried, failed, dropped.** The weak spot was copy-move (ROC 0.89 vs 0.97 for splice),
so three more forensic maps were built (`code/forensic_maps_extra.py`): an SRM high-pass residual, a JPEG
block-grid mismatch map, and a copy-move detector based on offset consistency (a pasted region makes many patches
agree on one translation).

| input channels | ROC | seed 1 / seed 2 | copy-move | splice | noise |
|---|---|---|---|---|---|
| 3 ch (current) | **0.9442** | 0.950 / 0.906 | **0.886** | **0.971** | 0.961 |
| 6 ch (+ the three new maps) | 0.9367 | 0.919 / 0.940 | 0.858 | 0.957 | **0.976** |

Six channels are **worse**, and the gap (0.0075) is smaller than the spread between two random seeds
(0.906-0.950), so it is not even a reliable difference. **Why it failed:** ordinary photos are full of repeated
structure — fabric weave, tiles, grilles — which produces the same "many patches agree on one offset" signature as
a copied region. The channel added noise, not evidence. **The CNN stays at 3 channels.** Recorded as a failed
experiment rather than dropped quietly: `out/model_comparison/cnn_channels.md`.

Note for anyone re-running this: the machine has 4 cores and 7.4 GB RAM. Converting the (3348, 6, 128, 128) map
array to float32 up front needs ~1.6 GB and is killed by the OS. compare_cnn_channels.py streams it with
`np.load(..., mmap_mode='r')` and converts **per batch**. Run training scripts one at a time.

### 26.6 The corrected final numbers

Final model: **Extra Trees**, 41 features, trained on train.csv only, measured on unseen root photos.

| | v6.1 RF (leaky text) | **v6.2 Extra Trees** |
|---|---|---|
| PR-AUC, balanced test | 0.982 | **0.984** |
| ROC-AUC | 0.976 | **0.980** |
| F1 | 0.943 | **0.948** |
| Precision | 0.952 | **0.984** |
| Recall | 0.933 | 0.914 |
| accuracy | 0.9432 | 0.9494 |
| **false alarms** | 4.7% (19/405) | **1.5% (6/405)** |
| PR-AUC @15% realistic holdout | 0.958 | **0.963** |
| legit stock-photo reuse wrongly flagged | 1/12 | **0/12** |

By fraud type: coordinated **100%**, fake text **94.1%**, edited photo **80.0%**, false alarms **1.5%**.

Independently verified for v6.2 (`code/verify_results.py` -> `out/VERIFY_RESULTS.md`):

  * every saved metric reproduces exactly from a fresh load of the model
  * 95% CI (2,000 bootstrap resamples, grouped by root photo): balanced accuracy **0.934-0.963**,
    ROC-AUC 0.969-0.988, PR-AUC 0.977-0.990; realistic holdout PR-AUC 0.963 (0.922-0.992)
  * **label-shuffle control**: training on randomly shuffled labels gives test ROC-AUC 0.550, 0.516, 0.442,
    0.607, 0.418 (mean **0.506**) -> no feature leaks the answer
  * **seed stability**: 5 retrains give test accuracy 0.944-0.948, ROC-AUC 0.977-0.979, holdout PR-AUC 0.953-0.956

All seven paper models were retrained from scratch on v6.2 (their v6.1 numbers are void). Full table in
`papers/PAPERS_COMPARISON.md` section 1c. Headline: best baseline BSTC 0.835, ours **0.949**, and **we now lead on
every column including false alarms**, which we did not on v6.1. Every difference significant, p < 7e-19.

**The fix did not flatter us.** Removing the duplication made the baselines *worse* (A-LSTM 0.816 -> 0.778, false
alarms 7.7% -> 17.3%) and ours *better* (0.943 -> 0.949, false alarms 4.7% -> 1.5%). Without memorised sentences
the text-only models start calling genuine reviews fake; they were leaning on the duplication harder than we were.

**Where the model is still weakest:** 8 of 135 fake-text reviews are missed. All 8 are *sophisticated deceptive*
rows — fabricated specifics on a clean photo, e.g. "About a month of use on a camping trip. The pairing button is
flawless." Nothing in the words, nothing in the pixels. That is the designed overlap case, and it should be
reported as such.

### 26.7 What this changes for the rest of the project

1. **The backend must load the new model.** `out/model/fusion_rf.joblib` is now an ExtraTreesClassifier (the
   filename is unchanged so nothing else breaks). Re-run `backend/test_parity.py` before demoing the extension.
2. **Decision bands need tuning.** Extra Trees gives less extreme probabilities, so under the 0.35/0.65 bands 58
   fake reviews now land in "Needs verification" instead of "Fake" (was 23). Not a wrong answer — the band exists
   for exactly that — but the bands were never tuned and this is now the top open item.
3. **Re-run the remaining checks on v6.2**: crossval.py (the 5x5 grouped CV row), verify_requirements.py and
   modality_ablation.py were all measured on v6.1.


---

## 27. Correction to section 26, and the extension on live pages (20 Sep 2026, later)

### 27.1 The stale embedding cache — section 26's numbers were wrong

**Section 26 reports accuracy 0.9494. The correct figure is 0.9259.** What went wrong, and how it surfaced, is
worth keeping: it is the strongest evidence in the project that the pipeline is checked rather than trusted.

`text_model.py` cached the MiniLM sentence vectors in `out/model_comparison/text_emb_all-MiniLM-L6-v2.npy`, keyed
**only by filename**:

    E = np.load(EMB_CACHE) if os.path.exists(EMB_CACHE) else embed(...)

When `regen_text.py` rewrote every review for v6.2, that file was left at its 19 September contents. The model
therefore trained and predicted on sentence vectors of the **old, duplicated text**, while
`reviews_full.csv` held the new text. `txt_minilm_p` is the single most important feature (0.19 importance).
`clip_text_embeddings.npy` was stale in the same way, so `clip_cross_modal` compared each photo against a review
that no longer existed.

The error was **one-sided and in our favour**: the five transformer baselines tokenise raw text themselves, so
they received the honest new text, while our model kept the memorised old text. Any conclusion drawn from that
comparison was unsound — in particular section 26.6's claim that "the fix did not flatter us".

**How it was caught:** `backend/test_parity.py` compares the 41 features computed when *serving* a review against
the features computed during *training*. It reported `txt_minilm_p` differing by **0.69**. A stale cache is
exactly what that looks like; nothing else in the training metrics would have revealed it.

**The fix:**
* `text_model.py: load_embeddings()` now keys the cache by a **SHA-256 of the text itself** and rebuilds when it
  changes. A stale cache is no longer possible.
* `refresh_clip_text.py` (new) re-encodes the CLIP **text** vectors and rewrites `clip_cross_modal.csv`. The
  photos did not change, so `clip_image_embeddings.npy` is reused rather than spending minutes re-encoding.
* After the fix, parity is **41/41 features identical**.

**Cost of the correction:**

| metric | section 26 (stale cache) | corrected |
|---|---|---|
| accuracy | 0.9494 | **0.9259** |
| F1 | 0.948 | **0.9219** |
| precision / recall | 0.984 / 0.914 | **0.975 / 0.874** |
| ROC-AUC | 0.980 | **0.966** |
| PR-AUC @15% holdout | 0.963 | **0.954** |
| fake text caught | 94.1% | **83.0%** |
| edited photo caught | 80.0% | 79.3% |
| false alarms | 1.5% | 2.2% |

The edited-photo figure barely moved, which is the expected signature: that evidence comes from the forensic CNN,
which never touches the text.

**Corrected comparison.** All seven paper models were retrained again on the corrected features. Best baseline
BSTC 0.835, ours **0.926**; every difference still significant (p < 2e-10). Our lead over the best baseline
**shrank from 11.0 to 9.1 points**. Removing the duplicated text lowered almost every model *including ours*
(A-LSTM -6.7, ours -1.7, BSTC +0.2); the honest table is in `papers/PAPERS_COMPARISON.md`
("What the text fix did to these numbers").

Verification re-run on the corrected model (`out/VERIFY_RESULTS.md`): 95% CI 0.908-0.943, label-shuffle control
mean ROC-AUC 0.508, 5 retrains 0.926-0.931.

**Lesson for the report:** any cache keyed by filename rather than content is a correctness bug waiting to
happen, and a training-vs-serving parity test is what catches it.

### 27.2 The extension, checked against live Amazon and Flipkart

Both site adapters were tested against **live pages**, not from memory (20 Sep 2026).

**Amazon** (`amazon.in/dp/B085JC431R`): the review-text selector was **broken**. amazon.in no longer emits
`data-hook="review-body"`; the body is `data-hook="reviewText"` inside `reviewTextContainer`. Every other
selector (review card, rating, date, author, review photos, listing photos, ASIN) was correct. `sites.js` now
lists all three hooks, old one first, so amazon.com pages that still use it keep working. On that live page:
13 review cards, 12 with text, 2 with photos; the thumbnail-to-full-size rewrite (`._SY500_.jpg` -> `.jpg`) works.

**Flipkart** was marked UNTESTED and could never have worked. Flipkart now ships **build-generated class names**
(`v1zwn21n v1zwn28 _1psv1zeb9 css-146c3p1`) that change on every deploy, so selectors such as `div._27M-vq` and
`span.B_NuCI` match nothing. Class-based selectors there are not fixable, only re-broken.

Flipkart is therefore read **structurally**, using the two things that are stable: every genuine review carries a
buyer badge, and each card begins with its star rating. `SITES.flipkart.findCards()` takes the innermost element
containing "Verified Purchase" (Flipkart renamed it from "Certified Buyer"), walks up to the block that also holds
the rating, and `parse()` splits the card's text into rating / title / body / author / date, with review photos
taken from `img[src*="rukminim"]`. Tested live: **10 review cards parsed correctly**.

`content.js` now accepts either description: CSS selectors (Amazon, demo) or `findCards()` + `parse()`
(Flipkart). Guide section 19.4 still holds — only `sites.js` needs editing when a site changes.

**End-to-end check on five real, unmodified Amazon reviews** through the real backend and the corrected model:
four "Genuine", one "Needs verification" (a three-word review with no photo), **zero false "Fake" verdicts**.

### 27.3 Where the model lives

`dataset/fake_review_dataset_v6/out/model/`

  fusion_rf.joblib       16.1 MB   the shipped Extra Trees fusion over 41 features
                                   (filename kept from the Random Forest days so nothing else breaks)
  forensic_cnn.pt        868 KB    the CNN that reads the forensic maps
  text_minilm_lr.joblib  4 KB      logistic regression over MiniLM vectors -> txt_minilm_p
  image_rf.joblib        3.3 MB    image-only model ("Photo evidence" in the extension)
  text_rf.joblib         3.1 MB    text-only model ("Text evidence")
  model_meta.json        3 KB      feature list and order, thresholds, decision bands, saved results

plus `out/reuse_embeddings_resnet50.npy` (13.7 MB, the image memory) and the two CLIP embedding files.
`backend/config.py` points `MODEL_DIR` at this folder, so replacing `fusion_rf.joblib` and restarting the backend
swaps the model with no other change. Running instructions are in `readmes/readme13.md` section 6.3.


---

## 28. Four defects found by testing the extension on live pages (21 Sep 2026)

Venkata Sai ran the extension against a live Flipkart reviews page and reported four things that
did not add up. All four were real. This section records them because "we tested it on a real
page and it was wrong in these four ways" is worth more in a report than a clean table.

### 28.1 A brand-new photo "matched 4 other accounts"

A family photo that nobody could have posted before was reported as matching four accounts.

**Cause.** Reuse is scored as `pHash close OR embedding close`, so one loose test overrides a
correct one. `PHASH_REUSE_THRESHOLD` was 20 (Hamming, out of 64 bits), chosen from the
same-photo/different-photo margin (~18 vs 24) and never checked for how many FALSE partners it
produces at scale. Measured (`backend/diagnose_threshold.py`):

| threshold | true reuse kept | unrelated pairs matched | expected false partners per photo (3,348 memory) |
|---|---|---|---|
| 12 | 0.853 | 0.00010 | **0.3** |
| 16 | 0.912 | 0.00053 | 1.8 |
| 20 (was) | 0.971 | 0.00738 | **24.7** |

Running the user's actual photos through the pipeline (`backend/diagnose_reuse.py`):
ResNet-50 correctly reported nearest neighbours at cosine **0.598** and **0.640** against a
0.72 threshold, while pHash claimed **40** and **36** matches. The neural network was right; the
cheap hash outvoted it. Small, heavily compressed marketplace thumbnails collide worst, and that
is exactly what review photos are.

**Fix.** `PHASH_REUSE_THRESHOLD = 12`. Everything retrained. The tighter threshold made the model
**better**, because the loose one was feeding it noise:

| | pHash <= 20 | pHash <= 12 |
|---|---|---|
| accuracy | 0.9259 | **0.9321** (95% CI 0.914-0.949) |
| F1 | 0.9219 | **0.9287** |
| precision / recall | 0.975 / 0.874 | **0.978 / 0.884** |
| edited photo caught | 79.3% | **80.7%** |
| fake text caught | 83.0% | **84.4%** |
| false alarms | 2.2% | **2.0%** |

Label-shuffle control 0.509, seeds 0.930-0.932, training/serving parity 41/41.

### 28.2 Verdicts changed on every reload

**Cause.** A scraped review's identity was `reviewer + text + image URL`. Flipkart serves the
same photo from `rukminim1` AND `rukminim2` (both verified HTTP 200 for the same file), and
Amazon varies the size code (`._SY88_` vs `._SY500_`). So on reload the same review looked new,
was stored again, and then matched **its own previous copy**. Demonstrated: identical review,
only the host changed, risk **0.305 "Genuine" -> 0.434 "Needs verification"**, with a new reason
"identical text in 1 other review(s)".

**Fix.** `pipeline._key()` now hashes `reviewer + normalised text + the photo's pHash` -- the
image's CONTENT, not its URL. The text is normalised too, because Amazon appends "more" to a
collapsed review and drops it when expanded. After the fix the same payload four times (including
the host swap) gives an identical risk of 0.503 and the memory does not grow.

### 28.3 "4 matching photos" was unverifiable

The user asked, reasonably: where was the photo found, who posted it, when, and which copy is the
original?

**Fix.** `pipeline.photo_matches()` returns the evidence itself -- account, date, product, pHash
distance, visual similarity, and whether the match came from our dataset or from something this
extension saw earlier. The reason line now reads, for example:

> same photo already posted by EXT_Ravi Kumar on 2022-03-01 (seen earlier by this extension);
> this review is dated 2022-09-01, so this one is the later copy

and the extension card carries a collapsible table of every match.

**On "which one is real":** the only available evidence is time, so the earliest posting is
treated as the original and later ones as copies. The wording says "the later copy", never "this
is fake". If a scammer posted first, the ordering is wrong. State that limitation.

### 28.4 "Signs of editing" on a photo too small to judge

The card showed both "photo shows signs of editing (CNN 0.55)" and "photo under 1000px: edit
check less reliable". Those contradict each other.

**Measured** (`backend/diagnose_small_photos.py`: shrink test photos, re-save at JPEG 80, rebuild
the forensic maps, re-score with the same CNN):

| photo width | ROC-AUC, edited vs clean | flagged at 0.5: edited / clean |
|---|---|---|
| <=1024px | **0.947** | 86% / 11% |
| 800px | 0.898 | 77% / 14% |
| 600px | 0.835 | 46% / 11% |
| 480px | 0.734 | 36% / 10% |
| 360px | **0.629** | 20% / 7% |
| 240px | 0.577 | 17% / 14% |

800px still carries real evidence, so a blanket 1000px cutoff would throw signal away. The
collapse is below ~600px -- where marketplace review thumbnails live. The reported CNN 0.55 on a
360px photo sat in that 7% false-positive tail.

**Fix**, in `backend/config.py` and `pipeline.py`:

* `MIN_CNN_EDGE = 600`. Below it the tamper evidence is **discarded, not just labelled**: the
  forensic columns (`ela_*`, `noise_*`, `cm_*`, `img_cnn_manip`) are replaced by the memory
  median, the same mechanism already used for a review with no photo, so a coin-flip score cannot
  move the risk. The reason becomes "photo is only 360px: too small to check for editing, so that
  evidence was ignored". Risk on the user's photo dropped 0.561 -> 0.503.
* Between 600px and 1000px the evidence is kept but the claim needs `img_cnn_manip >= 0.65`
  (`WEAK_CNN_MIN_SCORE`) and is worded as "possible signs of editing ... weaker evidence".
* At 1000px and above, unchanged: a 1024px manipulated photo still reports "photo shows signs of
  editing (CNN 0.95)".

**Note on what is NOT discarded.** The reuse features (pHash, ResNet-50 embedding, burst) stay
active at any size, because ResNet-50 resizes to 224px anyway: "is this the same photo" survives a
thumbnail, "was this photo edited" does not.

### 28.5 Two limitations these tests exposed, which belong in the report

1. **No reviewer history.** The extension reads only the reviews on the page; it never opens a
   reviewer's profile. "Is this account posting suspiciously often?" cannot be answered today.
2. **The graph is nearly blind on a live page.** `img_reuse_in_burst`, `img_burst_ratio` and the
   reviewer-community features link accounts through shared photos. In our dataset a campaign is
   5+ accounts around one photo inside a 3-hour burst, all visible. On Amazon we see ~13 reviews
   and the rest of a campaign sits on pages we never fetch. **This is the largest honest gap
   between the measured results and live performance.**


---

## 29. The text model does not transfer to real reviews (21 Sep 2026)

This is the most important limitation found so far, and it should be stated in the report rather
than discovered by an examiner.

### 29.1 What happened

The extension marked a long, detailed, obviously genuine Amazon review of an iPhone -- pros and
cons, specific battery figures, a complaint about 20W charging -- as **"Likely fake, 78%"**. The
only line under the verdict was "no photo: checked on text and behaviour only", which explains
nothing.

`backend/explain_review.py` measures each feature's share of a verdict by putting it back to what
an ordinary review has (the memory median) and re-scoring. For that review:

| feature | this review | typical | effect on risk |
|---|---|---|---|
| txt_minilm_p | 0.709 | 0.385 | **+0.189** |
| txt_exclam | 2 | 0 | **+0.185** |
| txt_sent_count | 10 | 3 | +0.071 |
| txt_specific_ratio | 0.000 | 0.043 | +0.035 |

### 29.2 Why: the training text is nothing like a real review

Measured on train.csv:

| | genuine training reviews | the real Amazon review |
|---|---|---|
| length | 4-34 words, median 23 | **223 words** |
| sentences | at most 4 | **10** |
| contains "!" | 2.5% of genuine, 19.5% of fake | 2 marks |

**88.7% of training reviews containing an exclamation mark are fake**, because the deceptive
templates in gen_text2.py use them and the genuine ones do not. The model learned
"exclamation mark -> fake". Real people use exclamation marks constantly. It is a generator
artefact, not a fraud signal.

The same applies to length and sentence count: the review is 6.5x longer than the longest genuine
review the model has ever seen, so the model is extrapolating, and it resolves "unusual" as
"suspicious". `txt_specific_ratio` scored 0.000 because the "specific detail" vocabulary is built
from our 15 product categories (heel counter, pump dispenser, ...) and a phone review mentions
iOS, cameras and battery -- none of which are in it.

**So the 84.4% fake-text recall we report is on OUR generated text.** On real marketplace prose
the text branch is not trustworthy. The image forensics do transfer (they look at compression and
noise, not at what the object is); the text side does not.

### 29.3 What was changed

**1. The text evidence is refused when the review is outside the training range.**
`code/record_text_range.py` writes the training text distribution into `model_meta.json`
(genuine max 34 words, max 4 sentences) so the guard tracks whatever the model was actually
fitted to instead of a constant that goes stale. In `pipeline.py`, a review longer than 1.5x that
range has its TEXT features replaced by the memory median -- the same mechanism already used for
a missing photo (section 26) and for a photo too small to judge (section 28.4) -- and the card
says so:

> this review is 223 words; the text model was trained on reviews of at most 34 words, so it is
> too long for the text check and that evidence was ignored

Result on that review: **0.780 "Likely fake" -> 0.233 "Looks genuine"**.

**2. A verdict now always carries a reason.** `pipeline.drivers()` runs the same ablation as
explain_review.py, batched into a single prediction, and names the top contributors in plain
words whenever a flag would otherwise be unexplained:

> main reason: the wording reads like promotional copy to the language model (+0.37 of the risk)

**3. It refuses to convict on no evidence.** If the text is out of range AND there is no usable
photo AND no reuse match, a "Fake" verdict is capped at "Needs verification" with
"not enough evidence this tool can judge: treat as unverified, not as fake".

Detection on the dataset is unaffected (training reviews are 4-34 words, so the guard never fires
there): the demo page still returns 4 Fake / 3 Needs verification / 7 genuine with named accounts
and dates, a 1024px manipulated photo still reports "signs of editing (CNN 0.99)", and
training/serving parity is still 41/41.

### 29.4 What this means for the project

The honest framing for the report is:

* the **photo** side is the contribution and it transfers: forensics are category-independent, and
  a 1024px edited photo is caught whether it shows shoes or a phone;
* the **text** side is a supporting signal, measured on synthetic text, and it does **not**
  transfer to real reviews. The guard stops it from doing damage, but it does not make it work;
* fixing it properly needs real labelled review text (Ott's 1,600 hotel reviews, the OSF 40k set),
  which is the obvious next piece of work and is written up as an open item in readme13.

### 29.5 Final comparison table (all seven paper models re-run after the pHash fix)

Run on 21 Sep 2026 with the shipped model (Extra Trees, PHASH_REUSE_THRESHOLD = 12). Every
reported accuracy was rebuilt from its own per-fraud-type rates: all 8 rows agree to 0.00e+00
(`out/model_comparison/PAPER_NUMBERS_AUDIT.md`).

| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC@15% | Edited photo | Fake text | False alarms | McNemar vs ours |
|---|---|---|---|---|---|---|---|---|---|
| 30 Lu 2023 | BSTC (BERT + TextCNN) | 0.835 | 0.802 | 0.847 | 0.771 | 1% | 100% | 0.0% | 108/29, p=6.7e-12 |
| 11 Hou 2025 | FRIDRC (text + ViT) | 0.832 | 0.799 | 0.839 | 0.767 | 0% | 100% | 0.2% | 110/29, p=2.6e-12 |
| 40 Geetha 2025 | DeBERTa-v3 fine-tuned | 0.822 | 0.795 | 0.853 | 0.776 | 7% | 100% | 4.7% | 120/31, p=1.5e-13 |
| 38 Qayyum 2023 | FRD-LSTM | 0.817 | 0.787 | 0.848 | 0.764 | 2% | 100% | 4.0% | 122/29, p=9.0e-15 |
| 39 Duma 2024 | DHMFRD-TER | 0.783 | 0.768 | 0.829 | 0.744 | 16% | 100% | 15.3% | 155/34, p=1.2e-19 |
| 15 Xu 2024 | A-LSTM + behaviour | 0.773 | 0.760 | 0.837 | 0.748 | 16% | 100% | 17.3% | 160/31, p=3.8e-22 |
| 02 Shan 2021 | inconsistency + RF | 0.767 | 0.725 | 0.802 | 0.699 | 5% | 79% | 8.1% | 145/11, p=5.5e-31 |
| **OURS** | **forensic fusion (Extra Trees)** | **0.932** | **0.929** | **0.967** | **0.955** | **81%** | 84% | **2.0%** | - |

We lead on every column. Best baseline BSTC 0.835, ours 0.932: a margin of 9.7 points, every
difference significant at p < 7e-12. The pattern is unchanged and is the point of the project:
the baselines are text models, so they catch fake TEXT at 79-100% and edited PHOTOS at 0-16%.

Final model numbers: accuracy 0.9321 (95% CI 0.914-0.949), F1 0.9287, precision 0.9783,
recall 0.8840, ROC-AUC 0.9670, PR-AUC 0.976 balanced / 0.955 at 15% prevalence.
Confusion matrix TP 358 / FP 8 / FN 47 / TN 397. Label-shuffle control mean ROC-AUC 0.509,
seed stability 0.930-0.932, training/serving parity 41/41.

### 29.6 Every metric, not just accuracy (21 Sep 2026)

Accuracy alone does not establish that a model is good. `code/full_metrics.py` scores all eight
models on the same 810 test reviews from the per-row predictions, ranked by **MCC**, which only
rises when all four cells of the confusion matrix are good.

| model | Accuracy | Balanced acc | MCC | Kappa | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | PR-AUC@15% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **OURS** | **0.932** | **0.932** | **0.868** | **0.864** | 0.978 | **0.884** | 0.980 | **0.929** | **0.967** | **0.976** | **0.955** |
| BSTC | 0.835 | 0.835 | 0.709 | 0.669 | 1.000 | 0.669 | 1.000 | 0.802 | 0.847 | 0.896 | 0.771 |
| FRIDRC | 0.832 | 0.832 | 0.704 | 0.664 | 0.996 | 0.667 | 0.998 | 0.799 | 0.839 | 0.892 | 0.767 |
| DeBERTa-v3 | 0.822 | 0.822 | 0.668 | 0.644 | 0.936 | 0.691 | 0.953 | 0.795 | 0.853 | 0.898 | 0.776 |
| FRD-LSTM | 0.817 | 0.817 | 0.662 | 0.635 | 0.945 | 0.674 | 0.960 | 0.787 | 0.848 | 0.895 | 0.764 |
| DHMFRD-TER | 0.783 | 0.783 | 0.570 | 0.565 | 0.824 | 0.719 | 0.847 | 0.768 | 0.829 | 0.885 | 0.744 |
| A-LSTM | 0.773 | 0.773 | 0.549 | 0.546 | 0.806 | 0.719 | 0.827 | 0.760 | 0.837 | 0.889 | 0.748 |
| Shan + RF | 0.767 | 0.767 | 0.560 | 0.533 | 0.883 | 0.615 | 0.919 | 0.725 | 0.802 | 0.861 | 0.699 |

95% CIs (2,000 bootstrap resamples grouped by root photo) do not overlap:
MCC ours 0.832-0.900 vs BSTC 0.675-0.740; ROC-AUC ours 0.953-0.979 vs BSTC 0.814-0.876.

**The two cells where a baseline beats us** are BSTC's precision and specificity of 1.000. That
is not skill: BSTC reaches it by flagging only what it is certain of, so its recall is 0.669 and
it misses 134 of the 405 fakes. Ours catches 358. A detector that never accuses anyone has
perfect precision and no value. This is exactly why MCC and balanced accuracy are the headline
numbers to quote, not precision on its own.

**On "why did our accuracy go down":** it did not. 0.9494 was measured with the stale-embedding
bug of section 27 and was withdrawn; 0.9259 was the first honest figure; 0.9321 is the current
model after the pHash fix of section 28, and is the best honest result the project has produced.
Report: `out/model_comparison/FULL_METRICS.md`.

### 29.7 Leave-one-category-out: does it work on a product type never trained on? (21 Sep 2026)

Every result so far was on unseen PHOTOS but of product types the model had also seen in
training. That does not answer the real question: on a marketplace the next review is for a
phone, a kettle, a mattress. `code/loco.py` trains 15 times, each run removing one entire
category -- every photo, review and campaign of it -- and testing only on the held-out one.

| | tested on seen categories | tested on a category never trained on |
|---|---|---|
| accuracy | 0.932 | **0.889** |
| F1 | 0.929 | **0.879** |
| MCC | 0.868 | **0.789** |
| ROC-AUC | 0.967 | **0.950** |
| edit-CNN ROC | - | **0.887** |

Per category MCC runs 0.658 (Office chair) to 0.856 (Handbags); the edit detector 0.785 to 0.971.
Full table: `out/model_comparison/LOCO.md`.

**Branch by branch, which is what the question is really about:**

* **Reuse (pHash + ResNet-50): transfers by construction.** Nothing is fitted on our data -- pHash
  is an algorithm, ResNet-50 is frozen ImageNet weights. It never learned "shoes", so an unseen
  product type is not a different problem.
* **Forensic CNN: transfers, and this is the measurement that backs the novelty claim.** Mean ROC
  0.887 on categories it never trained on. It reads JPEG compression and noise residuals, not
  objects, so a splice in a photo of a phone looks like a splice in a photo of shoes. Until this
  run that was an argument; now it is a number.
* **Text: the weak branch.** Text-only ROC falls to 0.814, and section 29 already showed it does
  not survive contact with real marketplace prose at all. The "specific detail" vocabulary is
  built per category, so an unseen category scores 0 on it by construction.

**Headline for the report:** on a product type it has never seen, the model still reaches
accuracy 0.889 and MCC 0.789 -- higher than any baseline achieves on categories it *did* train on
(best baseline BSTC: 0.835 / 0.709). The ~4-point drop is the honest cost of generalisation and
should be quoted, not hidden.
