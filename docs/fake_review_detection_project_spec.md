# Fake Review Detection via Visual Forensics — Complete Project Specification

**Version:** 1.0
**Last updated:** 2026-09-08
**Course:** Computer Vision (BTech, AI & Data Science)
**Status:** Dataset construction phase — root image pool complete, generated review dataset pending

---

## 0. How to use this document

This is the single source of truth for the project. It is written so that a
person or an AI assistant who has never seen this project before can read it
top to bottom and then do useful work without asking for background.

If you are an AI assistant reading this at the start of a new conversation:
everything you need is here. Sections 1–3 tell you what is being built and why.
Section 4 is the dataset, which is the part most likely to be misunderstood.
Sections 5–7 are the system design. Section 11 lists what is genuinely undecided
— do not invent answers to those, ask.

Terminology note used throughout: **"the extension"** is ambiguous in this
project and has caused real confusion. It always means one of two things, and
this document always says which:
- **research extension** — the novel contribution beyond the base paper
- **browser extension** — the Chrome plugin deliverable

---

## 1. One-paragraph summary

We are building a multimodal fake-review detector for e-commerce product reviews
in which the **attached review photograph is treated as forensic evidence**
rather than as a similarity feature. The system extracts three new visual signals
— image manipulation, platform-wide image reuse/provenance, and visual
coordination structure across reviewer accounts — and fuses them with text,
rating, and behavioural signals into a single risk score with a three-way output
(Genuine / Needs Human Verification / Fake). It extends He et al. (2022, PNAS),
which used review images only as a within-product similarity feature and reported
that network features outperformed image features. Our research extension also
tests something the base paper never did: whether the detector generalises to
product categories it has never seen during training. The final deliverable
includes a Chrome browser extension that scores reviews live on a product page.

---

## 2. Problem statement

Paid and fabricated product reviews distort purchasing decisions at scale.
Existing detection work falls into three groups:

1. **Text-based** — linguistic deception cues, BERT/RoBERTa classifiers. Strong
   within a domain, but the cues are domain-specific: fake reviews about
   headphones use different vocabulary from fake reviews about shoes.
2. **Behavioural / graph-based** — reviewer posting bursts, rating skew,
   reviewer–product bipartite graph structure. Currently the strongest family.
3. **Multimodal** — a small number of works add review images, almost always as
   an extra feature block fed alongside text.

The gap: **no significant work treats the review image as evidence with its own
provenance and integrity.** Images are embedded and compared, but nobody asks
whether the image was edited, whether it has appeared elsewhere on the platform
under a different account, or whether shared imagery links accounts into a
coordinated cluster. That gap is what this project fills, and it is a genuine
computer vision problem, which matters because this is a CV course.

---

## 3. Base paper and novelty

### 3.1 Base paper

> He, S., Hollenbeck, B., Overgoor, G., Proserpio, D., & Tosyali, A. (2022).
> *Detecting Fake-Review Buyers Using Network Structure: Direct Evidence from
> Amazon.* Proceedings of the National Academy of Sciences, 119(47).

Chosen from a survey of ~40 papers for three reasons:

- **Label quality is the strongest available.** The authors directly observed
  sellers recruiting reviewers through Facebook groups, so the fake/genuine
  label is tied to documented purchasing of reviews — not a platform's opaque
  filter (Yelp) and not lab-written fakes (Ott et al.).
- **It already uses review images**, which makes it the closest paper to extend
  rather than contradict.
- **It reports that network features beat image features.** That published
  weakness is precisely our opening.

### 3.2 What the base paper does

| Component | Base paper's approach |
|---|---|
| Labels | Direct evidence — Facebook recruitment posts tie sellers/reviews to paid campaigns |
| Network signal | Bipartite reviewer–product graph; products sharing unusually many reviewers flagged as clustered |
| Image signal | Pretrained CNN embeddings → cosine similarity between review–review and review–product images, aggregated (min/max/mean/SD) **within a single product only** |
| Text / behaviour | Rating skew, helpfulness votes, timing gaps as metadata features |
| Classifier | Random Forest (supervised); Gaussian-mixture clustering (unsupervised) |
| Evaluation | F1, feature-importance comparison across network vs image vs text feature sets |

Their image module answers exactly one question: *do this product's images look
similar to each other?* It never asks whether an image was manipulated, whether
it has been reused elsewhere on the platform, or whether visual similarity links
different reviewers into a coordination pattern.

### 3.3 Our novelty, stated precisely

**Base paper:** review images are a similarity-based feature block, scoped to a
single product, fed to a classifier alongside network and text features.

**This project:** review images are forensic evidence, generating three new
signal types —

1. **Manipulation / authenticity evidence** (absent from the base paper entirely)
2. **Cross-product and cross-reviewer provenance/reuse**, not confined to one
   product
3. **A visual coordination graph** built from image-similarity edges layered onto
   the reviewer–product graph

— then fuses these with text and behaviour through an evidence-aware model that
can **abstain** on uncertain cases.

**One sentence for a professor or viva:** *"He et al. use image similarity as a
feature within a product; we use image similarity, reuse, and manipulation
evidence as a forensic signal across the entire reviewer–product ecosystem,
fused with an uncertainty-aware decision layer."*

### 3.4 Honest novelty boundary

State these limits before an examiner finds them:

- Consistency/inconsistency checking in fake-review detection is not new. Prior
  work has checked rating-vs-sentiment inconsistency **within a single review**.
  Our contribution widens that to inconsistency **across review, reviewer,
  product, and time**.
- Perceptual hashing, ELA, CLIP embeddings and Louvain community detection are
  all standard, off-the-shelf techniques. **The novelty is the composition and
  the application**, not the invention of any component.
- The generated dataset is a controlled evaluation, not real-world ground truth
  (see §4.7).

### 3.5 Research extension (the second contribution)

He et al. evaluate only within a single domain (Amazon) and never test
generalisation. Our research extension:

> We explicitly build and test the detector for its ability to generalise across
> **product categories never seen during training**, and show that image-forensic
> signals degrade less than text signals do.

The reasoning: text features are domain-specific; image-forensic features
("was this photo edited, reused, or shared across suspicious accounts") are
largely independent of what the product is. This is *why* the image-first design
is a generalisation strategy and not just a stylistic choice for a CV course.

Tested by **leave-one-category-out (LOCO)** validation — see §8.3. Note that a
sharp drop on unseen categories is still a valid, reportable result; it is
honest data, not a failure.

---

## 4. Dataset — full specification

This section has caused the most confusion in the project. Read it carefully.
The dataset has **two layers** and they are not the same thing.

### 4.1 Layer 1 — root image pool (COMPLETE)

**Location:** https://github.com/Venkatasai200628/fake_review_detection

**Contents:** 15 product-category folders, each holding roughly 13–14 real
product photographs.

Categories: Bluetooth speaker, Handbag, Knifes, Office chair, Phone case, Shoes,
Sunglasses, Watches, Water bottle, Wireless_Earbuds, Yoga mat, backpack, mouse,
skincare set, suitcases.

**Total:** ~195–197 images.
**Source:** Unsplash / Pexels (properly licensed for this use).
**Formats:** JPEG, WEBP, AVIF (PIL reads all three without conversion).
**Resolutions:** ~600 px to ~6000 px on the long edge.
**Metadata present:** none. No CSV, no labels, no README, no pairing information.

**What this is:** a clean, category-tagged pool of *root images*. It is raw
material.

**What this is not:** the dataset. It has no reviews, no text, no labels, no
reviewers. Anyone who calls this "the dataset" will misunderstand the project.

**Why 15 unrelated categories matters:** it is what makes the LOCO
generalisation experiment (§8.3) possible. This structure is an asset, not an
accident.

### 4.2 Layer 2 — generated review dataset (PENDING)

Produced by three Python scripts (`build_dataset.py`, `gen_images.py`,
`gen_text.py`) that consume Layer 1 and emit a full review-level dataset.

**Target size:** ~500 review rows, ~540 images.
**Approximate composition:** ~56% genuine, ~40% fraud, ~4% borderline, spread
across ~12 coordinated fraud campaigns.

**Core generation principle:** each real photo from Layer 1 becomes a **root
image**. Individual reviews receive *different transform combinations* of a root
(crop, brightness, angle, JPEG recompression). This mirrors how real listings
actually work — a handful of true product photos and many customer re-uploads of
near-identical shots. This is why ~195 photos legitimately support ~500 review
rows without any imbalance being introduced.

### 4.3 Fraud type taxonomy

Text is generated **conditioned on** the image's ground-truth fraud type, so the
two modalities are coherent per review. Four types:

| fraud_type | Image tell | Text tell | Purpose |
|---|---|---|---|
| `genuine` | none | none | negative class |
| `visual_manipulation` | spliced / copy-moved / noise-injected | none — text sounds authentic on purpose | proves the image module earns its place |
| `text_deception` | none — clean unique real photo | generic superlatives, no concrete detail | proves text is still needed; image-only cannot catch this |
| `coordinated_reuse` | same root photo, differently transformed per account | shared template phrasing (simulated "talking points") | the campaign case; needs image + timing + text jointly |

**Rows 2 and 3 are the entire justification for building a fusion model.** If
every fake had both a fake image and fake text, the two signals would be
perfectly correlated, a text-only baseline would match the fusion model, and
there would be no result. Keep all four types populated.

### 4.4 Generation rules (each tied to documented fraud literature)

- **Rating skew** — fake reviews cluster at 1★ or 5★, rarely 2–3★. Matches the
  incentivized-review literature the base paper cites.
- **Burst timing for campaigns** — a reused image appears across unrelated
  accounts within a ~3-hour window. Directly mirrors He et al.'s finding that
  sellers recruit via Facebook groups, so paid reviews cluster in time.
- **Genuine reuse is spread over weeks** — deliberately the opposite temporal
  pattern, so the model must learn *reuse + timing jointly*, never reuse alone.
- **Text specificity gap** — genuine reviews receive randomised concrete details
  (a specific part failing after a specific duration); fake reviews receive
  generic superlatives; campaign reviews receive near-duplicate templated
  phrasing.
- **Manipulation ground truth** — actual pixel-level splicing, copy-move, and
  noise operations followed by a real JPEG recompression pass, so ELA and pHash
  detect something real rather than reading a label.

### 4.5 Deliberate negative controls

Built in on purpose so the provenance module never learns "any reuse = fraud":

- **Legitimate stock-photo reuse** — the manufacturer's listing photo re-uploaded
  by several genuine customers. Labelled Genuine. Must NOT trigger fraud.
- **Light-recompression-only case** — an image that has merely been re-saved.
  Must score *below* the fraud threshold (0.25).
- **Two-account ambiguous reuse pair** — a weak signal, not a full campaign.
  Labelled Needs Verification.

If your model flags these, the provenance module has learned a shortcut. Check
them explicitly.

### 4.6 Schema

One CSV row per review, plus an image folder.

| Column | Type | Notes |
|---|---|---|
| `review_id` | str | primary key |
| `product_id` | str | |
| `category` | str | one of the 15 folder names |
| `reviewer_id` | str | synthetic account identifier |
| `review_text` | str | conditioned on fraud_type |
| `rating` | int | 1–5, skewed per fraud_type |
| `timestamp` | datetime | burst vs spread per fraud_type |
| `image_file` | str | path into the derived image folder |
| `root_image_id` | str | **critical** — which Layer 1 photo this derives from |
| `transform_chain` | str | ordered list of ops applied |
| `manipulation_score` | float | ground-truth [0,1] |
| `fraud_type` | enum | genuine / visual_manipulation / text_deception / coordinated_reuse |
| `campaign_id` | str/null | non-null for coordinated_reuse members |
| `label` | enum | Genuine / Needs_Verification / Fake |

`root_image_id` is load-bearing for the split logic in §4.9. Do not drop it.

### 4.7 Why synthetic data, and what it does and does not prove

**Why:** the novel signals — coordinated image reuse and image manipulation — do
not exist as labelled data anywhere. No public dataset labels review images for
manipulation or cross-account reuse. Constructing controlled examples is
standard practice in multimodal-consistency work, not a fallback.

**What it proves:** that each module functions, that the fusion layer combines
them correctly, and that the design behaves as intended on cases with known
ground truth.

**What it does not prove:** real-world detection performance. State this
explicitly in the report and the viva.

**A specific circularity to disclose:** the same code that applies the transforms
is what pHash then detects. pHash will score near-perfectly on the provenance
task, and that number means very little. Report it as a controlled sanity check
that the module works, not as a performance claim. Raising this yourself is
evidence you understand your own method.

Similar caution for ELA: the Layer 1 stock photos have already been JPEG
compressed several times before we touch them, so ELA maps will be noisier than
textbook examples.

### 4.8 Class balance policy — RESOLVED

This question generated a false contradiction earlier in the project. The
resolution, which two independent analyses agreed on:

- **Training set: roughly balanced.** Anything from 50/50 to ~56/40 is fine. At
  real-world prevalence (5–15% fake) there would be too few fraud examples to
  learn from, so fraud is deliberately over-sampled. This is intentional and
  correct.
- **Evaluation set: a separate CSV at realistic prevalence** (~15% fake), built
  by subsampling fraud rows from the same pool. All headline metrics are
  reported on this set.
- **Never report plain accuracy on the realistic set.** At 15% prevalence a
  model that always predicts "genuine" scores 85%. Report precision, recall, F1,
  PR-AUC, and the chosen operating threshold.

Deliverables: `train.csv` and `holdout_realistic.csv`.

### 4.9 Split policy — CRITICAL

Two constraints, applied together:

1. **Split by `root_image_id`.** Every derived image of a held-out root stays
   held out. If transforms of the same root photo appear in both train and test,
   the model simply recognises the photo and reports a meaningless ~98%.
2. **Split by `category`.** Required for the LOCO generalisation experiment.

Never use a random row split. This is the single easiest way to invalidate the
entire results section.

### 4.10 Real datasets — considered, and disposition

Recorded so the choice is defensible and nobody re-litigates it.

| Dataset | Contents | Disposition |
|---|---|---|
| **YelpChi / YelpNYC / YelpZip** (Rayana & Akoglu, 2015/2016) | ~608k reviews, 5,044 restaurants, 260k reviewers; text, ratings, timestamps, reviewer IDs, filter-derived labels. Via Stony Brook ODDS. | **Deferred.** The standard benchmark and directly comparable to prior work, but has no review images, so it cannot exercise any of our three novel modules. Optional use for the text/behaviour baseline only. |
| **Yelp Open Dataset** | Real review photos, real reviews | **Rejected for primary use.** Has photos but no manipulation, reuse, or consistency labels. Would require constructing our own weak labels anyway — i.e. the same synthetic step, with less control. |
| **Ott et al. Deceptive Opinion Spam Corpus** | Hotel reviews, gold-standard human-written truthful vs deceptive | **Optional.** Good for a text-only baseline sanity check. No images. |
| **Amazon Reviews (McAuley et al.) / Kaggle CG-OR fake reviews** | Large-scale Amazon review text | **Optional.** Useful as a second text domain to support the generalisation claim. No usable image labels. |
| **FRIDRC** | Fake review dataset | **Blocked/uncertain.** Access was never confirmed. Do not plan around it. If access arrives, revisit §4.10. |
| **He et al. (2022) data** | The base paper's own Amazon data | **Not obtainable** for this project. This is the reason synthetic construction was chosen. |

**Decision:** synthetic Layer 2 built on real Layer 1 photographs is the primary
dataset. Public text corpora may be added later as supplementary baselines. This
decision is final unless FRIDRC access is confirmed.

---

## 5. System architecture

Pipeline, in order. Each module emits features consumed by the fusion layer (M6).

```
review (text, rating, timestamp, reviewer_id, product_id, image)
   │
   ├─ M1  Image representation      → embedding e_i ∈ R^d
   ├─ M2  Similarity                → cosine sim(e_i, e_j)
   ├─ M3  Provenance / reuse        → pHash + ANN search → provenance score
   ├─ M4  Manipulation              → ELA (+ optional CNN) → manipulation score m_i
   ├─ M5  Visual coordination graph → clustering coeff, community features
   ├─ MT  Text module               → text deception score
   ├─ MB  Behaviour module          → rating skew, burst timing features
   │
   └─ M6  Evidence fusion → risk score p ∈ [0,1] → 3-way decision + explanation
```

### M1 — Image representation
Pretrained CLIP or ViT image encoder → embedding `e_i ∈ R^d` per image.
The base paper used a plain CNN; upgrading the backbone is an improvement but
**is not itself the novelty**. Do not claim it as one.

*Alternatives considered:* ResNet-50 (weaker semantic separation), training from
scratch (impossible at ~195 root images).

### M2 — Similarity (shared building block, same as base paper)

    sim(e_i, e_j) = (e_i · e_j) / (‖e_i‖ ‖e_j‖)

### M3 — Provenance / reuse (NOVEL)
- **Perceptual hash** (pHash, DCT-based) per image → Hamming distance
  `d_H(h_i, h_j)` for fast near-duplicate detection.
- **Approximate nearest-neighbour search** (FAISS, HNSW or IVF index) over all
  embeddings **platform-wide**, not per-product as in the base paper. This is
  what catches the same image reused across different products or accounts.

*Observed separation on generated data:* pHash Hamming distance 0–16 for reused
images vs 30–36 for unrelated. Clean, but see the circularity caveat in §4.7.

*Alternatives considered:* dHash/aHash (faster, less robust to rotation); exact
kNN (fine at this dataset size, but FAISS demonstrates the platform-wide claim
scales).

### M4 — Manipulation / authenticity (NOVEL — absent from base paper)
- **Error Level Analysis**: recompress the image at fixed JPEG quality, take the
  absolute pixel difference map. Cheap, classical, appropriate for a course
  project.
- **Optional**: a CNN-based synthetic/GAN-image detector (e.g. ResNet-50
  fine-tuned on real-vs-generated) producing `m_i ∈ [0,1]`.

Fraud threshold on `m_i`: **0.25** (light-recompression control must sit below
this).

### M5 — Visual coordination graph (NOVEL — extends the base paper's network idea into vision)

`G = (V, E)` where `V` = reviewers ∪ products ∪ images.
An edge is added when `sim(e_i, e_j) > τ` **or** `d_H(h_i, h_j) < δ`.

- Clustering coefficient: `C_v = 2T_v / (k_v(k_v − 1))`
- Community detection via modularity maximisation (Louvain):
  `Q = (1/2m) Σ_ij [A_ij − k_i k_j / 2m] δ(c_i, c_j)`

*Alternatives considered:* a GNN over the graph. Rejected for now — more
parameters than ~500 rows can support, and it would obscure which signal is
doing the work. Explicit graph features keep the ablation interpretable. Worth
listing as future work.

### MT — Text module
Text deception score from the review text. A transformer sentence encoder or a
fine-tuned classifier both work. **Text is not optional** — see §6.

### MB — Behaviour module
Rating skew, inter-post timing, per-reviewer burst statistics.

### M6 — Evidence fusion (NOVEL — the base paper never combines image evidence beyond feature concatenation)

Feature vector
`x = [text score, rating/behaviour features, manipulation score, provenance score, graph features]`

Lightweight MLP:

    p = σ(W₂ · ReLU(W₁x + b₁) + b₂)

trained with binary cross-entropy:

    L = −[y log p + (1 − y) log(1 − p)]

Plus a **Random Forest on the identical feature vector** as a direct, fair
comparison back to the base paper's own classifier.

---

## 6. Role of text — explicit, because this has been misread

The image side carries the *novelty*. The system is *multimodal*. These are
different statements.

Text remains fully present in:
- every review row (§4.3)
- an entire fraud class (`text_deception`), where the image is clean and the text
  is the only tell — image-only models cannot catch this
- the fusion feature vector (first entry)
- the required baselines and ablations (§8.2)
- the generalisation claim itself: the research extension asserts image-forensic
  signals degrade *less than text signals do* across unseen categories. That
  claim is unmeasurable without a text module.

**Framing for the professor:** "Our contribution is on the visual-forensics side,
but the system is multimodal — text, rating and behaviour are fused with the
image evidence, and we report per-module ablations."

---

## 7. Inputs and outputs

**Input per review:** text, rating, timestamp, reviewer_id, product_id, one or
more review images, product images.

**Output:** risk score `p ∈ [0,1]`, mapped to a three-way decision using tuned
thresholds:

| Band | Decision |
|---|---|
| low | Genuine |
| middle | **Needs Human Verification** (abstain) |
| high | Fake |

Plus an **evidence explanation** naming which module drove the score
(manipulation / provenance / coordination / text), and supporting visuals:
nearest-neighbour matches, ELA heatmap, coordination subgraph.

The abstain band is part of the novelty ("evidence-aware model that can abstain")
and is also what makes the browser extension demo credible.

---

## 8. Evaluation protocol

### 8.1 Metrics

    Precision = TP/(TP+FP)      Recall = TP/(TP+FN)
    F1 = 2PR/(P+R)              FPR = FP/(FP+TN)      FNR = FN/(FN+TP)

Plus ROC-AUC, PR-AUC, and the confusion matrix. Report the chosen operating
threshold explicitly. On the realistic-prevalence holdout, **PR-AUC is the
headline number, not accuracy.**

### 8.2 Ablations (required)

Run all of these on identical splits:

| Configuration | Purpose |
|---|---|
| text-only | baseline; must be beaten |
| image-only | baseline; must be beaten |
| network/graph-only | reproduces the base paper's strongest family |
| base-paper-equivalent (within-product image similarity + text + network) | the thing we claim to improve on |
| full fusion (ours) | the claim |
| full fusion, Random Forest | fair comparison to base paper's classifier |

Also report **per-fraud-type breakdown**. A table showing that image-only fails
on `text_deception` and text-only fails on `visual_manipulation`, while fusion
catches both, is the single most persuasive result available to this project.

### 8.3 Leave-one-category-out (the research extension result)

Train on 11–12 of the 15 categories; test on 3–4 entirely unseen. Report:

| | Seen categories | Unseen categories | Δ |
|---|---|---|---|
| text-only | | | |
| image-forensic-only | | | |
| full fusion | | | |

The hypothesis is that the Δ for image-forensic is smaller than the Δ for text.
A sharp drop is still a reportable, honest result.

### 8.4 Negative-control check

Verify the three controls from §4.5 behave correctly. Report them.

---

## 9. Browser extension (the second deliverable)

The professor's request for "an extension" appears to cover both the research
extension (§3.5) and a working Chrome plugin. **Confirm which he means — possibly
both.** Build this last; it is worthless without a trained model and takes roughly
a weekend once one exists.

**Architecture (Manifest V3):**

1. **Content script** — reads review blocks from an Amazon or Flipkart product
   page DOM: review text, rating, reviewer name, image URLs.
2. **Local backend** — FastAPI on localhost, loads the trained fusion model,
   returns `{risk_score, decision, driving_module}` per review.
3. **Injection** — draws a badge per review card (green / amber "needs
   verification" / red) and a summary bar at the top of the page.

**Notes:** keep it read-only and local; do not log or redistribute scraped data.
The three-way output is what makes the demo convincing — an amber "needs human
verification" badge reads as far more credible than a binary verdict.

---

## 10. Build order and status

| # | Step | Status |
|---|---|---|
| 1 | Literature survey (~40 papers) | done |
| 2 | Base paper selected (He et al. 2022) | done |
| 3 | Novelty defined against base paper | done |
| 4 | Root image pool collected (Layer 1) | done |
| 5 | Beamer slide deck (intro/problem/novelty/objectives/dataset/proposed work/conclusion; literature review table left blank to fill) | done |
| 6 | Generator scripts written (`build_dataset.py`, `gen_images.py`, `gen_text.py`) | written; **rebuild against real Layer 1 photos was in progress and unfinished** |
| 7 | Generate `train.csv` + `holdout_realistic.csv` + derived images | **NEXT** |
| 8 | Verify splits (root_image_id + category, no leakage) | pending |
| 9 | Swap placeholder embedding for real CLIP/ViT — one function, `gen_images.compute_embedding()` | pending |
| 10 | Baselines: text-only, image-only, graph-only | pending |
| 11 | Fusion MLP + Random Forest, full ablations | pending |
| 12 | LOCO generalisation run | pending |
| 13 | Threshold tuning + abstain band calibration | pending |
| 14 | Browser extension | pending |
| 15 | Final report + viva prep | pending |

**Immediate blocker:** the generator scripts were delivered as a zip in a prior
session and the rebuild against the real photo folders never completed. Either
recover that zip or rebuild the scripts from this specification — §4.2 through
§4.6 contain every design decision needed to do so.

---

## 11. Open decisions — do not invent answers to these

1. Does the professor want the research extension, the browser extension, or
   both?
2. Is the generator zip recoverable, or do the scripts need rebuilding?
3. FRIDRC access — confirmed or abandoned?
4. Final dataset size: stay at ~500 rows, or scale up (the scripts expose count
   constants `N_GENUINE`, `N_CAMPAIGNS` near the top)?
5. Which 3–4 categories are held out for LOCO?
6. Is a text transformer (fine-tuned) in scope, or is a lighter text scorer
   sufficient given compute?

---

## 12. Known limitations (state these before an examiner does)

- Synthetic Layer 2 is a controlled evaluation, not real-world ground truth.
- Provenance ground truth is circular: the transform code and the detector are
  matched (§4.7).
- ELA on already-multiply-compressed stock photos is noisy.
- ~500 rows from ~195 root images is small. Pretrained encoders are frozen and
  only the fusion head is trained; results carry wide variance. Use k-fold
  cross-validation, not a single split.
- No real reviewer behaviour data — behavioural signals are simulated from
  documented patterns, not observed.
- Component techniques are all standard; the contribution is composition and
  application.

---

## 13. Glossary

- **Root image** — one of the ~195 real photos in Layer 1; multiple review
  images derive from it via transforms.
- **pHash** — perceptual hash; DCT-based image fingerprint compared by Hamming
  distance.
- **ELA** — Error Level Analysis; JPEG recompression difference map used to
  surface edited regions.
- **LOCO** — leave-one-category-out validation.
- **Abstain band** — the middle risk range mapped to "Needs Human Verification".
- **Negative control** — a case deliberately built to look suspicious but
  labelled genuine, to catch shortcut learning.
