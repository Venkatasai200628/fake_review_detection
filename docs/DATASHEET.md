# Dataset Datasheet — Layer 2 Generated Review Dataset

**Version:** v2 (rating/text polarity fix applied)
**Generated:** 2026-09-08
**Rows:** 503 reviews | **Images:** 503 | **Roots used:** 183 of 197 | **Categories:** 15
**Companion document:** `PROJECT_SPEC.md` (project design); this file documents the data only.

---

## 1. What this dataset is, and what it is not

**It is** a controlled, fully-labelled multimodal review dataset built to exercise
three detection modules — image manipulation, image provenance/reuse, and
coordination structure — alongside text and behavioural signals.

**It is not** real-world data and does not measure real-world performance. Every
number produced on it is a statement about whether the modules function as
designed, not about how well they would work on Amazon.

### Critical point for the viva

**No image in this dataset is AI-generated.** All 503 images derive from your 197
real Unsplash/Pexels photographs. "Fake" labels the *review's status*, never the
image's origin.

This is deliberate and it is the most important design decision in the dataset.
If fraudulent rows used AI-generated images while genuine rows used real photos,
any classifier would learn *"which tool produced this image"* and report ~97%
accuracy while proving nothing about fake-review detection. Because both classes
derive from the same photo pool, the only thing separating them is the fraud
signal itself.

---

## 2. The two layers

| | Layer 1 | Layer 2 |
|---|---|---|
| What | Root photo pool | Generated review dataset |
| Where | `github.com/Venkatasai200628/fake_review_detection` | this dataset |
| Contents | 197 real product photos, 15 category folders, ~13 each | 503 review rows + 503 derived images |
| Labels | none | full ground truth |
| Status | collected manually | produced by `build_dataset.py` |

Each Layer 1 photo is a **root image**. Multiple review rows derive from one root
via different transform chains. This mirrors real e-commerce listings: a handful
of true product photographs and many customer re-uploads of near-identical shots.
It is also why 197 photos legitimately support 503 review rows with no imbalance
introduced — row counts are controlled independently of photo counts.

---

## 3. Column dictionary

Every column in `reviews_full.csv`, `train.csv`, `holdout_realistic.csv`:

| Column | Type | Description |
|---|---|---|
| `review_id` | str | Primary key, `R00001`… |
| `product_id` | str | Product identity, derived from category |
| `category` | str | One of the 15 Layer 1 folder names |
| `reviewer_id` | str | Account identifier. `U#####` = ordinary population, `C#####` = campaign population |
| `review_text` | str | Generated conditioned on `fraud_type` (§6) |
| `rating` | int | 1–5, consistent with text polarity (§7) |
| `timestamp` | ISO datetime | Posting time; burst vs spread per `fraud_type` (§8) |
| `image_file` | str | Relative path into `images/` |
| `image_id` | str | `IMG00001`… unique per derived image |
| `root_image_id` | str | **Which Layer 1 photo this derives from.** Load-bearing for the split rule (§10) |
| `transform_chain` | str | Ordered `+`-joined list of ops applied (§5) |
| `manipulation_score` | float | Ground-truth manipulation strength in [0,1] |
| `phash` | str | 64-bit perceptual hash, hex |
| `embedding` | JSON list | 64-d image descriptor. **Placeholder** — see §11 |
| `fraud_type` | enum | Mechanism of fraud, 6 values (§4.2) |
| `campaign_id` | str | `CAMP001`… for coordinated members, empty otherwise |
| `label` | enum | Target variable, 3 values (§4.1) |
| `notes` | str | Free text; `NEGATIVE_CONTROL_*` marks deliberate traps (§9) |

---

## 4. Labels — the full explanation

There are **two** labelling columns and they answer different questions. This
distinction is the thing most likely to confuse someone reading the CSV cold.

### 4.1 `label` — the target variable (what the model predicts)

Three values, not two.

| `label` | Count | Meaning |
|---|---|---|
| `Genuine` | 293 | A real customer review. Model should pass it. |
| `Fake` | 198 | A fraudulent review by any mechanism. Model should flag it. |
| `Needs_Verification` | 12 | Genuinely ambiguous. Model should abstain. |

**Why three and not two.** The project's stated novelty includes "an
evidence-aware model that can abstain on uncertain cases." A binary label cannot
express abstention, so the ground truth has to carry the third state. Real fraud
detection works this way — a platform routes borderline cases to human review
rather than auto-deleting them. The three-way output also makes the browser
extension demo far more credible: an amber "needs human verification" badge reads
as a considered judgement, a red/green binary reads as overconfidence.

**Why the counts are what they are.** ~58% genuine / ~39% fake / ~2% borderline
is a *training-oriented* mix, deliberately over-representing fraud. At real-world
prevalence (5–15% fake) there would be too few fraud examples to learn from. The
realistic ratio is restored in the evaluation split instead (§10).

### 4.2 `fraud_type` — the mechanism (what kind of fraud, and where the tell is)

Six values. This is **not** the prediction target. It exists so results can be
broken down by mechanism, which is where the project's argument actually lives.

| `fraud_type` | Count | `label` | Image tell | Text tell |
|---|---|---|---|---|
| `genuine` | 293 | Genuine | none | none |
| `visual_manipulation` | 70 | Fake | edited pixels | **none — text sounds real** |
| `text_deception` | 70 | Fake | **none — clean real photo** | generic superlatives |
| `coordinated_reuse` | 63 | Fake | same root across accounts | shared templates |
| `borderline_recompress` | 6 | Needs_Verification | re-saved only | none |
| `borderline_ambiguous_reuse` | 6 | Needs_Verification | 2-account reuse only | none |

**Why `fraud_type` must exist as a separate column.** Your headline result is not
one accuracy number, it is this table:

|  | catches `visual_manipulation` | catches `text_deception` |
|---|---|---|
| text-only baseline | ✗ expected to fail | ✓ |
| image-only baseline | ✓ | ✗ expected to fail |
| fusion (ours) | ✓ | ✓ |

Without `fraud_type` you cannot compute that breakdown, and without that
breakdown you have no evidence that fusion is worth building. The two failure
cells are the entire justification for the project.

**Why `visual_manipulation` has authentic-sounding text.** Deliberate. If
manipulated images also carried fake-sounding text, the two signals would
correlate perfectly, a text-only model would match the fusion model, and there
would be no result. The text is written to look real *specifically so that only
the image module can catch it*.

**Why `text_deception` has a clean unmanipulated photo.** The mirror of the
above, for the same reason.

### 4.3 Mapping between the two

`fraud_type` → `label` is deterministic and one-directional. Never train on
`fraud_type`; it encodes the answer. Use it only for reporting.

---

## 5. Image generation — exact procedure

Two independent layers of operation.

### 5.1 Benign transforms (applied to every row)

Simulates a different customer uploading a near-identical shot. Two to four ops
chosen at random per image, in random order:

| Op | Parameter range | What it simulates |
|---|---|---|
| `crop` | 90–99% of frame, random offset | slightly different framing |
| `rotate` | ±3° bicubic, no expand | handheld camera angle |
| `brightness` | ×0.82–1.18 | different lighting conditions |
| `contrast` | ×0.85–1.20 | different device or display profile |
| `saturation` | ×0.80–1.20 | camera colour processing |
| `blur` | Gaussian σ 0.2–0.7 | slight focus miss |
| `recompress` | JPEG quality 55–80 | platform upload re-encoding |

**Why these ranges are mild — measured, not assumed.** The first build used ±7°
rotation and crops down to 82%. Validation showed reused and unrelated images
overlapping in pHash space (same-root p95 = 26 vs different-root p5 = 24, margin
−2): the provenance signal had collapsed into the noise floor. pHash is not
rotation-invariant. Softening to ±3° and 90% restored a clean margin of +6.

This is worth carrying into your limitations section as a concrete finding: **a
determined evader who rotates hard defeats pHash outright.** It is also the
argument for keeping embedding-based similarity alongside pHash, since cosine
similarity degraded far less under the harsh transforms than the hash did.

### 5.2 Manipulations (applied only to `visual_manipulation` rows)

Real pixel edits, each followed by a genuine JPEG recompression pass so that ELA
and forensic checks detect something real rather than reading a label.

| Op | Extent | What it simulates |
|---|---|---|
| `copy_move` | 12–22% patch duplicated within the same image | cloning clean area over a defect |
| `splice` | 15–28% region pasted in from a *different* product photo | compositing a fake product shot |
| `noise` | Gaussian σ18 burst into a 25% region | inconsistent noise floor from editing |

`manipulation_score` by group:

| `fraud_type` | mean | range |
|---|---|---|
| `genuine` | 0.06 | 0.01–0.12 |
| `visual_manipulation` | 0.82 | 0.66–0.96 |
| `text_deception` | 0.07 | 0.01–0.12 |
| `coordinated_reuse` | 0.08 | 0.02–0.15 |
| `borderline_recompress` | 0.13 | 0.07–0.18 |

Fraud threshold: **0.25**. The borderline group sits below it by design.

### 5.3 Coordinated reuse — a different mechanism

Not a manipulation. One root photo is selected, then each campaign member
receives a *different* benign transform chain of it. Six accounts therefore post
six visually distinct images that all trace to one source. This is precisely what
the provenance module exists to detect, and it is the signal the base paper
cannot see because it compares images only within a single product.

Output images: JPEG, long edge 512 px, quality 88.

---

## 6. Text generation — exact procedure and basis

Text is **never generated independently of the image.** It is conditioned on the
row's `fraud_type` so that the two modalities tell a coherent story.

### 6.1 `genuine`

Templates filled from a per-category vocabulary of real components, usage
situations, and failure modes. Fifteen category vocabularies, each with ~5 parts,
~4 situations, ~4 failure modes.

> [4★] *"Does the job at the gym. Not perfect, a strap pin sheared eventually, but for the price it is acceptable."*

> [2★] *"Worked fine at first but after about eight months the seat foam flattened. The lumbar support was the weak point."*

**Basis:** the *specificity gap* is a documented spam signal. Genuine reviewers
name a specific part, a specific duration, and a concrete situation. Sentiment is
split 62% positive / 22% negative / 16% mixed, so "genuine" is never a synonym
for "positive" — otherwise sentiment alone would separate the classes.

### 6.2 `visual_manipulation`

Uses the **genuine positive** generator unchanged.

> [5★] *"Using it during workouts for about eight months. The mic does exactly what it needs to, nothing fancy but it works."*

**Basis:** the image is the only tell. Any text-only model must fail on this
group, and that failure is a required result, not a defect.

### 6.3 `text_deception`

Generic superlatives with nothing verifiable — no part, no duration, no
situation.

> [5★] *"Best purchase ever. Perfect in every way. Five stars, will buy again!"*

> [1★] *"Terrible product. Complete waste of money. Do not buy this. Very bad quality."*

**Basis:** the specificity gap inverted. 85% positive, 15% negative (fake reviews
are sometimes competitor attacks, not only paid praise). The image is a clean
unmanipulated real photo, so any image-only model must fail here.

### 6.4 `coordinated_reuse`

Each campaign draws from **3 shared templates**, so members are near-duplicates
of each other without being byte-identical.

Real example, CAMP003 — six earbud reviews inside a 68-minute window:

```
C00058  16:51  [5★]  Excellent set of earbuds. The quality is really good and it came fast.
C00040  17:05  [5★]  Great set of earbuds, quality is excellent and delivery was quick.
C00066  17:11  [5★]  Really happy with this set of earbuds. The quality is great...
C00047  17:14  [5★]  Excellent set of earbuds. The quality is really good and it came fast.
C00038  17:41  [5★]  Really happy with this set of earbuds. The quality is great...
C00067  17:59  [5★]  Really happy with this set of earbuds. The quality is great...
```

**Basis:** directly mirrors He et al.'s finding that sellers recruit reviewers
through Facebook groups and supply talking points. Recruited reviewers paraphrase
a brief rather than writing independently.

### 6.5 Measured specificity gap

| `fraud_type` | unique-text ratio | mean length |
|---|---|---|
| `genuine` | 1.00 | 21.8 words |
| `text_deception` | 0.14 | 11.1 words |
| `coordinated_reuse` | 0.43 | 13.1 words |

---

## 7. Ratings

**Rule 1 — extremes.** Fraud ratings cluster at 1★ or 5★, rarely 2–3★. Matches
the incentivized-review literature the base paper cites.

**Rule 2 — polarity consistency (added in v2).** The rating must agree with the
polarity of the generated text.

v1 assigned fraud ratings independently of text, which produced rows such as
*"Excellent suitcase, quality is really good"* at 1★. That contradiction is
itself a real fraud signal — but it was **uncontrolled and undesigned**, which
means the model could have learned rating–text sentiment mismatch as a shortcut
instead of the three signals the project actually claims. Roughly 12 of 501 rows
were affected. v2 threads text polarity into the rating function.

Verified in v2: 0 positive-text-with-low-rating rows, 0 negative-text-with-high-rating rows.

| `fraud_type` | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---|---|---|---|---|
| `genuine` | 42 | 45 | 14 | 73 | 114 |
| `visual_manipulation` | 0 | 0 | 0 | 9 | 61 |
| `text_deception` | 10 | 2 | 0 | 2 | 56 |
| `coordinated_reuse` | 0 | 0 | 0 | 4 | 59 |

---

## 8. Timing and reviewer behaviour

| Group | Pattern | Measured |
|---|---|---|
| Campaign members | all inside one tight window | max span 2.8h across 12 campaigns |
| Everything else | spread across the collection period | 179 days |

**Basis:** these are deliberately *opposite* patterns so the model must learn
reuse **and** timing jointly. Reuse alone is not fraud — the manufacturer's photo
is legitimately reused by many genuine buyers over weeks. What separates fraud is
*who* reused it and *when*.

Two reviewer populations: `U#####` (320 ordinary accounts) and `C#####` (70
campaign accounts). Campaign accounts appear only in coordinated fraud.

---

## 9. Negative controls — deliberate traps

Rows whose `notes` begin with `NEGATIVE_CONTROL_`. They exist to catch shortcut
learning, and each targets a specific failure mode.

| Control | n | `label` | The trap | Pass condition |
|---|---|---|---|---|
| `legit_stock_reuse` | 8 | Genuine | Strong reuse signal on genuine reviews — the manufacturer's listing photo re-uploaded by real buyers, spread over weeks | Model must NOT flag these |
| `recompress_only_below_threshold` | 6 | Needs_Verification | Image re-saved but nothing edited | `manipulation_score` < 0.25 |
| `weak_reuse_not_campaign` | 6 | Needs_Verification | Only two accounts share an image; too weak to be a campaign | Should abstain, not flag |

Current status: all three pass by construction. **Re-check them after training.**
If your trained model flags `legit_stock_reuse`, the provenance module has
learned "any reuse = fraud" and the result is invalid regardless of headline
accuracy.

---

## 10. Splits

### 10.1 Files

| File | Rows | Fraud % | Purpose |
|---|---|---|---|
| `reviews_full.csv` | 503 | 39.4% | Everything generated |
| `train.csv` | 389 | 41.6% | Training. Deliberately fraud-rich. |
| `holdout_realistic.csv` | 86 | 15.1% | All reported metrics. Realistic prevalence. |
| `loco_folds.json` | — | — | Leave-one-category-out fold definitions |

### 10.2 The split rule — non-negotiable

Rows are partitioned **by `root_image_id`**, ~25% of roots reserved, stratified
within each category so all 15 categories appear on both sides.

**Why.** If transform variants of the same root photo appear in both train and
test, the model recognises the photograph rather than the fraud signal and
reports a meaningless ~98%. A random row split would do exactly this. Verified at
build time by assertion: root overlap = 0, image overlap = 0.

### 10.3 Why two different prevalences

- **Training at 41.6% fraud** — at real prevalence there would be too few fraud
  examples to learn from. Over-sampling fraud for training is standard and
  correct.
- **Reporting at 15.1% fraud** — approximates real platform rates so the reported
  numbers are honest.
- **Never report plain accuracy on the holdout.** At 15% prevalence, a model that
  always predicts Genuine scores 85%. Report precision, recall, F1, PR-AUC, and
  the operating threshold. **PR-AUC is the headline number.**

### 10.4 LOCO folds

`loco_folds.json` defines 3 folds of 4 held-out categories each, for the
generalisation experiment: train on the other 11, test on the unseen 4, and
compare how much text-only degrades versus image-forensic-only.

---

## 11. Known limitations

1. **Circular provenance ground truth.** The same code applies the transforms and
   the detector then finds them. pHash will score near-perfectly and that number
   means very little. Report it as a controlled check that the module functions,
   never as a performance claim. Raising this yourself is evidence you understand
   your own method.
2. **The embedding is a placeholder.** 64-d gradient-orientation + coarse colour
   descriptor, not a learned representation. **Swap point:
   `gen_images.compute_embedding()`** — the schema does not change.
3. **ELA on already-compressed sources.** Layer 1 stock photos were JPEG
   compressed before you obtained them, so ELA maps will be noisier than textbook
   examples.
4. **Small scale.** 503 rows from 183 roots. Use k-fold cross-validation, not a
   single split; expect wide variance.
5. **Behaviour is simulated, not observed.** Reviewer timing and rating patterns
   follow documented literature, but no real reviewer logs exist here.
6. **Text is template-generated.** Real deceptive text is more varied than any
   template pool. A text model trained here will not transfer to real reviews.
7. **Products are category-level, not item-level.** One `product_id` per
   category, so within-product comparisons are coarser than a real catalogue.

---

## 12. Reproduction

```
build_dataset.py   orchestration, campaigns, controls, splits, validation asserts
gen_images.py      root loading, transforms, manipulations, pHash, embeddings
gen_text.py        per-fraud-type text and rating generation
validate.py        8 post-build checks (separation, controls, integrity)
```

Scale by editing `N_GENUINE`, `N_VISUAL_MANIPULATION`, `N_TEXT_DECEPTION`,
`N_CAMPAIGNS` at the top of `build_dataset.py` and re-running. Seed is fixed at
`SEED = 20260908`; change it for a different draw.

Always run `validate.py` after regenerating. Check 1 (pHash separation) is the
one that fails first if transform magnitudes are altered.
