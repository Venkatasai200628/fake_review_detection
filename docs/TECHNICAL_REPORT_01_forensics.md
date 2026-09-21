# Technical Report 01 — Manipulation Module Investigation

**Date:** 2026-09-08
**Scope:** why the manipulation-detection module failed, three measurement leaks
found and removed, and the resolution fix that recovered the signal.
**Dataset version produced:** v4 (1024px, equalised compression history)
**Companion documents:** `PROJECT_SPEC.md` (design), `DATASHEET.md` (data)

---

## 1. Why this investigation happened

The first ablation run reported that image-forensic-only detected **100%** of
`visual_manipulation` cases and that the fusion model achieved a perfect
PR-AUC of 1.000. Both numbers were wrong, and finding out why took four
iterations. Each iteration removed a different way the measurement was cheating.

This report exists because every one of those four problems is the kind of thing
an examiner asks about, and because three of the four are mistakes that would
have quietly invalidated the final results if they had not been caught.

**Summary of what was wrong:**

| # | Problem | Symptom | Status |
|---|---|---|---|
| 1 | Feature read the generator's ground truth | image-only = 1.000 | fixed |
| 2 | Manipulated images had an extra JPEG pass | ROC-AUC 0.198 (inverted) | fixed |
| 3 | Same root photo in both classes of the CV split | AUC systematically < 0.5 | fixed |
| 4 | Output resolution too low | all detectors at chance | fixed |

---

## 2. Leak 1 — the feature was reading the answer key

### What was wrong

`features.py` used this as its principal image feature:

```python
'img_manipulation': r.manipulation_score,
```

`manipulation_score` is a **column written by the generator**. It is ground
truth: 0.01–0.12 for clean images, 0.66–0.96 for manipulated ones. Feeding it to
the classifier was equivalent to handing over the labels.

### Why it produced a perfect score

Nothing was being detected. The model read a column that said "this one is
manipulated" and reported 100% accuracy on it.

### Why it was a natural mistake to make

The column legitimately belongs in the dataset — it is ground truth, and the
datasheet documents it as such. The error was including a ground-truth column in
the *feature* set. This is the single most common way a machine-learning result
gets silently invalidated, and it is worth stating in the report as a
methodological point rather than hiding.

### The fix

Built `forensics.py`, which computes manipulation evidence **from the pixels of
the saved JPEG files**, exactly as a deployed system would receive them:

- **Error Level Analysis** — recompress at fixed quality, take the absolute
  difference. The signal is not the mean error but its *inconsistency* across the
  frame: a spliced patch carries a different compression history than its
  surroundings. Features: block-level standard deviation, coefficient of
  variation, max-to-mean ratio, skew, hot-block fraction.
- **Noise residual** — subtract a median-filtered copy to isolate
  high-frequency residual, then measure how much local noise energy varies block
  to block. A camera produces a roughly uniform noise floor; a splice from
  another photograph does not.
- **Copy-move indicator** — block signature matching for near-identical,
  spatially distant regions.

`manipulation_score` was removed from the feature set entirely.

### Effect

| | Before (leaking) | After (forensic) |
|---|---|---|
| image-only PR-AUC | 0.804 | 0.531 |
| fusion RF PR-AUC | 1.000 | 0.824 |
| fusion beats base-paper-equivalent? | yes | **no** |

Removing the leak removed the result. That was the correct outcome, and it is
what prompted everything below.

---

## 3. Leak 2 — the classes had different compression histories

### How it was found

An experiment comparing four detector families reported ROC-AUCs of **0.198**,
**0.281**, and **0.200**. Values below 0.5 are not weak results; they are
*anti-predictive*. An AUC of 0.198 becomes 0.802 when inverted, which means the
features were finding something strong and systematically backwards.

### What was wrong

`apply_manipulation()` ended with its own JPEG recompression pass at quality
70–85. Genuine images never received an equivalent pass. So manipulated images
were simply **more compressed** — smoother, lower error energy, lower noise
variance — than genuine ones.

The detectors were learning *"this image went through an extra save"*. That is a
property of the generation pipeline, not evidence of manipulation. It is the same
class of error as Leak 1: a signal that exists only because of how the data was
built.

### Why this one matters beyond the project

This is the trap that makes deepfake and manipulation detectors fail in
deployment. A detector trained where fake images share a processing history that
real images do not will learn the processing, not the manipulation, and collapse
on real data. Worth citing directly in the limitations section.

### The fix

Removed the recompression from `apply_manipulation()`. `save_derived()` now
applies **exactly one** JPEG pass to every row, so all images share an identical
compression history regardless of class.

---

## 4. Leak 3 — root photo leakage inside the experiment

### What was wrong

Even after fixing Leak 2, AUCs stayed inverted at 512px (0.185–0.265).

The experiment built one genuine **and** one manipulated sample from each root
photograph. With ordinary stratified cross-validation, the classifier partially
identifies *which root photo* an image came from. Since every root contributes
exactly one sample of each class, recognising a root in the training fold implies
the **opposite** label for the test-fold sample from that same root. Hence the
systematic inversion.

### Why this is notable

This is precisely the leakage rule the project specification already states for
the main dataset (`PROJECT_SPEC.md` §4.9: split by `root_image_id`). The main
dataset splits correctly. The experiment I wrote to diagnose the forensic module
did not. The rule has to be applied everywhere, not once.

### The fix

Switched to `GroupKFold` with `root_image_id` as the grouping variable. All AUCs
immediately moved to the plausible range (0.51–0.55 at 512px), confirming the
inversion was an artifact.

---

## 5. The real problem — output resolution

With all three leaks removed, the honest measurement:

| Detector family | 512px | 1024px |
|---|---|---|
| ELA only | 0.513 | **0.739** |
| Noise residual only | 0.507 | 0.692 |
| ORB copy-move | 0.535 | 0.429 |
| Local colour statistics | 0.522 | 0.551 |
| All combined | 0.554 | **0.758** |

### What this shows

At 512px, every manipulation detector performs at chance. At 1024px, ELA and
noise residual both carry real signal.

The manipulations were being applied at 768px and the derived images then saved
at 512px. That downscale averages away exactly the high-frequency evidence both
detectors depend on — the compression-history discontinuity and the noise-floor
break. The evidence was destroyed by the pipeline before any detector saw it.

### Why raising the resolution is a legitimate fix and not result-shopping

This is worth being precise about, because "we changed a parameter until the
result appeared" is a fair criticism to anticipate.

- 512px was chosen for **file size convenience**, not for any modelling reason.
  It was never justified.
- Real e-commerce platforms serve product photographs at roughly 1000–1500px.
  Amazon's standard product image is 1500px on the long edge. **1024px is the
  more realistic setting**, so the change moves the dataset toward reality, not
  away from it.
- The change was made *after* measuring both settings side by side under
  identical, leak-free conditions, and the measurement is reported here in full
  including the settings that did not work.

What would **not** have been legitimate: removing the platform re-encode pass,
increasing manipulation strength until it became visible, or reverting to the
ground-truth column. None of those were done.

### ORB copy-move: a genuine negative

ORB keypoint self-matching underperformed at both resolutions and got *worse* at
1024 (0.429). The hypothesis was that geometry-based matching would survive
re-encoding better than compression-based methods. It did not. Reported as a
negative result and excluded from the feature set.

---

## 6. Current honest performance

### Forensic module in isolation

Corpus-wide grouped cross-validation, manipulated vs genuine, n=358 (70
manipulated):

```
ROC-AUC = 0.622 +/- 0.093
```

Real but modest. ELA on stock photographs that were already JPEG-compressed
before collection, then re-encoded once more by the pipeline, is a hard setting.
This number is honest and it is what should be reported.

### Full ablation (holdout, 15.0% fraud prevalence)

| Config | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| text-only | 0.467 | 0.583 | 0.519 | 0.664 | 0.824 |
| image-forensic-only | 0.227 | 0.417 | 0.294 | 0.495 | 0.677 |
| behaviour-only | 0.240 | 1.000 | 0.387 | 0.187 | 0.603 |
| graph-only | 0.200 | 0.500 | 0.286 | 0.270 | 0.626 |
| base-paper-equivalent | 0.333 | 0.833 | 0.476 | 0.691 | 0.882 |
| **FUSION (LR)** | 0.385 | 0.833 | 0.526 | **0.758** | **0.904** |
| FUSION (RF) | 0.375 | 0.750 | 0.500 | 0.735 | 0.890 |

**Fusion beats the base-paper-equivalent configuration: 0.758 vs 0.691 PR-AUC.**
That is the project's central claim, and it now rests on honestly computed
features. The margin is small and the holdout has only 12 fraud rows, so it is
not yet statistically meaningful.

### What is working and what is not

| Module | Status |
|---|---|
| Provenance / reuse (pHash + embeddings) | **Working.** `coordinated_reuse` detected at 1.00 by every image-based config. |
| Coordination graph | Working, weak alone (PR-AUC 0.270) as expected — it is context, not a standalone detector. |
| Manipulation (ELA + noise) | **Weak but real.** 0.622 AUC in isolation. |
| Text | Working, 0.664 PR-AUC after the overlap cases were added. |
| Behaviour | Poor alone (0.187) with a 58.5% false-positive rate; useful only in fusion. |

---

## 7. Two negative controls now failing

This regressed and must be dealt with before any further work:

```
legit_stock_reuse                 n=8  mean_p=0.039  PASS
recompress_only_below_threshold   n=6  mean_p=0.448  FAIL (2 of 6 flagged)
weak_reuse_not_campaign           n=6  mean_p=0.379  FAIL (3 of 6 flagged)
```

**Interpretation.** The model is drifting toward treating *any* reuse as fraud —
exactly the failure mode these controls exist to catch (`DATASHEET.md` §9). The
`recompress_only` failure is new and suggests the forensic features now fire on
benign recompression, which is a false-positive mode worth understanding rather
than tuning away.

The good news is that `legit_stock_reuse` still passes cleanly, so the strongest
version of the trap is being resisted.

---

## 8. Changes made to the codebase

| File | Change |
|---|---|
| `forensics.py` | **New.** ELA, noise residual, copy-move computed from pixels. |
| `experiment_forensics.py` | **New.** Resolution and detector-family comparison. |
| `features.py` | `manipulation_score` removed; forensic features joined in. Docstring records why. |
| `gen_images.py` | `WORK_SIZE` 768→1280, `OUT_SIZE` 512→1024. Recompression removed from `apply_manipulation`; `save_derived` now applies exactly one uniform pass. |
| `build_dataset.py` | (from previous session) holdout fraud subsample stratified by `fraud_type`. |
| `gen_text.py` | (from previous session) overlap cases: lazy-genuine and sophisticated-fake text. |

Dataset regenerated as **v4**: 502 rows, 1024px images, equalised compression,
383 train rows at 39.4% fraud, 80 holdout rows at 15.0% fraud, split integrity
verified, pHash separation margin +4 (clean).

---

## 9. What this means for the paper

The instinct that a failed module cannot be published is only half right. What
cannot be published is a module that fails *and is reported as working*. What is
entirely publishable, and stronger than a clean sweep:

> We evaluated three visual-forensic signal families. Provenance and coordination
> evidence transfer robustly. Compression-based manipulation evidence is
> substantially degraded by platform re-encoding and image downscaling, and we
> quantify the resolution threshold below which it disappears entirely.

That last clause is a contribution in its own right: a measured statement that
manipulation detection collapses below ~1024px. Nobody in the surveyed literature
reports this, because most work evaluates on full-resolution research datasets
rather than platform-processed images.

The project's actual novelty — platform-wide provenance and visual coordination —
is intact and is carried by the modules that work.

---

## 10. Next steps, in order

1. **Diagnose the two failing negative controls.** Determine which features fire
   on benign recompression and weak reuse. Do not simply raise the threshold
   until they pass.
2. **Repeated grouped k-fold cross-validation** over the full corpus. The
   0.758-vs-0.691 gap rests on 12 fraud rows and is currently inside the noise.
   Report means with confidence intervals.
3. **Swap the placeholder embedding for CLIP/ViT.** Still the single-function
   change at `gen_images.compute_embedding()`. This should strengthen the
   provenance module, which is the part that works.
4. **Leave-one-category-out run** for the generalisation claim.
5. Optional: a small CNN on ELA/noise residual maps rather than the seven summary
   statistics, to see whether the 0.622 can be improved.
