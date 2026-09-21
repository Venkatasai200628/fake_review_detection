# README 4: Is our accuracy from images + text combined, or only text?

**Owner:** Venkata Sai · **Date:** 19 September 2026 · **Previous:** `readme3.md`

## Your question
"Is the accuracy we got for images and text combined, or only the text?"

## Short answer
The headline numbers (**accuracy 0.852, PR-AUC 0.843**) come from the **combined** model. It uses all four
blocks together: **text + image + behaviour (star rating, posting time) + graph**. But when I split the model
apart, you are right to be suspicious. **Most of that score comes from the text and the star rating. The image
part adds only a little.**

## How I checked
I trained the same Random Forest 7 times, each time with a different group of features, on the **same** train photos,
and tested all 7 on the **same** unseen test photos (`test_balanced.csv`, 50% fake). The last column uses the
realistic 15%-fake holdout. The results are saved in `out/model/modality_ablation_test_balanced.csv`.

| Features used | Accuracy | F1 | ROC-AUC | PR-AUC (15% fake) | Catches edited-photo fakes | Catches fake-text fakes | False alarms on genuine |
|---|---|---|---|---|---|---|---|
| Text only (8 features) | 0.796 | 0.772 | 0.883 | 0.766 | **23%** | 84% | 10% |
| Image only (22 features) | 0.686 | 0.622 | 0.730 | 0.480 | **43%** | **13%** | 14% |
| Behaviour only (rating/time) | 0.694 | 0.699 | 0.752 | 0.354 | 74% ⚠ | 64% | 32% |
| Graph only | 0.564 | 0.569 | 0.599 | 0.237 | 42% | 42% | 45% |
| **Text + Image** | 0.841 | 0.823 | 0.915 | 0.790 | 44% | 79% | 6% |
| Text + Behaviour + Graph (**no image**) | 0.833 | 0.837 | 0.935 | **0.846** | 62% | 95% | 19% |
| **ALL four (our model)** | **0.852** | **0.852** | **0.946** | **0.843** | **70%** | 87% | 15% |

## What this honestly means
1. **Combined, yes.** 0.852 accuracy is the text + image + behaviour + graph model.
2. **The image part on its own is weak.** It is 68.6% accurate and catches only 43% of edited photos.
3. **On this single test split, adding the image barely helps.** Without images, PR-AUC is 0.846; with images it is 0.843
   (the same, within noise). Accuracy goes up by only +0.019 and ROC-AUC by +0.011. Images do raise edited-photo
   detection (62% → 70%) and cut false alarms (19% → 15%). But they lower fake-text detection (95% → 87%).
4. **Cross-validation says images help more (+0.042, significant over 25 splits).** This single split has only
   69 fakes in the realistic holdout, so it is noisy. Even so, **the honest summary is "images help a little",
   not "images carry the model".** My earlier "the model is fine" was too strong for the image part.
5. **⚠ There is a shortcut in our generated data.** Behaviour alone (just star rating + time) catches 74% of
   edited-photo fakes, without looking at the photo. The reason: every edited-photo fake gets 4–5 stars with happy
   text, while genuine reviews spread over 1–5 stars. So part of what looks like "image detection" in the combined model
   is really the model noticing the star rating. This is a flaw in how we generated the data (guide §11), not
   real forensic skill.

## What should be fixed (in order)
1. **Remove the rating shortcut** in `gen_text.py` / `build_dataset.py`. Give edited-photo fakes the same star
   distribution as genuine reviews (including some negative reviews with negative text). Then behaviour alone can
   no longer spot them, and only the photo can.
2. **Make the image evidence stronger:** a small CNN on the ELA / noise-residual maps instead of 13 summary
   numbers (guide §12.5). This is the core computer-vision work.
3. Rebuild, then re-run cross-validation **and** this 7-way table. The goal is for the image-only column to catch
   clearly more than 43% of edited photos, and for "no image" vs "all" to show a clear gap.

Until then, when presenting: say the **fusion** model reaches 0.852 / 0.843, show this table, and state openly that
the image module is currently the weakest part. Examiners respect that far more than an overclaim.
