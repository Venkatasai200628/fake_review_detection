# README 7: "Is the dataset imbalanced?" and the redesigned extension UI

**Owner:** Venkata Sai · **Date:** 19 September 2026 · **Previous:** `readme6.md`

## 1. "7% and 9% means the dataset is imbalanced, right?"
**No.** Those numbers are **detection rates**, not shares of the dataset. The dataset is balanced at every level:

| | Count | Share |
|---|---|---|
| Genuine | 1,650 | 50% |
| Fake | 1,650 | 50% |
| of which edited photo | 555 | 33.6% of fakes |
| of which fake text | 555 | 33.6% of fakes |
| of which coordinated reuse | 540 | 32.7% of fakes |

and every product category has exactly 110 genuine + 110 fake (train and test separately).

**What 7% actually means:** of the 555 edited-photo fakes, the **text-only model** flags only about 40. The text of
those reviews was deliberately written to sound genuine, so there is nothing in the text to find. **9%** is the same
the other way round: the image-only model cannot catch fake-text reviews, because their photos are real and unedited.

**Would more data fix it?** No. Adding more edited-photo reviews would not teach a text model to see pixels. It is
**blindness by design**, and it is the reason the combined model exists: fusion catches 81% and 75% of the same fakes.

**What we actually do:**
- Report the per-type table (readme5 §4), because it *proves* each modality is needed.
- Use the **combined** model in the extension, and show the text and photo scores next to it as explanation.
- To make the individual modules stronger *at their own job*: RoBERTa for text (guide §16 step 4). The image CNN
  is already strong (ROC-AUC 0.959 at spotting edits).

## 2. Extension redesign
**Before:** a plain coloured box with one line of scores and bullet points, and a dark text bar on top.

**After** (files: `extension\styles.css`, `content.js`, `popup.html`, `popup.js`):

| Part | New design |
|---|---|
| Review card | White card with a coloured header: **verdict pill** with icon (✕ Likely fake / ⚠ Needs verification / ✓ Looks genuine) and a large **fake-risk %** |
| Evidence | Two **meter bars**, "Text evidence" and "Photo evidence" (green / orange / red by level). "no photo" when there is none |
| Reasons | Bullets coloured by verdict. Neutral facts ("spread over time, can be normal") get grey bullets so they don't look like accusations |
| Loading | A spinner with "Checking photo and text…" and shimmering placeholder lines, instead of a dashed outline |
| Summary panel | Logo, "N reviews checked", three **count chips** (likely fake / need verification / look genuine), a **proportion bar**, and a note that a verdict is a signal, not proof |
| Error state | A red panel that says "Review check unavailable" and shows the exact command to start the backend |
| Popup | Dark header with logo, a green/red **status dot** for the backend, model features ("40 + CNN"), photos in memory, the last page's counts as chips, and a proper **on/off switch** |

Technical care taken:
- Every class starts with `rpf-` and the cards reset their own font, so Amazon's CSS cannot distort them.
- All review text is inserted with `textContent` (no HTML injection from page content). Only our own fixed SVG icons use `innerHTML`.
- The page watcher now ignores changes the extension itself makes, so drawing cards no longer triggers a re-check.
- `backend/app.py` got `/popup-preview`, which shows the popup in a normal page for screenshots.

## 3. Screenshots (`readmes\screenshots\`)
- `extension_v2_top.png`: summary panel + the first verdict cards
- `extension_v2_edit_detected.png`: the edited mouse photo. Text 22% (it sounds genuine), photo 65%, CNN 0.99, verdict **Needs verification**
- `extension_v2_popup.png`: the toolbar popup
- `extension_v2_full.png`: the whole demo page

These were taken in headless Chrome using `/demo?preview=1` (the extension's own `content.js` + `styles.css`, running in the page).
In your normal Chrome, load `extension\` with **Load unpacked**. It looks the same.

## 4. Next
Tune the decision bands (the mouse case should become "Likely fake"), then test on a live Amazon page.
