# README 6: Why the individual models score lower, and screenshots of the extension

**Owner:** Venkata Sai · **Date:** 19 September 2026 · **Previous:** `readme5.md`

## 1. "All combined is good, but the individual ones are lower, right?"
Yes, and that is expected and intended. Each individual model is **built to see only one kind of fake**:

| | Edited-photo fakes | Fake-text reviews | Coordinated reuse | False alarms |
|---|---|---|---|---|
| Text only | 7% (blind to it) | **83%** | 100% | 8% |
| Image only | **84%** | 9% (blind to it) | 95% | 8% |
| All combined | 81% | 75% | 100% | 3.5% |

- Text only is scored on *all* fakes, including the 555 edited-photo fakes whose text is written to sound genuine.
  No text model can catch those, so its overall number looks low. It is still strong on its own job (83%).
- Image only is the same the other way round: 84% of edited photos, but it cannot see deceptive writing on a clean photo (9%).
- This is the result the whole project is built to show: **each single view misses what the other sees, and only the
  combination catches both.** If one individual model scored as high as fusion, the project would have nothing to prove.
- **Where the individuals can honestly improve:** text is still 8 hand-counted features. Switching to RoBERTa (guide §16 step 4)
  would raise text-only. The image CNN is already strong on its job (ROC-AUC 0.959 for spotting edits).

## 2. Screenshots
Saved in `readmes\screenshots\`:
- `extension_top.png`: summary bar ("14 reviews checked · 6 likely fake · 2 need verification · 6 genuine") and the first badges
- `extension_bottom.png`: close-up of photo-edit detections (CNN 0.72 and 0.99)
- `extension_demo_full.png`: the whole demo page

### How they were made
- The backend ran locally. The page is the `/demo?preview=1` page, which runs the extension's **own** `content.js`
  and `styles.css`, so the badges look exactly as they do with the extension installed.
- I first tried loading the real extension into a headless Chrome. Newer Chrome versions block `--load-extension`
  from the command line, so no badges appeared. You can load it yourself with `chrome://extensions` → Load unpacked.
- My `chrome --version` check printed "Opening in existing browser session" and may have opened an empty tab in your Chrome.
  It is safe to close.

### What the screenshots show (read honestly)
- **Coordinated reuse** (speaker reviews): "4 matching photos from other accounts posted within 6 hours" plus
  "identical text in N other reviews". Correct, **Likely fake**.
- **Genuine reviews**: green, risk 19–32%. Correct.
- **Mouse review (edited photo)**: you can see the pasted square on the mouse. CNN 0.99 and photo score 65%, but the text
  sounds real (text 22%), so the combined risk is 55%: **Needs verification**, not Likely fake. It is a partial catch.
- **Handbag review (fake text)**: correctly **Likely fake** (text 100%). The CNN also said 0.72 "signs of editing",
  although this photo was **not** edited. That is a CNN false alarm; the right answer came from the text.

## 3. Next
Tune the decision bands (so cases like the mouse become "Likely fake" and borderline ones abstain), then test on a
live Amazon page.
