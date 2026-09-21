# Backend + Chrome extension: how to run

## 1. Start the backend (PowerShell)
```powershell
cd "D:\sem-5 projects\computer vision\backend"
$env:HF_HUB_OFFLINE = "1"      # CLIP weights are already downloaded; don't go online
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Wait until it prints `Application startup complete`, then open http://127.0.0.1:8000/health.
You should see `"status": "ok"` and the number of model features.

## 2. Load the extension in Chrome
1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked** and choose `D:\sem-5 projects\computer vision\extension`
4. Pin "Review Photo Forensics". Its popup shows whether the backend is running.

## 3. Try it
- **Demo page (always works):** http://127.0.0.1:8000/demo shows 14 held-out reviews in Amazon's
  markup. Each gets a green / amber / red badge with reasons. The true answers are at `/demo/answers`.
- **Real Amazon page:** open any amazon.in / amazon.com product page and scroll to the reviews. Only
  reviews with photos get the full photo check. The others are scored on text and rating only (the badge says so).

## 4. Plug in a better model later
Train it with `dataset\fake_review_dataset_v6\code\train_model.py`. It writes `out\model\fusion_rf.joblib`,
`text_rf.joblib`, `image_rf.joblib`, `model_meta.json` (and `forensic_cnn.pt` from `cnn_forensics.py`).
Then **restart the backend**. On start-up, `check_model()` refuses to run if the model needs a feature
the pipeline cannot compute, so a mismatch cannot slip through silently.
The extension itself never changes.

## 5. Check the backend matches training
```powershell
python test_parity.py
```
It sends held-out reviews through the backend as if they were scraped from a page, and compares every
feature with what training computed for the same rows.
