"""
app.py -- the local backend the Chrome extension talks to.

    uvicorn app:app --host 127.0.0.1 --port 8000

GET  /health          is the model loaded? how many features? memory size?
POST /analyze         reviews scraped from a page -> risk, decision, reasons
GET  /demo            a fake "product page" in Amazon's review markup, built
                      from held-out dataset reviews, to test the extension
                      without depending on Amazon
GET  /demo-images/..  the photos used by /demo

Runs only on 127.0.0.1: nothing leaves the laptop, nothing is logged to disk.
"""
import html
import os
import time
import random

import hmac

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline import Pipeline


def pick_demo_rows():
    """Held-out TEST reviews: some genuine, edited-photo, fake-text and one
    campaign. They are removed from the image memory so the backend meets
    them for the first time, like reviews on a real page."""
    t = pd.read_csv(os.path.join(config.OUT_DIR, 'test_balanced.csv'))
    rng = random.Random(config.DEMO_SEED)
    pick = []
    for ft, n in [('genuine', 5), ('visual_manipulation', 3), ('text_deception', 3)]:
        ids = t[t.fraud_type == ft].review_id.tolist()
        pick += rng.sample(ids, n)
    camp = t[t.fraud_type == 'coordinated_reuse']
    cid = rng.choice(sorted(camp.campaign_id.unique()))
    pick += camp[camp.campaign_id == cid].review_id.tolist()[:3]
    return t.set_index('review_id').loc[pick].reset_index()


DEMO = pick_demo_rows()
PIPE = Pipeline(exclude_review_ids=DEMO.review_id.tolist())
N_FEATURES = PIPE.check_model()

app = FastAPI(title='Fake review forensics backend')
# Only the extension and the local demo page need to call this. '*' was fine while it was
# bound to 127.0.0.1; a hosted copy should not be callable from any website a victim visits.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r'^(chrome-extension://[a-z]+|https?://(127\.0\.0\.1|localhost)(:\d+)?)$',
    allow_methods=['GET', 'POST'], allow_headers=['*'])


def require_token(authorization: str):
    """No token configured = open, which is correct on 127.0.0.1 and wrong once hosted."""
    if not config.API_TOKEN:
        return
    sent = authorization[7:].strip() if authorization.lower().startswith('bearer ') else authorization.strip()
    if not hmac.compare_digest(sent, config.API_TOKEN):
        raise HTTPException(status_code=401, detail='missing or bad token')
app.mount('/demo-images', StaticFiles(directory=os.path.join(config.OUT_DIR, 'images')), name='img')
EXT_DIR = os.path.join(config.PROJECT, 'extension')
app.mount('/ext', StaticFiles(directory=EXT_DIR), name='ext')

# /demo?preview=1 runs the extension's OWN content.js + styles.css inside the
# page (with a tiny stand-in for the chrome.* API), so the badges can be seen
# without installing the extension -- e.g. in a screenshot or a presentation.
PREVIEW_SHIM = """
<link rel="stylesheet" href="/ext/styles.css?v=__V__">
<script>
// Stand-in for the chrome.* extension API, backed by localStorage so the parts that are
// supposed to be remembered -- your own Genuine/Fake verdicts -- really do survive a reload
// here, exactly as they would in the installed extension.
const _store = () => { try { return JSON.parse(localStorage.getItem('rpf') || '{}'); } catch (e) { return {}; } };
const _listeners = [];
window.chrome = {
  storage: {
    local: {
      get: async (keys) => {
        const all = _store();
        if (!keys) return all;
        const ks = typeof keys === 'string' ? [keys] : (Array.isArray(keys) ? keys : Object.keys(keys));
        const out = {};
        ks.forEach((k) => { if (k in all) out[k] = all[k]; });
        return out;
      },
      set: async (obj) => {
        const all = _store();
        const changes = {};
        Object.entries(obj).forEach(([k, v]) => { changes[k] = {oldValue: all[k], newValue: v}; all[k] = v; });
        localStorage.setItem('rpf', JSON.stringify(all));
        _listeners.forEach((fn) => fn(changes, 'local'));
      },
    },
    onChanged: { addListener: (fn) => _listeners.push(fn) },
  },
  runtime: { sendMessage: (msg, cb) => {
    fetch('/analyze', {method: 'POST', headers: {'Content-Type': 'application/json'},
                       body: JSON.stringify(msg.payload)})
      .then(r => r.json()).then(data => cb({ok: true, data}))
      .catch(e => cb({ok: false, error: String(e)}));
  } }
};
</script>
<script src="/ext/sites.js?v=__V__"></script>
<script src="/ext/content.js?v=__V__"></script>
"""


@app.get('/health')
def health():
    return {'status': 'ok', 'auth_required': bool(config.API_TOKEN), 'model_dir': config.MODEL_DIR, 'features': N_FEATURES,
            'uses_cnn': PIPE.uses_cnn, 'side_models': sorted(PIPE.side),
            'memory_size': len(PIPE.mem),
            'decision_bands': {'genuine_below': PIPE.t_low, 'fake_above': PIPE.t_high}}


@app.post('/analyze')
def analyze(payload: dict, authorization: str = Header(default='')):
    require_token(authorization)
    return PIPE.analyze(payload)


@app.get('/demo', response_class=HTMLResponse)
def demo(preview: int = 0):
    rng = random.Random(config.DEMO_SEED)
    rows = DEMO.sample(frac=1, random_state=config.DEMO_SEED)
    cards = []
    for r in rows.itertuples():
        ts = pd.Timestamp(r.timestamp)
        stars = f"{r.rating}.0 out of 5 stars"
        img = os.path.basename(r.image_file)
        cards.append(f"""
      <div data-hook="review" class="a-section review" id="{r.review_id}">
        <div class="a-profile"><span class="a-profile-name">{html.escape(r.reviewer_id)}</span></div>
        <i data-hook="review-star-rating" class="stars"><span class="a-icon-alt">{stars}</span></i>
        <span data-hook="review-date">Reviewed in India on {ts:%d %B %Y}, {ts:%H:%M}</span>
        <span data-hook="review-body"><span>{html.escape(r.review_text)}</span></span>
        <div class="review-image-tile-section">
          <img data-hook="review-image-tile" class="review-image-tile" src="/demo-images/{img}">
        </div>
      </div>""")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Demo product - fake review forensics</title>
<style>
 body{{font-family:Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#111}}
 .review{{border-top:1px solid #ddd;padding:14px 0}} .a-profile-name{{font-weight:bold}}
 .stars{{display:block;color:#c45500;font-style:normal;margin:4px 0}}
 [data-hook=review-date]{{color:#565959;font-size:13px}}
 [data-hook=review-body]{{display:block;margin:8px 0}}
 .review-image-tile{{width:120px;height:120px;object-fit:cover;border:1px solid #ccc}}
 .note{{background:#f3f3f3;padding:10px;border-radius:6px;font-size:13px}}
</style></head><body>
<h1 id="productTitle">Demo product page ({len(rows)} held-out reviews)</h1>
<p class="note">This page reuses Amazon's review markup (<code>data-hook="review"</code>) with
held-out reviews from the v6 dataset. The extension should add a badge to every review.</p>
<div id="cm-cr-dp-review-list">{''.join(cards)}</div>
{PREVIEW_SHIM.replace('__V__', str(int(time.time()))) if preview else ''}
</body></html>"""


@app.get('/popup-preview', response_class=HTMLResponse)
def popup_preview():
    """The extension popup rendered in a normal page (for screenshots)."""
    with open(os.path.join(EXT_DIR, 'popup.html'), encoding='utf-8') as f:
        page = f.read()
    shim = """<script>
window.chrome = {
  runtime: { sendMessage: (msg, cb) => fetch('/health').then(r => r.json())
      .then(data => cb({ok: true, data})).catch(e => cb({ok: false, error: String(e)})) },
  storage: { local: { get: (k, cb) => cb({enabled: true,
      lastSummary: {checked: 14, fake: 6, needs_verification: 2, genuine: 6}}), set: () => {} } }
};
</script>"""
    page = page.replace('<head>', '<head><base href="/ext/">', 1)
    return page.replace('<script src="popup.js">', shim + '<script src="popup.js">', 1)


@app.get('/demo/answers')
def demo_answers():
    """Ground truth for the demo page, only for checking the badges yourself."""
    return DEMO[['review_id', 'label', 'fraud_type']].to_dict('records')
