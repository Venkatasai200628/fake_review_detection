"""
test_parity.py -- guide section 19.6 step 1: does the backend compute the SAME
features as training?

Takes held-out test reviews, removes them from the image memory, sends them
through Pipeline.analyze() as if scraped from a page, and compares every
feature with what training computed for the same rows (features.build_all on
the full dataset). Big differences = training/serving skew = the model would
silently break in the extension.

    python test_parity.py
"""
import os
import sys
import numpy as np
import pandas as pd

import config
from pipeline import Pipeline, F

N = 12
out = config.OUT_DIR
test = pd.read_csv(os.path.join(out, 'test_balanced.csv')).sample(N, random_state=1)

# training-side features for these rows
os.chdir(config.CODE_DIR)
F.CNN_SCORES_CSV = os.path.join(out, 'cnn_manip_scores_split.csv')
F.TEXT_SCORES_CSV = os.path.join(out, 'text_model_scores_split.csv')
full = pd.read_csv(os.path.join(out, 'reviews_full.csv'))
blocks = F.build_all(full)
Xt = pd.concat(blocks.values(), axis=1)
Xt.index = full.review_id.values
Xt = Xt.loc[test.review_id]

# serving-side features for the same rows
pipe = Pipeline(exclude_review_ids=test.review_id.tolist())
payload = {'reviews': [{'id': r.review_id, 'text': r.review_text, 'rating': r.rating,
                        'date': r.timestamp, 'reviewer': r.reviewer_id,
                        'product_id': r.product_id,
                        'images': [f"/demo-images/{os.path.basename(r.image_file)}"]}
                       for r in test.itertuples()]}
captured = {}
orig = F.build_all


def spy(corpus, overrides=None):
    b = orig(corpus, overrides)
    X = pd.concat(b.values(), axis=1)
    X.index = corpus.review_id.values
    captured['X'] = X.tail(N)
    return b


F.build_all = spy
res = pipe.analyze(payload, remember=False, raw_ids=True)
Xs = captured['X']
Xs.index = test.review_id.values

cols = [c for c in pipe.meta['features'] if c in Xs.columns]
diff = (Xs[cols].astype(float) - Xt[cols].astype(float)).abs()
worst = diff.max().sort_values(ascending=False)
print(f"{N} held-out reviews, {len(cols)} model features compared")
print("largest absolute differences (0 = identical):")
print(worst.head(12).round(4).to_string())
exact = (worst < 1e-3).sum()
print(f"\n{exact}/{len(cols)} features identical (|diff| < 0.001)")
print("\nexpected small differences: img_min_phash_dist (training hashed the photo just "
      "before its JPEG save, serving hashes the saved file)")
print("\nmodel output on these reviews:")
lab = test.set_index('review_id')
for r in res['reviews']:
    print(f"  {r['id']}  truth={lab.loc[r['id'], 'fraud_type']:20s} risk={r['risk']:.2f}  {r['decision']}")
