"""
explain_review.py -- WHY did this one review get that risk score?

The extension showed "Likely fake, 78%" on a long, detailed, obviously real Amazon review and
the only line under it was "no photo: checked on text and behaviour only", which explains
nothing. If the tool flags something it has to be able to say why.

This computes the same 41 features the model uses, then measures each feature's contribution by
putting it back to the value an ordinary review would have (the memory median) and re-scoring.
The drop in risk is that feature's share of the verdict -- a straightforward ablation, no
approximations.

    python explain_review.py payload.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import Pipeline                     # noqa: E402

PLAIN = {
    'txt_minilm_p': 'how deceptive the wording looks to the sentence model',
    'txt_generic_ratio': 'share of generic praise words',
    'txt_specific_ratio': 'share of concrete, checkable detail',
    'txt_len': 'review length in words',
    'txt_sent_count': 'number of sentences',
    'txt_exclam': 'exclamation marks',
    'txt_upper_ratio': 'CAPITAL LETTERS',
    'txt_exact_dup_count': 'identical copies of this text elsewhere',
    'rating': 'star rating',
    'img_cnn_manip': 'photo editing score',
    'img_max_cosine': 'closest matching photo',
    'img_reuse_in_burst': 'photos reused in a short burst',
    'img_burst_ratio': 'share of reuse inside a burst',
    'clip_cross_modal': 'how well the photo matches the words',
    'b_reviewer_review_count': 'how many reviews this account has here',
    'g_community_size': 'size of the account cluster it belongs to',
}

payload = json.load(open(sys.argv[1], encoding='utf-8'))
p = Pipeline()
res = p.analyze(payload, remember=False)
X = p.last_X                                        # set by analyze()
med = p.last_med
cols = p.meta['features']

for i, r in enumerate(res['reviews']):
    print(f"\n=== {r['id']}  risk {r['risk']:.3f}  -> {r['decision']}")
    print(f"    text branch {r['text_score']}   photo branch {r['image_score']}")
    base = r['risk']
    row = X.iloc[[i]].copy()
    contrib = []
    for c in cols:
        alt = row.copy()
        alt[c] = med[c]
        q = float(p.fusion.predict_proba(alt[cols].values)[:, 1][0])
        contrib.append((base - q, c, float(row[c].iloc[0]), float(med[c])))
    contrib.sort(key=lambda t: -abs(t[0]))
    print(f"\n    {'feature':26s} {'this review':>12s} {'typical':>10s}  effect on risk")
    for d, c, v, m in contrib[:10]:
        arrow = 'pushes UP  ' if d > 0 else 'pulls DOWN '
        print(f"    {c:26s} {v:12.3f} {m:10.3f}  {arrow} {abs(d):.3f}   {PLAIN.get(c, '')}")
    print(f"\n    reasons currently shown: {r['reasons']}")
