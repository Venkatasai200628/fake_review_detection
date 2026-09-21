"""
diagnose_reuse.py -- why did a brand-new photo get "4 matching photos from other accounts"?

Venkata Sai tested the extension on a live Flipkart page and a family photo that nobody could
have posted before was reported as matching 4 other accounts. This script takes real photo URLs,
runs them through the SAME code the backend uses, and prints what they actually matched and how
strongly, so the answer is measured rather than argued.

    python diagnose_reuse.py <url> [<url> ...]

For each photo it reports the closest photos in the image memory by both tests the pipeline uses:
    pHash Hamming distance <= 20   (PHASH_REUSE_THRESHOLD)
    ResNet-50 cosine      >= 0.72  (RESNET_REUSE_THRESHOLD)
A pair counts as "reuse" if EITHER fires, so either one being loose creates false matches.
"""
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import config                                     # noqa: E402
sys.path.insert(0, config.CODE_DIR)
import features as F                              # noqa: E402
import gen_images as GI                           # noqa: E402
import reuse_embedding as RE                      # noqa: E402
from pipeline import Pipeline                     # noqa: E402

urls = sys.argv[1:]
if not urls:
    print(__doc__)
    sys.exit(1)

p = Pipeline()
mem = p.mem.reset_index(drop=True)
print(f'image memory: {len(mem)} photos\n')

# memory side
mem_ph = mem.phash.tolist()
emb = F._load_reuse_embeddings()
have_emb = emb is not None
if have_emb:
    ids = mem.review_id.tolist()
    M = np.stack([emb[i] for i in ids if i in emb])
    M_ids = [i for i in ids if i in emb]
    M = M / np.linalg.norm(M, axis=1, keepdims=True)

for u in urls:
    im, _ = p.prepare(p.load_image(u))
    ph = GI.compute_phash(im)
    d = np.array([GI.phash_distance(ph, q) for q in mem_ph])
    print(f'--- {u[:88]}')
    print(f'    size {im.size[0]}x{im.size[1]}'
          + ('  (under 1000px -> edit check less reliable)' if min(im.size) < config.MIN_FORENSIC_EDGE else ''))
    n_ph = int((d <= F.PHASH_REUSE_THRESHOLD).sum())
    print(f'    pHash <= {F.PHASH_REUSE_THRESHOLD}: {n_ph} memory photos "match"')
    for k in np.argsort(d)[:5]:
        r = mem.iloc[k]
        flag = 'MATCH' if d[k] <= F.PHASH_REUSE_THRESHOLD else '     '
        print(f'       {flag} dist {int(d[k]):2d}  {r.review_id}  {str(r.reviewer_id)[:22]:22s} {str(r.image_id)[:12]}')
    if have_emb:
        v = RE.encode([im])[0]
        v = v / np.linalg.norm(v)
        c = M @ v
        n_c = int((c >= F.RESNET_REUSE_THRESHOLD).sum())
        print(f'    ResNet-50 cosine >= {F.RESNET_REUSE_THRESHOLD}: {n_c} memory photos "match"')
        for k in np.argsort(-c)[:5]:
            flag = 'MATCH' if c[k] >= F.RESNET_REUSE_THRESHOLD else '     '
            rid = M_ids[k]
            r = mem[mem.review_id == rid].iloc[0]
            print(f'       {flag} cos {c[k]:.3f}  {rid}  {str(r.reviewer_id)[:22]:22s} {str(r.image_id)[:12]}')
        union = int(((d <= F.PHASH_REUSE_THRESHOLD) | np.isin(mem.review_id.values,
                     [M_ids[k] for k in np.where(c >= F.RESNET_REUSE_THRESHOLD)[0]])).sum())
        print(f'    => counted as reuse partners (either test): {union}')
    print()
