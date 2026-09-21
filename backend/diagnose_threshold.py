"""
diagnose_threshold.py -- is PHASH_REUSE_THRESHOLD = 20 defensible?

diagnose_reuse.py showed a brand-new Flipkart photo "matching" 40 memory photos by pHash while
ResNet-50 correctly said no (max cosine 0.60 vs a 0.72 threshold). This measures the pHash
distance distribution on our own data:

    TRUE pairs      two rows sharing a root_image_id  (really the same photo, lightly edited)
    FALSE pairs     two rows with different roots      (unrelated photos)

and reports, for each candidate threshold, how many true pairs it keeps and how many false pairs
it lets through -- both as a rate and as the expected number of false partners per photo against
a memory of this size, which is the number the user actually sees in the extension.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                                     # noqa: E402
sys.path.insert(0, config.CODE_DIR)
import gen_images as GI                           # noqa: E402

full = pd.read_csv(os.path.join(config.OUT_DIR, 'reviews_full.csv'))
ph = full.phash.tolist()
root = full.root_image_id.values
n = len(full)
print(f'{n} photos in the dataset\n')

rng = np.random.default_rng(0)
true_d, false_d = [], []
for _ in range(60000):
    i, j = rng.integers(0, n, 2)
    if i == j:
        continue
    d = GI.phash_distance(ph[i], ph[j])
    (true_d if root[i] == root[j] else false_d).append(d)
true_d, false_d = np.array(true_d), np.array(false_d)
print(f'sampled {len(true_d)} same-photo pairs and {len(false_d)} different-photo pairs')
print(f'same-photo    pHash distance: median {np.median(true_d):.0f}  p95 {np.percentile(true_d,95):.0f}  max {true_d.max()}')
print(f'different     pHash distance: median {np.median(false_d):.0f}  p5  {np.percentile(false_d,5):.0f}  min {false_d.min()}\n')

print('| threshold | same-photo pairs kept (recall) | different pairs wrongly matched | expected false partners per photo |')
print('|---|---|---|---|')
for t in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
    tpr = (true_d <= t).mean()
    fpr = (false_d <= t).mean()
    mark = '  <-- current' if t == 20 else ''
    print(f'| {t} | {tpr:.3f} | {fpr:.5f} | {fpr * n:.1f}{mark} |')
print('\n"expected false partners per photo" = false-pair rate x memory size. That is what shows up')
print('in the extension as "N matching photos from other accounts".')
