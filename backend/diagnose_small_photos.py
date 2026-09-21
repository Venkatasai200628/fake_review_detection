"""
diagnose_small_photos.py -- is the edit detector still trustworthy on a 360px review photo?

The extension showed both "photo shows signs of editing (CNN 0.55)" and "photo under 1000px:
edit check less reliable" on the same Flipkart card. Those pull against each other, so the
question is whether the CNN keeps any skill once a photo is small and re-compressed the way
Flipkart and Amazon serve them.

Method: take test-split photos with known labels, shrink each to a target width the way a site
would, re-save as JPEG at web quality, rebuild the forensic maps and re-score with the SAME CNN.
ROC-AUC at each size says how much evidence survives.

    python diagnose_small_photos.py [n_per_class]
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config                                     # noqa: E402
sys.path.insert(0, config.CODE_DIR)
import cnn_forensics as CF                        # noqa: E402
import forensic_maps as FM                        # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 90
SIZES = [None, 800, 600, 480, 360, 240]           # None = original (<=1024px)
QUALITY = 80                                      # what a marketplace CDN typically serves

full = pd.read_csv(os.path.join(config.OUT_DIR, 'reviews_full.csv'))
te = full[full.split == 'test']
rng = np.random.default_rng(0)
pos = te[te.fraud_type == 'visual_manipulation'].sample(min(N, (te.fraud_type == 'visual_manipulation').sum()),
                                                        random_state=0)
neg = te[te.fraud_type == 'genuine'].sample(min(N, (te.fraud_type == 'genuine').sum()), random_state=0)
rows = pd.concat([pos, neg])
y = (rows.fraud_type == 'visual_manipulation').astype(int).values
print(f'{len(pos)} edited + {len(neg)} clean test photos, JPEG quality {QUALITY}\n')

model = CF.ForensicCNN()
import torch                                      # noqa: E402
model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'forensic_cnn.pt'), map_location='cpu'))
model.eval()


def shrink(im, width):
    if width is None:
        return im
    w, h = im.size
    if w <= width:
        return im
    im = im.resize((width, max(1, round(h * width / w))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=QUALITY)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


print('| photo width | ROC-AUC (edited vs clean) | mean score, edited | mean score, clean | flagged at 0.5: edited / clean |')
print('|---|---|---|---|---|')
for size in SIZES:
    maps = []
    for f in rows.image_file:
        im = Image.open(os.path.join(config.OUT_DIR, f)).convert('RGB')
        maps.append(FM.maps_from_image(shrink(im, size)))
    s = CF.predict(model, np.stack(maps))
    auc = roc_auc_score(y, s)
    lab = 'original (<=1024)' if size is None else f'{size}px'
    print(f'| {lab} | **{auc:.3f}** | {s[y==1].mean():.3f} | {s[y==0].mean():.3f} | '
          f'{(s[y==1]>=0.5).mean():.0%} / {(s[y==0]>=0.5).mean():.0%} |')

print('\nROC-AUC 0.5 means the score carries no information: at that point "signs of editing"')
print('is a coin flip and should neither be shown to the user nor allowed to move the risk score.')
