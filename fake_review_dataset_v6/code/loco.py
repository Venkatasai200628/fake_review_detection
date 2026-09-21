"""
loco.py -- leave-one-category-out: does the model work on a PRODUCT TYPE it never trained on?

The model is always tested on unseen PHOTOS, but so far always of product types it has also seen
in training (shoes, watches, mouse, ...). That does not answer Venkata Sai's question: on a real
marketplace the next review is for a phone, a kettle, a mattress -- categories the training set
never contained.

This trains 15 times. Each run holds out one entire category -- every photo, every review, every
campaign of it -- trains on the other 14, and tests only on the held-out one. It also reports
each branch separately, because they are expected to behave very differently:

  reuse (pHash + ResNet-50)  nothing was fitted on our data at all: pHash is an algorithm and
                             ResNet-50 is frozen ImageNet weights, so it should not care
  forensic CNN               trained on our photos: reads compression and noise, which in
                             principle is independent of what the object is -- this run checks
                             whether that is true or just a hope
  text                       trained on our templates, so this is where transfer should fail

Output: ../out/model_comparison/LOCO.md
    python loco.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
SEED = 20260919
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores.csv'          # cross-fitted over all rows
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores.csv'

full = pd.read_csv(f'{OUT}/reviews_full.csv')
meta = json.load(open(f'{OUT}/model/model_meta.json'))
cols = meta['features']
blocks = F.build_all(full)
X = pd.concat(blocks.values(), axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
X.index = full.review_id.values
y = (full.label == 'Fake').astype(int).values
cat = full.category.values
img_cols = [c for c in blocks['IMAGE'].columns]
txt_cols = [c for c in blocks['TEXT'].columns]

rows = []
for c in sorted(pd.unique(cat)):
    te = cat == c
    tr = ~te
    m = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=2, max_features='sqrt',
                             class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)
    m.fit(X.loc[full.review_id[tr], cols].values, y[tr])
    p = m.predict_proba(X.loc[full.review_id[te], cols].values)[:, 1]
    d = (p >= 0.5).astype(int)
    r = {'held-out category': c, 'rows': int(te.sum()), 'fake %': round(100 * y[te].mean(), 1),
         'accuracy': accuracy_score(y[te], d), 'F1': f1_score(y[te], d),
         'MCC': matthews_corrcoef(y[te], d), 'ROC-AUC': roc_auc_score(y[te], p)}
    # per-branch, same held-out split
    for name, cc in [('image-only ROC', img_cols), ('text-only ROC', txt_cols)]:
        mb = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                  random_state=SEED, class_weight='balanced_subsample')
        mb.fit(X.loc[full.review_id[tr], cc].values, y[tr])
        pb = mb.predict_proba(X.loc[full.review_id[te], cc].values)[:, 1]
        r[name] = roc_auc_score(y[te], pb)
    # can the edit detector still spot edited photos of an unseen product type?
    vm = te & (full.fraud_type == 'visual_manipulation').values
    gen = te & (full.fraud_type == 'genuine').values
    s = X.loc[full.review_id[te], 'img_cnn_manip'].values
    yy = np.r_[np.ones(vm.sum()), np.zeros(gen.sum())]
    ss = np.r_[X.loc[full.review_id[vm], 'img_cnn_manip'].values,
               X.loc[full.review_id[gen], 'img_cnn_manip'].values]
    r['edit-CNN ROC'] = roc_auc_score(yy, ss) if len(set(yy)) > 1 else float('nan')
    rows.append(r)
    print(f"  {c:20s} acc {r['accuracy']:.3f}  MCC {r['MCC']:.3f}  ROC {r['ROC-AUC']:.3f}  "
          f"img {r['image-only ROC']:.3f}  txt {r['text-only ROC']:.3f}  editCNN {r['edit-CNN ROC']:.3f}",
          flush=True)

d = pd.DataFrame(rows)
d.to_csv(f'{DST}/loco.csv', index=False)
L = ['# Leave-one-category-out: performance on a product type never seen in training\n',
     'Each row: the whole category is removed from training -- every photo, review and campaign -- '
     'the model is trained on the other 14, and tested only on the held-out one.\n',
     '| held-out category | rows | accuracy | F1 | MCC | ROC-AUC | image-only ROC | text-only ROC | edit-CNN ROC |',
     '|---|---|---|---|---|---|---|---|---|']
for r in rows:
    L.append(f"| {r['held-out category']} | {r['rows']} | {r['accuracy']:.3f} | {r['F1']:.3f} | "
             f"{r['MCC']:.3f} | {r['ROC-AUC']:.3f} | {r['image-only ROC']:.3f} | "
             f"{r['text-only ROC']:.3f} | {r['edit-CNN ROC']:.3f} |")
mean = d[['accuracy', 'F1', 'MCC', 'ROC-AUC', 'image-only ROC', 'text-only ROC', 'edit-CNN ROC']].mean()
L.append(f"| **mean** | | **{mean['accuracy']:.3f}** | **{mean['F1']:.3f}** | **{mean['MCC']:.3f}** | "
         f"**{mean['ROC-AUC']:.3f}** | **{mean['image-only ROC']:.3f}** | **{mean['text-only ROC']:.3f}** | "
         f"**{mean['edit-CNN ROC']:.3f}** |")
L.append(f"\nWorst category: {d.loc[d.MCC.idxmin(), 'held-out category']} "
         f"(MCC {d.MCC.min():.3f}); best: {d.loc[d.MCC.idxmax(), 'held-out category']} "
         f"(MCC {d.MCC.max():.3f}).")
open(f'{DST}/LOCO.md', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L[-6:]))
