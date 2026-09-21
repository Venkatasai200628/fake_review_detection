"""
full_metrics.py -- every metric for every model, not just accuracy.

Venkata Sai's point is correct: accuracy alone does not tell you a model is good. This scores all
eight models on the same 810 test reviews using the per-row predictions saved by
compare_paper_models.py, and adds the metrics that are hard to fake:

  Balanced accuracy   mean of recall on fakes and recall on genuine -- immune to class ratio
  MCC                 Matthews correlation; -1..+1, and it only goes high if ALL FOUR cells of
                      the confusion matrix are good. The standard answer to "accuracy is not
                      everything".
  Specificity         share of genuine reviews correctly left alone
  ROC-AUC / PR-AUC    threshold-free: how well the model RANKS, independent of the 0.5 cutoff
  Cohen's kappa       agreement above what guessing at the same base rate would give

with 95% confidence intervals from 2,000 bootstrap resamples, so "better" can be checked rather
than asserted.

Output: ../out/model_comparison/FULL_METRICS.md
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             average_precision_score, precision_score, recall_score,
                             roc_auc_score)

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores_split.csv'

te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
y, yh_ = te.is_fake.values, ho.is_fake.values
res = pd.read_csv(f'{DST}/paper_models_on_our_data.csv').set_index('key')

# ---- our model's per-row scores
full = pd.read_csv(f'{OUT}/reviews_full.csv')
blocks = F.build_all(full)
X = pd.concat(blocks.values(), axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
X.index = full.review_id.values
meta = json.load(open(f'{OUT}/model/model_meta.json'))
model = joblib.load(f'{OUT}/model/fusion_rf.joblib')
P = {'OURS': (model.predict_proba(X.loc[te.review_id, meta['features']].values)[:, 1],
              model.predict_proba(X.loc[ho.review_id, meta['features']].values)[:, 1])}
for k in res.index:
    P[res.loc[k, 'model']] = (pd.read_csv(f'{DST}/preds/{k}_test.csv').p_fake.values,
                              pd.read_csv(f'{DST}/preds/{k}_holdout.csv').p_fake.values)


def metrics(p, p_ho):
    d = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, d).ravel()
    return {
        'Accuracy': accuracy_score(y, d), 'Balanced acc': balanced_accuracy_score(y, d),
        'MCC': matthews_corrcoef(y, d), 'Kappa': cohen_kappa_score(y, d),
        'Precision': precision_score(y, d, zero_division=0), 'Recall': recall_score(y, d),
        'Specificity': tn / (tn + fp), 'F1': f1_score(y, d),
        'ROC-AUC': roc_auc_score(y, p), 'PR-AUC': average_precision_score(y, p),
        'PR-AUC @15%': average_precision_score(yh_, p_ho),
    }


rng = np.random.default_rng(0)
root = te.root_image_id.values
groups = pd.unique(root)
BOOT = [rng.choice(len(groups), len(groups), replace=True) for _ in range(2000)]
idx_of = {g: np.where(root == g)[0] for g in groups}


def ci(p, fn):
    vals = []
    for b in BOOT:
        idx = np.concatenate([idx_of[groups[i]] for i in b])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


rows = {n: metrics(*v) for n, v in P.items()}
df = pd.DataFrame(rows).T.sort_values('MCC', ascending=False)
names = list(df.columns)

L = ['# Every metric, not just accuracy\n',
     'Same 810 test reviews, same threshold (0.5). Ranked by **MCC**, which only goes high when all '
     'four cells of the confusion matrix are good -- the standard answer to "accuracy is not everything".\n',
     '| model | ' + ' | '.join(names) + ' |',
     '|---' * (len(names) + 1) + '|']
for n, r in df.iterrows():
    b = '**' if n == 'OURS' else ''
    L.append(f'| {b}{n}{b} | ' + ' | '.join(f'{b}{r[c]:.3f}{b}' for c in names) + ' |')

L.append('\n## Is the gap real? 95% CI, 2,000 bootstrap resamples grouped by root photo\n')
L.append('| model | Accuracy [95% CI] | MCC [95% CI] | ROC-AUC [95% CI] |')
L.append('|---|---|---|---|')
for n in df.index:
    p = P[n][0]
    a = ci(p, lambda yy, pp: accuracy_score(yy, (pp >= 0.5).astype(int)))
    m = ci(p, lambda yy, pp: matthews_corrcoef(yy, (pp >= 0.5).astype(int)))
    r_ = ci(p, roc_auc_score)
    b = '**' if n == 'OURS' else ''
    L.append(f'| {b}{n}{b} | {rows[n]["Accuracy"]:.3f} [{a[0]:.3f}, {a[1]:.3f}] | '
             f'{rows[n]["MCC"]:.3f} [{m[0]:.3f}, {m[1]:.3f}] | '
             f'{rows[n]["ROC-AUC"]:.3f} [{r_[0]:.3f}, {r_[1]:.3f}] |')

L.append('\nNon-overlapping intervals mean the difference is not sampling noise.\n')
open(f'{DST}/FULL_METRICS.md', 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L))
