"""
verify_paper_numbers.py -- audit of the paper-model table.

Question being audited (Venkata Sai, 20 Sep 2026):
  "the baselines catch 100% of fake text and we catch 98%; if they are 100% good,
   why is their accuracy only 0.83? that does not match."

This script does NOT re-train anything and does NOT re-state the table. It recomputes,
from the raw split files and the stored per-type rates, the accuracy each model MUST have
if the rates are real, and compares it with the accuracy that was reported. If any number
had been written by hand or produced by a buggy metric, the two would disagree.

  accuracy = [ (1 - false_alarm_rate)*n_genuine
             + edited_photo_caught*n_visual + fake_text_caught*n_text
             + coordinated_caught*n_coord ] / n_test

It then breaks OUR model down the same way, row by row, and prints the individual
text_deception reviews we get wrong (the "98% not 100%" cases), with their scores.

Output: ../out/model_comparison/PAPER_NUMBERS_AUDIT.md
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import json

import joblib
import numpy as np
import pandas as pd

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores_split.csv'

full = pd.read_csv(f'{OUT}/reviews_full.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
res = pd.read_csv(f'{DST}/paper_models_on_our_data.csv')

TYPES = ['visual_manipulation', 'text_deception', 'coordinated_reuse']
n = te.fraud_type.value_counts()
n_gen, N = int(n['genuine']), len(te)
lines = []
P = lines.append

P('# Audit of the paper-model table (no re-statement, everything recomputed)\n')
P(f'Test set: **{N} reviews** read from `out/test_balanced.csv`.\n')
P('| group | rows | share of test set |')
P('|---|---|---|')
for k in ['genuine'] + TYPES:
    P(f'| {k} | {int(n[k])} | {int(n[k])/N:.1%} |')
P('')
P(f'Fake rows: {N - n_gen}. A model that is perfect on text but blind to photos can therefore be right on at most '
  f'{(N - int(n["visual_manipulation"]))/N:.4f} of the test set '
  f'({N} - {int(n["visual_manipulation"])} edited-photo rows), i.e. **{(N-int(n["visual_manipulation"]))/N:.1%}**.\n')

# ---------------------------------------------------------------- reconciliation
P('## 1. Does each reported accuracy follow from its own per-type rates?\n')
P('`rebuilt = [(1-false_alarms)*genuine + caught_visual*135 + caught_text*135 + caught_coord*135] / 810`\n')
P('| model | edited photo | fake text | coordinated | false alarms | rows right (rebuilt) | rebuilt acc | reported acc | difference |')
P('|---|---|---|---|---|---|---|---|---|')
bad = 0
for r in res.to_dict('records'):
    right = ((1 - r['false_alarms']) * n_gen
             + r['edited_photo_caught'] * n['visual_manipulation']
             + r['fake_text_caught'] * n['text_deception']
             + r['coordinated_caught'] * n['coordinated_reuse'])
    acc = right / N
    d = acc - r['accuracy']
    bad += abs(d) > 1e-9
    P(f"| {r['model']} | {r['edited_photo_caught']:.1%} | {r['fake_text_caught']:.1%} | "
      f"{r['coordinated_caught']:.1%} | {r['false_alarms']:.1%} | {right:.1f} | {acc:.4f} | "
      f"{r['accuracy']:.4f} | {d:+.2e} |")

# ---------------------------------------------------------------- our model, row by row
blocks = F.build_all(full)
for k in blocks:
    blocks[k] = blocks[k].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    blocks[k].index = full.review_id.values
Xall = pd.concat(blocks.values(), axis=1)
cols = json.load(open(f'{OUT}/model/model_meta.json'))['features']
model = joblib.load(f'{OUT}/model/fusion_rf.joblib')
p = model.predict_proba(Xall.loc[te.review_id, cols].values)[:, 1]
t = te.assign(p=p, pred=(p >= 0.5).astype(int))
rate = t.groupby('fraud_type').pred.mean()
right = int((t.pred == t.is_fake).sum())
r_ours = {'edited_photo_caught': rate['visual_manipulation'], 'fake_text_caught': rate['text_deception'],
          'coordinated_caught': rate['coordinated_reuse'], 'false_alarms': rate['genuine'],
          'accuracy': right / N}
rebuilt = ((1 - r_ours['false_alarms']) * n_gen + r_ours['edited_photo_caught'] * n['visual_manipulation']
           + r_ours['fake_text_caught'] * n['text_deception'] + r_ours['coordinated_caught'] * n['coordinated_reuse'])
P(f"| **OURS (recomputed now)** | {r_ours['edited_photo_caught']:.1%} | {r_ours['fake_text_caught']:.1%} | "
  f"{r_ours['coordinated_caught']:.1%} | {r_ours['false_alarms']:.1%} | {rebuilt:.1f} | {rebuilt/N:.4f} | "
  f"{r_ours['accuracy']:.4f} | {rebuilt/N - r_ours['accuracy']:+.2e} |")
P('')
P(f'Rows where the rebuilt accuracy disagrees with the reported accuracy by more than 1e-9: **{bad}**.\n')

# ---------------------------------------------------------------- the mistake budget
P('## 2. Where each model loses its 810 rows (counts, not percentages)\n')
P('| model | misses: edited photo | misses: fake text | misses: coordinated | false alarms | total mistakes | accuracy |')
P('|---|---|---|---|---|---|---|')
rows = res.to_dict('records') + [dict(model='**OURS**', **r_ours)]
for r in rows:
    mv = round((1 - r['edited_photo_caught']) * n['visual_manipulation'])
    mt = round((1 - r['fake_text_caught']) * n['text_deception'])
    mc = round((1 - r['coordinated_caught']) * n['coordinated_reuse'])
    fa = round(r['false_alarms'] * n_gen)
    P(f"| {r['model']} | {mv} | {mt} | {mc} | {fa} | {mv+mt+mc+fa} | {1-(mv+mt+mc+fa)/N:.3f} |")
P('')

# ---------------------------------------------------------------- the 98% vs 100% rows
miss = t[(t.fraud_type == 'text_deception') & (t.pred == 0)]
P(f'## 3. The "98% not 100%" cases: the {len(miss)} fake-text reviews our model misses\n')
P(f"Our fake-text recall is {r_ours['fake_text_caught']:.4f} = "
  f"{int(n['text_deception']) - len(miss)}/{int(n['text_deception'])}. The missed rows:\n")
P('| review_id | our score | rating | text |')
P('|---|---|---|---|')
for m in miss.to_dict('records'):
    txt = str(m['review_text']).replace('|', '/')[:110]
    P(f"| {m['review_id']} | {m['p']:.3f} | {m['rating']} | {txt} |")
P('')
tm = t[t.fraud_type == 'text_deception']
P(f"Our score on those rows: {miss.p.mean():.3f} mean (threshold 0.5). "
  f"On the {len(tm)-len(miss)} fake-text rows we do catch: {tm[tm.pred==1].p.mean():.3f} mean.\n")

open(f'{DST}/PAPER_NUMBERS_AUDIT.md', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
print(f'\nwritten: {DST}/PAPER_NUMBERS_AUDIT.md')
