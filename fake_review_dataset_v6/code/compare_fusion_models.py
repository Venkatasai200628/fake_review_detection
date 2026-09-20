"""
compare_fusion_models.py -- Phase A: is there a better fusion model than our Random Forest?

Same 41 features, same 2,490 train rows, same 810 test rows, same grouped CV by root_image_id.
Only the classifier on top changes. A candidate replaces the current model ONLY if it wins on
the honest column (accuracy on test rows whose review text never appears in training), not on
the inflated overall accuracy.

Output: ../out/model_comparison/fusion_models.csv / .md
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores_split.csv'
SEED = 20260919
CACHE = f'{DST}/_X_cache.pkl'

full = pd.read_csv(f'{OUT}/reviews_full.csv')
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
cols = json.load(open(f'{OUT}/model/model_meta.json'))['features']

if os.path.exists(CACHE):
    Xall = joblib.load(CACHE)
else:
    blocks = F.build_all(full)
    for k in blocks:
        blocks[k] = blocks[k].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        blocks[k].index = full.review_id.values
    Xall = pd.concat(blocks.values(), axis=1)
    joblib.dump(Xall, CACHE)

Xtr, ytr = Xall.loc[tr.review_id, cols].values, tr.is_fake.values
Xte, yte = Xall.loc[te.review_id, cols].values, te.is_fake.values
Xho, yho = Xall.loc[ho.review_id, cols].values, ho.is_fake.values
groups = tr.root_image_id.values

# the honest subset: test rows whose text is NOT a verbatim copy of a training text
trtxt = set(tr.review_text.fillna('').str.strip().str.lower())
newtxt = ~te.review_text.fillna('').str.strip().str.lower().isin(trtxt).values
print(f'test rows with unseen text: {newtxt.sum()} / {len(te)}')
SPLITS = list(GroupKFold(n_splits=5).split(Xtr, ytr, groups))   # folds never split a root photo


def cands():
    yield 'Random Forest 500 (current)', RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED, class_weight='balanced')
    yield 'Extra Trees 500', ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED, class_weight='balanced_subsample')
    yield 'Extra Trees 1000, leaf 1', ExtraTreesClassifier(
        n_estimators=1000, min_samples_leaf=1, n_jobs=-1, random_state=SEED, class_weight='balanced_subsample')
    yield 'Extra Trees 1000, leaf 2, sqrt', ExtraTreesClassifier(
        n_estimators=1000, min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=SEED,
        class_weight='balanced_subsample')
    yield 'Soft vote ET+RF+HGB', VotingClassifier(estimators=[
        ('et', ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=-1, random_state=SEED,
                                    class_weight='balanced_subsample')),
        ('rf', RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=SEED,
                                      class_weight='balanced')),
        ('hgb', HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=SEED))],
        voting='soft', n_jobs=1)
    yield 'HistGradientBoosting', HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
    yield 'XGBoost', XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.85, colsample_bytree=0.8,
        reg_lambda=2.0, n_jobs=8, random_state=SEED, eval_metric='logloss', tree_method='hist')
    yield 'XGBoost deep', XGBClassifier(
        n_estimators=900, learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3, reg_lambda=3.0, n_jobs=8, random_state=SEED, eval_metric='logloss', tree_method='hist')
    yield 'Logistic regression', make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=4000, C=1.0, class_weight='balanced'))
    yield 'Stacking (RF+XGB+HGB -> LR)', StackingClassifier(
        estimators=[('rf', RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                                  random_state=SEED, class_weight='balanced_subsample')),
                    ('xgb', XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.85,
                                          colsample_bytree=0.8, reg_lambda=2.0, n_jobs=8, random_state=SEED,
                                          eval_metric='logloss', tree_method='hist')),
                    ('hgb', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=SEED))],
        final_estimator=LogisticRegression(max_iter=2000), cv=SPLITS, n_jobs=1)


import sys
ONLY = sys.argv[1:]
rows = []
for name, clf in cands():
    if ONLY and not any(o.lower() in name.lower() for o in ONLY):
        continue
    t0 = time.time()
    fit_kw = {}
    clf.fit(Xtr, ytr, **fit_kw)
    p_te = clf.predict_proba(Xte)[:, 1]
    p_ho = clf.predict_proba(Xho)[:, 1]
    yh = (p_te >= 0.5).astype(int)
    t = te.assign(f=yh, ok=(yh == yte))
    # grouped CV on train, honest estimate of generalisation to unseen photos
    # stacking already uses the grouped folds internally; nesting them breaks the
    # partition, so it reports test numbers only
    if isinstance(clf, StackingClassifier):
        cv_roc, cv_acc = float('nan'), float('nan')
    else:
        cvp = cross_val_predict(clf, Xtr, ytr, cv=SPLITS, method='predict_proba')[:, 1]
        cv_roc, cv_acc = roc_auc_score(ytr, cvp), accuracy_score(ytr, (cvp >= 0.5).astype(int))
    r = {'model': name,
         'cv_roc_auc': cv_roc,
         'cv_accuracy': cv_acc,
         'accuracy': accuracy_score(yte, yh), 'f1': f1_score(yte, yh),
         'roc_auc': roc_auc_score(yte, p_te),
         'pr_auc_holdout15': average_precision_score(yho, p_ho),
         'acc_unseen_text': t.ok.values[newtxt].mean(),
         'edited_photo_caught': t[t.fraud_type == 'visual_manipulation'].f.mean(),
         'fake_text_caught': t[t.fraud_type == 'text_deception'].f.mean(),
         'coordinated_caught': t[t.fraud_type == 'coordinated_reuse'].f.mean(),
         'false_alarms': t[t.fraud_type == 'genuine'].f.mean(),
         'seconds': round(time.time() - t0)}
    rows.append(r)
    print(f"  {name:30s} cvROC {r['cv_roc_auc']:.4f}  acc {r['accuracy']:.4f}  "
          f"UNSEEN-TEXT acc {r['acc_unseen_text']:.4f}  ROC {r['roc_auc']:.4f}  "
          f"photo {r['edited_photo_caught']:.1%}  FA {r['false_alarms']:.1%}  [{r['seconds']}s]", flush=True)
    old = pd.read_csv(f'{DST}/fusion_models.csv') if os.path.exists(f'{DST}/fusion_models.csv') else pd.DataFrame()
    if len(old):
        old = old[old.model != name]
    pd.concat([old, pd.DataFrame([r])], ignore_index=True).to_csv(f'{DST}/fusion_models.csv', index=False)

df = pd.read_csv(f'{DST}/fusion_models.csv').sort_values('acc_unseen_text', ascending=False)
md = ['# Phase A: fusion models compared (same 41 features, same splits)\n',
      'Ranked by accuracy on test rows whose review text never appears in training '
      '(the column that is not inflated by template duplication).\n',
      '| model | CV ROC (grouped) | test acc | **acc, unseen text** | F1 | ROC | PR-AUC@15% | edited photo | fake text | coord | false alarms |',
      '|---|---|---|---|---|---|---|---|---|---|---|']
for r in df.to_dict('records'):
    md.append(f"| {r['model']} | {r['cv_roc_auc']:.4f} | {r['accuracy']:.4f} | **{r['acc_unseen_text']:.4f}** | "
              f"{r['f1']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc_holdout15']:.4f} | {r['edited_photo_caught']:.1%} | "
              f"{r['fake_text_caught']:.1%} | {r['coordinated_caught']:.1%} | {r['false_alarms']:.1%} |")
open(f'{DST}/fusion_models.md', 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))
