"""
baselines.py — the ablation table from PROJECT_SPEC.md section 8.2.

Task: binary, Fake vs not-Fake. Needs_Verification rows are excluded from
training and scoring but retained for the negative-control audit at the end.

All metrics are reported on holdout_realistic.csv (~15% fraud prevalence).
Accuracy is deliberately NOT the headline -- PR-AUC is.
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score)

import features as F

OUT = '../out'
SEED = 20260908

CONFIGS = {
    'text-only':            ['TEXT'],
    'image-forensic-only':  ['IMAGE'],
    'behaviour-only':       ['BEHAVIOUR'],
    'graph-only':           ['GRAPH'],
    'base-paper-equivalent': ['TEXT', 'BEHAVIOUR', 'GRAPH'],
    'FUSION (ours)':        ['TEXT', 'IMAGE', 'BEHAVIOUR', 'GRAPH'],
}


def load():
    full = pd.read_csv(f'{OUT}/reviews_full.csv')
    tr = pd.read_csv(f'{OUT}/train.csv')
    ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
    # features computed over the full corpus (structure only, no labels)
    blocks = F.build_all(full)
    X = pd.concat(blocks.values(), axis=1)
    X.index = full.review_id
    block_cols = {k: list(v.columns) for k, v in blocks.items()}
    return full, tr, ho, X, block_cols


def subset(X, full, ids, cols):
    m = full.set_index('review_id')
    rid = [r for r in ids if r in X.index]
    return X.loc[rid, cols], m.loc[rid]


def evaluate(name, cols, X, full, tr, ho, model='lr'):
    tr_ids = tr[tr.label != 'Needs_Verification'].review_id.tolist()
    ho_ids = ho[ho.label != 'Needs_Verification'].review_id.tolist()
    Xtr, mtr = subset(X, full, tr_ids, cols)
    Xho, mho = subset(X, full, ho_ids, cols)
    ytr = (mtr.label == 'Fake').astype(int).values
    yho = (mho.label == 'Fake').astype(int).values

    if model == 'rf':
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight='balanced', random_state=SEED)
    else:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight='balanced',
                                               random_state=SEED))
    clf.fit(Xtr.values, ytr)
    p = clf.predict_proba(Xho.values)[:, 1]
    yhat = (p >= 0.5).astype(int)

    res = {
        'config': name, 'model': model.upper(), 'n_feat': len(cols),
        'precision': precision_score(yho, yhat, zero_division=0),
        'recall': recall_score(yho, yhat, zero_division=0),
        'f1': f1_score(yho, yhat, zero_division=0),
        'pr_auc': average_precision_score(yho, p),
        'roc_auc': roc_auc_score(yho, p),
    }
    # per-fraud-type recall -- the result that matters most
    per = {}
    for ft in ['visual_manipulation', 'text_deception', 'coordinated_reuse']:
        mask = (mho.fraud_type == ft).values
        per[ft] = float(yhat[mask].mean()) if mask.sum() else float('nan')
    mask_g = (mho.fraud_type == 'genuine').values
    per['genuine_FPR'] = float(yhat[mask_g].mean()) if mask_g.sum() else float('nan')
    res.update(per)
    res['_probs'] = p
    res['_meta'] = mho
    return res


def main():
    full, tr, ho, X, block_cols = load()
    print(f"corpus={len(full)}  train={len(tr)}  holdout={len(ho)}  "
          f"features={X.shape[1]}")
    print(f"holdout fraud prevalence = "
          f"{100*(ho.label=='Fake').mean():.1f}%\n")

    results = []
    for name, blocks in CONFIGS.items():
        cols = [c for b in blocks for c in block_cols[b]]
        for m in (['lr', 'rf'] if name == 'FUSION (ours)' else ['lr']):
            results.append(evaluate(name, cols, X, full, tr, ho, model=m))

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                       for r in results])

    print("=== ABLATION TABLE (holdout, realistic prevalence) ===")
    print(df[['config', 'model', 'precision', 'recall', 'f1',
              'pr_auc', 'roc_auc']].round(3).to_string(index=False))

    print("\n=== PER-FRAUD-TYPE DETECTION RATE (recall within each type) ===")
    print(df[['config', 'model', 'visual_manipulation', 'text_deception',
              'coordinated_reuse', 'genuine_FPR']].round(3).to_string(index=False))

    df.drop(columns=[]).to_csv(f'{OUT}/ablation_results.csv', index=False)

    # ---- negative control audit on the FUSION model -------------------
    fusion = [r for r in results if r['config'] == 'FUSION (ours)'
              and r['model'] == 'RF'][0]
    audit_controls(full, tr, X, block_cols)
    return df


def audit_controls(full, tr, X, block_cols):
    """Score the negative controls with the fusion model trained on train.csv."""
    cols = [c for b in ['TEXT', 'IMAGE', 'BEHAVIOUR', 'GRAPH']
            for c in block_cols[b]]
    tr_ids = tr[tr.label != 'Needs_Verification'].review_id.tolist()
    Xtr, mtr = subset(X, full, tr_ids, cols)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 class_weight='balanced', random_state=SEED)
    clf.fit(Xtr.values, (mtr.label == 'Fake').astype(int).values)

    ctrl = full[full.notes.astype(str).str.startswith('NEGATIVE_CONTROL')]
    print("\n=== NEGATIVE CONTROL AUDIT (fusion RF) ===")
    for note, g in ctrl.groupby('notes'):
        ids = [r for r in g.review_id if r in X.index]
        p = clf.predict_proba(X.loc[ids, cols].values)[:, 1]
        flagged = int((p >= 0.5).sum())
        verdict = 'PASS' if flagged == 0 else f'FAIL ({flagged} flagged)'
        print(f"  {note[17:]:40s} n={len(ids):2d}  "
              f"mean_p={p.mean():.3f}  max_p={p.max():.3f}  {verdict}")


if __name__ == '__main__':
    main()
