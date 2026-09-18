"""
train_model.py -- train the FINAL fusion model once, test it on photos it has
never seen, and save it to disk so the browser-extension backend can load it.

crossval.py answers "is fusion better than the baseline?" (25 folds, paired
test). This script answers a different question: "give me one trained model I
can ship". It:

  1. builds the four feature blocks (TEXT / IMAGE / BEHAVIOUR / GRAPH) over the
     whole corpus -- the same transductive setting as crossval.py, i.e. the
     reuse index sees every image, but no test LABEL is ever used
  2. trains a Random Forest on train.csv rows only (balanced 50/50)
  3. evaluates on test_balanced.csv (50/50) and holdout_realistic.csv (~15%
     fake, the number to report), per fraud type, and on the
     Needs_Verification controls
  4. applies the three-way decision (Genuine / Needs Verification / Fake)
  5. saves ../out/model/fusion_rf.joblib + feature list + thresholds

Run:  python train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

import features as F

OUT = '../out'
MODEL_DIR = f'{OUT}/model'
SEED = 20260919

# Three-way decision bands on the fake-probability (guide section 5.3).
# Below T_LOW -> Genuine, above T_HIGH -> Fake, in between -> Needs Verification.
T_LOW, T_HIGH = 0.35, 0.65


def decide(p):
    return np.where(p < T_LOW, 'Genuine',
                    np.where(p > T_HIGH, 'Fake', 'Needs_Verification'))


def report(name, y, p):
    yhat = (p >= 0.5).astype(int)
    r = {
        'set': name, 'rows': int(len(y)), 'fake_pct': round(100 * y.mean(), 1),
        'PR-AUC': round(average_precision_score(y, p), 3),
        'ROC-AUC': round(roc_auc_score(y, p), 3),
        'F1': round(f1_score(y, yhat), 3),
        'Precision': round(precision_score(y, yhat, zero_division=0), 3),
        'Recall': round(recall_score(y, yhat), 3),
    }
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    r.update(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    return r


def main():
    full = pd.read_csv(f'{OUT}/reviews_full.csv')
    blocks = F.build_all(full)
    X = pd.concat(blocks.values(), axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = list(X.columns)
    X.index = full.review_id.values

    train_ids = pd.read_csv(f'{OUT}/train.csv').review_id
    test = pd.read_csv(f'{OUT}/test_balanced.csv')
    hold = pd.read_csv(f'{OUT}/holdout_realistic.csv')
    nv = pd.read_csv(f'{OUT}/needs_verification_controls.csv')
    y_of = dict(zip(full.review_id, (full.label == 'Fake').astype(int)))

    model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   class_weight='balanced', n_jobs=-1,
                                   random_state=SEED)
    model.fit(X.loc[train_ids].values, train_ids.map(y_of).values)
    print(f"trained on {len(train_ids)} rows, {len(cols)} features")

    rows = []
    for name, part in [('test_balanced (50% fake)', test),
                       ('holdout_realistic (~15% fake)', hold)]:
        p = model.predict_proba(X.loc[part.review_id].values)[:, 1]
        rows.append(report(name, part.review_id.map(y_of).values, p))
    res = pd.DataFrame(rows)
    print("\n=== FINAL MODEL on unseen root photos (threshold 0.5) ===")
    print(res.to_string(index=False))

    # per fraud type on the balanced test set
    p_test = model.predict_proba(X.loc[test.review_id].values)[:, 1]
    test = test.assign(p_fake=p_test, flagged=(p_test >= 0.5).astype(int),
                       decision=decide(p_test))
    per = test.groupby('fraud_type').flagged.mean().round(3)
    print("\n=== share flagged as Fake, by fraud_type (test_balanced) ===")
    print(per.to_string())

    # three-way decision
    print(f"\n=== three-way decision (Genuine < {T_LOW} <= Needs_Verification "
          f"<= {T_HIGH} < Fake), test_balanced ===")
    print(pd.crosstab(test.label, test.decision).to_string())

    p_nv = model.predict_proba(X.loc[nv.review_id].values)[:, 1]
    nv = nv.assign(decision=decide(p_nv))
    print("\n=== Needs_Verification controls -> model decision ===")
    print(pd.crosstab(nv.fraud_type, nv.decision).to_string())

    legit = test[test.notes.astype(str).str.contains('legit_stock_reuse')]
    if len(legit):
        print(f"\nlegit stock reuse (must NOT be flagged): "
              f"{int(legit.flagged.sum())}/{len(legit)} flagged")

    # top features
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    print("\n=== top 10 features ===")
    print(imp.head(10).round(4).to_string())

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f'{MODEL_DIR}/fusion_rf.joblib', compress=3)
    with open(f'{MODEL_DIR}/model_meta.json', 'w') as f:
        json.dump({
            'model': 'RandomForestClassifier(n_estimators=500, min_samples_leaf=2, '
                     'class_weight=balanced)',
            'trained_on': 'train.csv (balanced 50/50, train root photos only)',
            'features': cols,
            'feature_blocks': {k: list(v.columns) for k, v in blocks.items()},
            'cosine_reuse_threshold': F.COSINE_REUSE_THRESHOLD,
            'phash_reuse_threshold': F.PHASH_REUSE_THRESHOLD,
            'decision_bands': {'genuine_below': T_LOW, 'fake_above': T_HIGH},
            'results': rows,
            'per_fraud_type_flag_rate': per.to_dict(),
        }, f, indent=2, default=float)
    res.to_csv(f'{MODEL_DIR}/final_model_results.csv', index=False)
    imp.to_csv(f'{MODEL_DIR}/feature_importance.csv', header=['importance'])
    print(f"\nsaved {MODEL_DIR}/fusion_rf.joblib")


if __name__ == '__main__':
    main()
