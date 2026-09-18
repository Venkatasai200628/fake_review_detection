"""
modality_ablation.py -- answers "is the score coming from text, images, or both?"

Trains the same Random Forest on 7 different feature groups, on the SAME train
photos, and tests all of them on the SAME unseen test photos. Uses the clean
train-roots-only CNN score (cnn_manip_scores_split.csv).

Output: ../out/model/modality_ablation.csv  (+ printed table)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
import features as F

OUT = '../out'
CONFIGS = {
    'TEXT only': ['TEXT'],
    'IMAGE only': ['IMAGE'],
    'BEHAVIOUR only (rating/time)': ['BEHAVIOUR'],
    'GRAPH only': ['GRAPH'],
    'TEXT + IMAGE': ['TEXT', 'IMAGE'],
    'TEXT + BEHAVIOUR + GRAPH (no image)': ['TEXT', 'BEHAVIOUR', 'GRAPH'],
    'ALL (text+image+behaviour+graph)': ['TEXT', 'IMAGE', 'BEHAVIOUR', 'GRAPH'],
}


def main():
    F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
    full = pd.read_csv(f'{OUT}/reviews_full.csv')
    B = F.build_all(full)
    for k in B:
        B[k] = B[k].replace([np.inf, -np.inf], np.nan).fillna(0)
        B[k].index = full.review_id.values
    y = dict(zip(full.review_id, (full.label == 'Fake').astype(int)))
    tr = pd.read_csv(f'{OUT}/train.csv').review_id
    te = pd.read_csv(f'{OUT}/test_balanced.csv')
    ho = pd.read_csv(f'{OUT}/holdout_realistic.csv').review_id
    rows = []
    for name, blocks in CONFIGS.items():
        X = pd.concat([B[b] for b in blocks], axis=1)
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                   class_weight='balanced', n_jobs=-1,
                                   random_state=20260919)
        m.fit(X.loc[tr].values, tr.map(y).values)
        p = m.predict_proba(X.loc[te.review_id].values)[:, 1]
        yt = te.review_id.map(y).values
        yh = (p >= 0.5).astype(int)
        ph = m.predict_proba(X.loc[ho].values)[:, 1]
        r = {'features': name, 'n_features': X.shape[1],
             'accuracy': accuracy_score(yt, yh), 'precision': precision_score(yt, yh),
             'recall': recall_score(yt, yh), 'f1': f1_score(yt, yh),
             'roc_auc': roc_auc_score(yt, p),
             'pr_auc_holdout15': average_precision_score(ho.map(y).values, ph)}
        t = te.assign(flag=yh)
        for ft in ['visual_manipulation', 'text_deception', 'coordinated_reuse', 'genuine']:
            r[f'flagged_{ft}'] = t[t.fraud_type == ft].flag.mean()
        rows.append(r)
    d = pd.DataFrame(rows).round(3)
    pd.set_option('display.width', 250)
    print(d.to_string(index=False))
    d.to_csv(f'{OUT}/model/modality_ablation.csv', index=False)


if __name__ == '__main__':
    main()
