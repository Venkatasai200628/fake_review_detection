"""
crossval.py — repeated grouped cross-validation with confidence intervals.

Why this replaces the single train/holdout evaluation:
the earlier holdout carried 12 fraud rows. A 0.067 PR-AUC gap between fusion and
the base-paper-equivalent configuration on 12 positives is not a result, it is
noise. This script measures the same comparison across many folds and reports
whether the gap survives.

Design decisions:
  * GroupKFold on root_image_id -- the same rule as PROJECT_SPEC.md 4.9. An
    ungrouped split lets transform variants of one photo straddle the split and
    inflates every score.
  * Repeated with reshuffled group assignment, so the estimate does not depend
    on one partition of the roots.
  * Test folds are subsampled to realistic fraud prevalence (~15%) before
    scoring, so PR-AUC is comparable to the deployment setting rather than to
    the fraud-rich training mix.
  * Paired comparison: fusion and the baseline see IDENTICAL folds, so the
    per-fold difference is paired and a paired t-test is valid.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)

import features as F

OUT = '../out'
N_REPEATS = 5
N_FOLDS = 5
TARGET_PREVALENCE = 0.15
SEED = 20260908

CONFIGS = {
    'text-only':             ['TEXT'],
    'image-forensic-only':   ['IMAGE'],
    'behaviour-only':        ['BEHAVIOUR'],
    'graph-only':            ['GRAPH'],
    'base-paper-equivalent': ['TEXT', 'BEHAVIOUR', 'GRAPH'],
    'FUSION (ours)':         ['TEXT', 'IMAGE', 'BEHAVIOUR', 'GRAPH'],
}


def make_model(kind):
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight='balanced', n_jobs=-1,
                                      random_state=0)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000,
                                            class_weight='balanced'))


def subsample_to_prevalence(y, idx, rng, target=TARGET_PREVALENCE):
    """Keep all genuine test rows, thin the fraud rows to ~target prevalence."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n_pos = int(round(len(neg) * target / (1 - target)))
    n_pos = min(n_pos, len(pos))
    if n_pos < 3:
        return idx
    keep = rng.choice(pos, size=n_pos, replace=False)
    return np.concatenate([neg, keep])


def run():
    full = pd.read_csv(f'{OUT}/reviews_full.csv')
    blocks = F.build_all(full)
    X_all = pd.concat(blocks.values(), axis=1)
    block_cols = {k: list(v.columns) for k, v in blocks.items()}

    keep = full.label != 'Needs_Verification'
    full = full[keep].reset_index(drop=True)
    X_all = X_all[keep.values].reset_index(drop=True)
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = (full.label == 'Fake').astype(int).values
    groups = full.root_image_id.values
    ftype = full.fraud_type.values

    print(f"corpus={len(full)}  fraud={y.sum()} ({100*y.mean():.1f}%)  "
          f"roots={len(set(groups))}  features={X_all.shape[1]}")
    print(f"protocol: {N_REPEATS}x{N_FOLDS} grouped CV, "
          f"test folds thinned to {TARGET_PREVALENCE:.0%} prevalence\n")

    rng = np.random.default_rng(SEED)
    scores = {name: {'pr': [], 'roc': [], 'f1': [], 'prec': [], 'rec': []}
              for name in CONFIGS}
    per_type = {name: {t: [] for t in
                       ['visual_manipulation', 'text_deception',
                        'coordinated_reuse']} for name in CONFIGS}

    uniq = np.array(sorted(set(groups)))
    for rep in range(N_REPEATS):
        perm = rng.permutation(len(uniq))
        remap = {g: i for g, i in zip(uniq, perm)}
        gk = np.array([remap[g] for g in groups])
        for tr_i, te_i in GroupKFold(N_FOLDS).split(X_all, y, groups=gk):
            te_i = subsample_to_prevalence(y, te_i, rng)
            for name, blks in CONFIGS.items():
                cols = [c for b in blks for c in block_cols[b]]
                m = make_model('rf')
                m.fit(X_all.iloc[tr_i][cols].values, y[tr_i])
                p = m.predict_proba(X_all.iloc[te_i][cols].values)[:, 1]
                yt = y[te_i]
                yhat = (p >= 0.5).astype(int)
                s = scores[name]
                s['pr'].append(average_precision_score(yt, p))
                s['roc'].append(roc_auc_score(yt, p))
                s['f1'].append(f1_score(yt, yhat, zero_division=0))
                s['prec'].append(precision_score(yt, yhat, zero_division=0))
                s['rec'].append(recall_score(yt, yhat, zero_division=0))
                for t in per_type[name]:
                    mask = ftype[te_i] == t
                    if mask.sum():
                        per_type[name][t].append(yhat[mask].mean())
        print(f"  repeat {rep+1}/{N_REPEATS} done", flush=True)

    def ci(v):
        v = np.array(v)
        h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        return v.mean(), h

    print(f"\n=== {N_REPEATS}x{N_FOLDS} GROUPED CV (Random Forest), "
          f"mean +/- 95% CI ===")
    rows = []
    for name in CONFIGS:
        s = scores[name]
        r = {'config': name, 'n_folds': len(s['pr'])}
        for k, lab in [('pr', 'PR-AUC'), ('roc', 'ROC-AUC'), ('f1', 'F1'),
                       ('prec', 'Precision'), ('rec', 'Recall')]:
            m, h = ci(s[k])
            r[lab] = f"{m:.3f} +/-{h:.3f}"
            r[lab + '_mean'] = m
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(tab[['config', 'PR-AUC', 'ROC-AUC', 'F1', 'Precision',
               'Recall']].to_string(index=False))

    print("\n=== PER-FRAUD-TYPE DETECTION RATE (mean across folds) ===")
    pt = pd.DataFrame({name: {t: np.mean(v) for t, v in d.items()}
                       for name, d in per_type.items()}).T.round(3)
    print(pt.to_string())

    print("\n=== PAIRED COMPARISON: fusion vs base-paper-equivalent ===")
    a = np.array(scores['FUSION (ours)']['pr'])
    b = np.array(scores['base-paper-equivalent']['pr'])
    d = a - b
    t, p = stats.ttest_rel(a, b)
    m, h = ci(d)
    print(f"  fusion PR-AUC        {a.mean():.3f}")
    print(f"  base-equivalent      {b.mean():.3f}")
    print(f"  paired difference    {m:+.3f} (95% CI {m-h:+.3f} to {m+h:+.3f})")
    print(f"  paired t-test        t={t:.2f}  p={p:.4g}  "
          f"n={len(d)} folds")
    print(f"  fusion wins in       {(d > 0).sum()}/{len(d)} folds")
    verdict = ("SIGNIFICANT -- the gap survives cross-validation"
               if p < 0.05 and m > 0 else
               "NOT SIGNIFICANT -- the gap is inside the noise")
    print(f"  verdict: {verdict}")

    tab.to_csv(f'{OUT}/crossval_results.csv', index=False)
    pt.to_csv(f'{OUT}/crossval_per_fraud_type.csv')
    return tab


if __name__ == '__main__':
    run()
