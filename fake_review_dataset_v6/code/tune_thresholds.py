"""
tune_thresholds.py — pick COSINE_REUSE_THRESHOLD from the data.

MUST be run after recompute_embeddings.py. The current value (0.94) was chosen
for the 64-d placeholder descriptor. CLIP's similarity distribution is different
-- CLIP embeddings of two unrelated product photos of the same category often
sit at 0.8+, far higher than the placeholder produced -- so 0.94 will either
flood the provenance module with false reuse edges or miss real ones.

Method: for every image, compute similarity to all others, split the pairs into
same-root (true reuse) and different-root (not reuse), then choose the threshold
that maximises Youden's J = TPR - FPR. Also reports the value at a fixed 1%
false-positive rate, which is the safer operating point for a provenance
module feeding a fraud score.
"""

import json
import argparse
import numpy as np
import pandas as pd

OUT = '../out'


def load(out):
    df = pd.read_csv(f'{out}/reviews_full.csv')
    try:
        E = np.load(f'{out}/clip_image_embeddings.npy')
        src = 'clip_image_embeddings.npy'
    except FileNotFoundError:
        E = np.stack([np.array(json.loads(e)) for e in df.embedding])
        src = 'embedding column'
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    return df, E, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--max-pairs', type=int, default=400000)
    args = ap.parse_args()

    df, E, src = load(args.out)
    print(f"{len(df)} embeddings, dim={E.shape[1]}, source={src}\n")

    roots = df.root_image_id.values
    n = len(df)
    rng = np.random.default_rng(0)

    same, diff = [], []
    C = E @ E.T
    iu = np.triu_indices(n, k=1)
    sims = C[iu]
    is_same = roots[iu[0]] == roots[iu[1]]

    same = sims[is_same]
    diff = sims[~is_same]
    if len(diff) > args.max_pairs:
        diff = rng.choice(diff, args.max_pairs, replace=False)

    print(f"same-root pairs      n={len(same):7d}  "
          f"mean={same.mean():.3f}  p5={np.percentile(same,5):.3f}")
    print(f"different-root pairs n={len(diff):7d}  "
          f"mean={diff.mean():.3f}  p95={np.percentile(diff,95):.3f}")
    print(f"p5(same) - p95(diff) = {np.percentile(same,5)-np.percentile(diff,95):+.3f}"
          f"   {'CLEAN' if np.percentile(same,5) > np.percentile(diff,95) else 'OVERLAP'}\n")

    grid = np.linspace(min(diff.min(), same.min()), 1.0, 400)
    best_j, best_t = -1, None
    fpr1_t = None
    print(f"{'threshold':>10} {'TPR':>7} {'FPR':>8} {'J':>7}")
    for t in grid:
        tpr = (same >= t).mean()
        fpr = (diff >= t).mean()
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
        if fpr1_t is None and fpr <= 0.01:
            fpr1_t = (t, tpr, fpr)
    for t in np.linspace(best_t - 0.06, min(best_t + 0.06, 1.0), 9):
        print(f"{t:10.3f} {(same>=t).mean():7.3f} "
              f"{(diff>=t).mean():8.4f} {(same>=t).mean()-(diff>=t).mean():7.3f}")

    print(f"\nRECOMMENDED")
    print(f"  max Youden's J : COSINE_REUSE_THRESHOLD = {best_t:.3f}  "
          f"(TPR {(same>=best_t).mean():.3f}, FPR {(diff>=best_t).mean():.4f})")
    if fpr1_t:
        t, tpr, fpr = fpr1_t
        print(f"  at 1% FPR      : COSINE_REUSE_THRESHOLD = {t:.3f}  "
              f"(TPR {tpr:.3f}, FPR {fpr:.4f})   <-- safer for provenance")

    print(f"\nEdit features.py:  COSINE_REUSE_THRESHOLD = {fpr1_t[0]:.3f}"
          if fpr1_t else "\nEdit features.py with the value above")
    print("Then: python crossval.py   and compare against PR-AUC 0.818 +/- 0.022")


if __name__ == '__main__':
    main()
