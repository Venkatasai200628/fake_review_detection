"""
experiment_forensics.py — why did ELA fail, and what recovers the signal?

Hypothesis 1 (RESOLUTION): the dataset saves derived images at 512px. The
manipulation happens at 768px, so the final downscale averages away the
noise-floor and compression-history discontinuities before any detector sees
them. Real platforms serve product images at ~1000-1500px, so 512 is the
unrealistic choice, not the other way round.

Hypothesis 2 (METHOD): ELA is compression-history based, so a uniform re-encode
by the platform erases it. Keypoint-based copy-move detection (ORB) is
geometry based, not compression based, and should survive re-encoding.

Test: build matched genuine/manipulated pairs at 512 and 1024, run three
detector families, compare ROC-AUC. Nothing here touches the main dataset.
"""

import os, io, random, time
import numpy as np
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold

import gen_images as GI
import forensics as FX

SEED = 7
N_PER_CLASS = 90
rng = random.Random(SEED)


# ---------------------------------------------------------- ORB copy-move

def orb_copymove_features(pil_im):
    """
    Keypoint self-matching. A cloned region produces keypoint pairs that are
    descriptor-identical but spatially distant. Survives re-encoding because it
    depends on image structure, not compression history.
    """
    g = np.asarray(pil_im.convert('L'))
    orb = cv2.ORB_create(nfeatures=1500)
    kp, des = orb.detectAndCompute(g, None)
    if des is None or len(kp) < 10:
        return {'orb_pairs': 0.0, 'orb_pair_frac': 0.0,
                'orb_mean_dist': 0.0, 'orb_max_cluster': 0.0}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des, des, k=3)

    pairs, offsets = 0, []
    for m_list in matches:
        for m in m_list[1:]:                      # skip self-match
            if m.distance > 20:
                continue
            p1 = np.array(kp[m.queryIdx].pt)
            p2 = np.array(kp[m.trainIdx].pt)
            d = np.linalg.norm(p1 - p2)
            if d > 40:                            # distant => not just texture
                pairs += 1
                offsets.append(tuple(np.round((p1 - p2) / 8).astype(int)))

    # cloned regions share a CONSISTENT translation offset -- that is the tell
    cluster = 0
    if offsets:
        from collections import Counter
        cluster = Counter(offsets).most_common(1)[0][1]

    return {
        'orb_pairs': float(pairs),
        'orb_pair_frac': pairs / max(len(kp), 1),
        'orb_mean_dist': float(np.mean([np.linalg.norm(
            np.array(kp[m[1].queryIdx].pt) - np.array(kp[m[1].trainIdx].pt))
            for m in matches if len(m) > 1]) if matches else 0),
        'orb_max_cluster': float(cluster),
    }


# ------------------------------------------------------ local colour stats

def local_stats_features(pil_im):
    """
    A spliced region comes from a different photograph: different white balance
    and different local contrast. Measure how much colour statistics vary block
    to block, relative to the image's own level.
    """
    a = np.asarray(pil_im.convert('RGB')).astype(np.float32)
    h, w, _ = a.shape
    bs = max(32, min(h, w) // 8)
    stats = []
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            b = a[y:y + bs, x:x + bs]
            stats.append([b[..., 0].mean() - b[..., 2].mean(),   # R-B, white balance
                          b.std(),
                          b.mean()])
    if len(stats) < 4:
        return {'loc_wb_cv': 0.0, 'loc_std_cv': 0.0, 'loc_wb_outlier': 0.0}
    S = np.array(stats)
    wb, sd = S[:, 0], S[:, 1]
    return {
        'loc_wb_cv': float(wb.std() / (abs(wb.mean()) + 1e-3)),
        'loc_std_cv': float(sd.std() / (sd.mean() + 1e-6)),
        'loc_wb_outlier': float((np.abs(wb - wb.mean()) > 2 * wb.std() + 1e-6).mean()),
    }


# ------------------------------------------------------------- build pairs

def build_sample(out_size):
    pool = GI.load_root_pool('../repo')
    roots = [(rid, p) for cat in pool for rid, p in pool[cat]]
    rng.shuffle(roots)
    rows, labels, groups = [], [], []

    def save_and_read(im):
        im = im.copy()
        im.thumbnail((out_size, out_size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=88)   # platform re-encode
        buf.seek(0)
        out = Image.open(buf); out.load()
        return out

    for i in range(N_PER_CLASS):
        rid, path = roots[i % len(roots)]
        base = GI.open_root(path)
        base = base.resize((min(1400, base.size[0] * 2), min(1400, base.size[1] * 2)),
                           Image.LANCZOS) if max(base.size) < out_size else base

        # genuine -- MUST receive an equivalent extra compression pass.
        # apply_manipulation() ends with a JPEG save at q70-85. Without a
        # matching pass here, the classes differ in compression history alone
        # and every detector learns that artifact instead of the manipulation.
        # This confound produced ROC-AUC 0.198 (i.e. 0.802 inverted) before the
        # fix.
        im, _ = GI.apply_transform_chain(base, rng)
        im = GI._recompress(im, rng, quality=rng.randint(70, 85))
        rows.append(save_and_read(im)); labels.append(0); groups.append(rid)

        # manipulated
        donor_path = roots[(i + 17) % len(roots)][1]
        im2, _ = GI.apply_transform_chain(base, rng, n_ops=2)
        im2, _, _ = GI.apply_manipulation(im2, rng, donor=GI.open_root(donor_path))
        rows.append(save_and_read(im2)); labels.append(1); groups.append(rid)

    return rows, np.array(labels), np.array(groups)


def evaluate(images, y, groups, families):
    results = {}
    for fam_name, fns in families.items():
        X = []
        for im in images:
            d = {}
            for fn in fns:
                d.update(fn(im))
            X.append(list(d.values()))
        X = np.array(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        clf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                     random_state=0)
        # GROUPED folds: every root contributes one genuine AND one
        # manipulated sample, so an ungrouped split lets the model identify the
        # root in training and infer the opposite label at test time. That
        # produced systematically inverted AUC (0.185-0.28). Same root-leakage
        # rule as PROJECT_SPEC.md 4.9.
        s = cross_val_score(clf, X, y, groups=groups, cv=GroupKFold(5),
                            scoring='roc_auc')
        results[fam_name] = (s.mean(), s.std())
    return results


FAMILIES = {
    'ELA only':            [FX.ela_features],
    'noise only':          [FX.noise_features],
    'ORB copy-move':       [orb_copymove_features],
    'local colour stats':  [local_stats_features],
    'ALL combined':        [FX.ela_features, FX.noise_features,
                            orb_copymove_features, local_stats_features],
}

if __name__ == '__main__':
    for size in (512, 1024):
        t0 = time.time()
        imgs, y, grp = build_sample(size)
        print(f"\n=== resolution {size}px  (n={len(y)}, {int(y.sum())} manipulated) ===")
        for name, (m, s) in evaluate(imgs, y, grp, FAMILIES).items():
            bar = '#' * int(max(0, (m - 0.5)) * 60)
            print(f"  {name:20s} ROC-AUC {m:.3f} +/- {s:.3f}  {bar}")
        print(f"  ({time.time()-t0:.0f}s)")
