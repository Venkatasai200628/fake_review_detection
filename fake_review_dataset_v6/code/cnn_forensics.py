"""
cnn_forensics.py -- a small CNN that learns to spot EDITED photos from their
forensic maps (forensic_maps.py). Its output, "probability this photo was
edited", becomes one new IMAGE feature: `img_cnn_manip`.

WHY A CNN (guide section 12.5)
The 13 hand-made ELA/noise numbers only say "how uneven is the error field
overall". A CNN looks at the 128x128 MAP and can learn the SHAPE of evidence:
a rectangular patch with a different error level (splice), a square burst of
noise, etc. Global max-pooling at the end means "is there ANY odd region
anywhere", which is what a local edit looks like.

TRAINING TARGET
"Was this photo edited?" = fraud_type == 'visual_manipulation'. This is a
sub-task label, used ONLY to train this image detector, never given to the
fusion model directly. Honest caveat (guide section 11 item 1): the same code
made the edits, so the CNN learns OUR three edit types (copy-move, splice,
noise). Report its numbers as a controlled check, not as real-world tamper
detection.

NO LEAKAGE: every score is OUT-OF-FOLD, grouped by root photo
  cnn_manip_scores.csv        5-fold grouped cross-fitting over all rows;
                              used by crossval.py (a row is always scored by
                              a CNN that never saw its root photo)
  cnn_manip_scores_split.csv  CNN trained on TRAIN roots only (train rows get
                              inner 5-fold out-of-fold scores, test rows are
                              scored by a CNN trained on all train roots);
                              used by train_model.py -> clean holdout numbers
Run after forensic_maps.py:   python cnn_forensics.py
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = '../out'
SEED = 20260919
EPOCHS = 14
BATCH = 64
torch.set_num_threads(8)


class ForensicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True))
        self.f = nn.Sequential(block(3, 16), nn.MaxPool2d(2),     # 64
                               block(16, 32), nn.MaxPool2d(2),    # 32
                               block(32, 64), nn.MaxPool2d(2),    # 16
                               block(64, 96))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(96 * 2, 1))

    def forward(self, x):
        h = self.f(x)
        h = torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))], dim=1)
        return self.head(h).squeeze(1)


def augment(x):
    # flips and 90-degree turns: an edit is an edit whichever way up
    if np.random.rand() < 0.5:
        x = torch.flip(x, dims=[3])
    if np.random.rand() < 0.5:
        x = torch.flip(x, dims=[2])
    return torch.rot90(x, k=np.random.randint(4), dims=[2, 3])


def train(X, y, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = ForensicCNN()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=EPOCHS * ((len(X) + BATCH - 1) // BATCH))
    pos = y.sum()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / max(pos, 1)))
    Xt = torch.from_numpy(X.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    m.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(m(augment(Xt[b])), yt[b])
            loss.backward()
            opt.step()
            sched.step()
    return m


@torch.no_grad()
def predict(m, X):
    m.eval()
    Xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(Xt), 256):
        xb = Xt[i:i + 256]
        # test-time augmentation: average over the 4 rotations
        p = sum(torch.sigmoid(m(torch.rot90(xb, k, dims=[2, 3]))) for k in range(4)) / 4
        out.append(p.numpy())
    return np.concatenate(out)


def oof(X, y, groups, seed, n=5):
    s = np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(n).split(X, y, groups)):
        s[te] = predict(train(X[tr], y[tr], seed + k), X[te])
    return s


def report(name, y, s, ops):
    print(f"  {name:34s} ROC-AUC {roc_auc_score(y, s):.3f}   AP {average_precision_score(y, s):.3f}")
    neg = s[y == 0]
    for op in ['copy_move', 'splice', 'noise']:
        pos = s[(y == 1) & (ops == op)]
        if len(pos):
            auc = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])
            print(f"      {op:10s} vs clean photos   ROC-AUC {auc:.3f}  (n={len(pos)})")


def main():
    t0 = time.time()
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    ids = pd.read_csv(f'{OUT}/forensic_maps_ids.csv').review_id
    assert (ids.values == df.review_id.values).all(), 'maps out of sync -- rerun forensic_maps.py'
    X = np.load(f'{OUT}/forensic_maps.npy')
    y = (df.fraud_type == 'visual_manipulation').astype(int).values
    g = df.root_image_id.values
    ops = df.transform_chain.str.extract(r'(copy_move|splice|noise)$')[0].fillna('').values
    print(f"{len(y)} photos, {y.sum()} edited; maps {X.shape}")

    # ---- A: cross-fitted scores over everything (for crossval.py) ----------
    s_all = oof(X, y, g, SEED)
    pd.DataFrame({'review_id': df.review_id, 'img_cnn_manip': s_all}).to_csv(
        f'{OUT}/cnn_manip_scores.csv', index=False)
    print(f"\n[A] cross-fitted, all rows ({time.time() - t0:.0f}s)")
    report('CNN on forensic maps', y, s_all, ops)

    # baseline: the 13 hand-made forensic numbers, same folds, same target
    from sklearn.ensemble import RandomForestClassifier
    Fh = pd.read_csv(f'{OUT}/forensic_features.csv').set_index('review_id').loc[df.review_id].values
    s_h = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Fh, y, g):
        rf = RandomForestClassifier(400, min_samples_leaf=2, class_weight='balanced',
                                    n_jobs=-1, random_state=0).fit(Fh[tr], y[tr])
        s_h[te] = rf.predict_proba(Fh[te])[:, 1]
    report('13 hand-made ELA/noise numbers', y, s_h, ops)

    # ---- B: clean split (for train_model.py) --------------------------------
    tr = (df.split == 'train').values
    s_split = np.zeros(len(y))
    s_split[tr] = oof(X[tr], y[tr], g[tr], SEED + 100)
    final = train(X[tr], y[tr], SEED + 200)
    s_split[~tr] = predict(final, X[~tr])
    pd.DataFrame({'review_id': df.review_id, 'img_cnn_manip': s_split}).to_csv(
        f'{OUT}/cnn_manip_scores_split.csv', index=False)
    torch.save(final.state_dict(), f'{OUT}/model/forensic_cnn.pt')
    print(f"\n[B] trained on train roots only, scored on unseen test roots ({time.time() - t0:.0f}s)")
    report('CNN, test photos', y[~tr], s_split[~tr], ops[~tr])
    print(f"\nsaved cnn_manip_scores.csv, cnn_manip_scores_split.csv, model/forensic_cnn.pt")


if __name__ == '__main__':
    main()
