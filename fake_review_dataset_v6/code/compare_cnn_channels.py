"""
compare_cnn_channels.py -- Phase B: do the 3 EXTRA forensic maps make the edit
detector better, and does a bigger CNN help?

Current detector: 3 maps (ELA level, ELA roughness, noise energy) -> ROC 0.953,
but copy-move only 0.907. forensic_maps_extra.py adds SRM residual, JPEG grid
mismatch and a copy-move offset-consistency map. This trains the same CNN on
different channel sets and measures, so the choice is made by numbers, not by
whether the idea sounded good.

Protocol is exactly cnn_forensics.py part B: train on TRAIN roots only, score
unseen TEST roots. Every variant is run with 2 seeds and averaged, because the
differences we are looking for are smaller than seed noise.

Output: ../out/model_comparison/cnn_channels.csv / .md
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

import os

import cnn_forensics as C

OUT = '../out'
DST = f'{OUT}/model_comparison'
SEEDS = [20260919, 771]
torch.set_num_threads(8)


class CNN(nn.Module):
    """cnn_forensics.ForensicCNN, but the input channel count and width are arguments."""

    def __init__(self, in_ch, w=1):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        c = [int(16 * w), int(32 * w), int(64 * w), int(96 * w)]
        self.f = nn.Sequential(block(in_ch, c[0]), nn.MaxPool2d(2),
                               block(c[0], c[1]), nn.MaxPool2d(2),
                               block(c[1], c[2]), nn.MaxPool2d(2),
                               block(c[2], c[3]))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(c[3] * 2, 1))

    def forward(self, x):
        h = self.f(x)
        return self.head(torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))], dim=1)).squeeze(1)


def train(X, y, seed, chans, rows, w=1):
    """X is a float16 memmap (N,6,128,128); only `chans` are used and each BATCH is
    converted to float32, so peak memory is one batch, not the whole array. This machine
    has ~1 GB free, and converting the array up front (1.6 GB) is what killed the first run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = CNN(len(chans), w)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=C.EPOCHS * ((len(y) + C.BATCH - 1) // C.BATCH))
    pos = y.sum()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / max(pos, 1)))
    yt = torch.from_numpy(y.astype(np.float32))
    m.train()
    for _ in range(C.EPOCHS):
        perm = np.random.permutation(len(y))
        for i in range(0, len(y), C.BATCH):
            j = np.sort(perm[i:i + C.BATCH])
            xb = torch.from_numpy(np.asarray(X[rows[j]][:, chans], dtype=np.float32))
            opt.zero_grad()
            lossf(m(C.augment(xb)), yt[j]).backward()
            opt.step()
            sched.step()
    return m


@torch.no_grad()
def predict(m, X, chans, rows):
    m.eval()
    out = []
    for i in range(0, len(rows), 128):
        b = rows[i:i + 128]
        xb = torch.from_numpy(np.asarray(X[b][:, chans], dtype=np.float32))
        out.append((sum(torch.sigmoid(m(torch.rot90(xb, k, dims=[2, 3]))) for k in range(4)) / 4).numpy())
    return np.concatenate(out)


df = pd.read_csv(f'{OUT}/reviews_full.csv')
X = np.load(f'{OUT}/forensic_maps_all.npy', mmap_mode='r')      # 6 channels, read from disk
y = (df.fraud_type == 'visual_manipulation').astype(int).values
ops = df.transform_chain.str.extract(r'(copy_move|splice|noise)$')[0].fillna('').values
tr = (df.split == 'train').values
print(f'{len(y)} photos, {y.sum()} edited; train {tr.sum()}, test {(~tr).sum()}')

ONLY = sys.argv[1:]
VARIANTS = [
    ('3 ch: ELA, ELA-std, noise (current)', [0, 1, 2], 1),
    ('4 ch: + SRM residual', [0, 1, 2, 3], 1),
    ('5 ch: + SRM + JPEG grid', [0, 1, 2, 3, 4], 1),
    ('6 ch: all (+ copy-move votes)', [0, 1, 2, 3, 4, 5], 1),
    ('6 ch, 1.5x wider CNN', [0, 1, 2, 3, 4, 5], 1.5),
]

tr_rows, te_rows = np.where(tr)[0], np.where(~tr)[0]
rows = []
for name, chans, w in VARIANTS:
    if ONLY and not any(o in name for o in ONLY):
        continue
    t0 = time.time()
    per_seed = []
    for sd in SEEDS:
        m = train(X, y[tr], sd + 200, chans, tr_rows, w)
        per_seed.append(predict(m, X, chans, te_rows))
    s = np.mean(per_seed, axis=0)
    yt, ot = y[~tr], ops[~tr]
    r = {'variant': name, 'channels': len(chans), 'width': w,
         'roc_auc_test': roc_auc_score(yt, s), 'ap_test': average_precision_score(yt, s),
         'roc_seed1': roc_auc_score(yt, per_seed[0]), 'roc_seed2': roc_auc_score(yt, per_seed[1])}
    neg = s[yt == 0]
    for op in ['copy_move', 'splice', 'noise']:
        pos = s[(yt == 1) & (ot == op)]
        r[f'roc_{op}'] = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])
    thr = np.quantile(s[yt == 0], 0.95)          # threshold at 5% false alarms, comparable across variants
    r['caught_at_5pct_fa'] = (s[yt == 1] >= thr).mean()
    r['seconds'] = round(time.time() - t0)
    rows.append(r)
    print(f"  {name:38s} ROC {r['roc_auc_test']:.4f} (seeds {r['roc_seed1']:.3f}/{r['roc_seed2']:.3f})  "
          f"copy-move {r['roc_copy_move']:.3f}  splice {r['roc_splice']:.3f}  noise {r['roc_noise']:.3f}  "
          f"caught@5%FA {r['caught_at_5pct_fa']:.1%}  [{r['seconds']}s]", flush=True)
    old = pd.read_csv(f'{DST}/cnn_channels.csv') if os.path.exists(f'{DST}/cnn_channels.csv') else pd.DataFrame()
    if len(old):
        old = old[old.variant != name]
    pd.concat([old, pd.DataFrame([r])], ignore_index=True).to_csv(f'{DST}/cnn_channels.csv', index=False)

d = pd.read_csv(f'{DST}/cnn_channels.csv')
md = ['# Phase B: which forensic maps should the edit detector see?\n',
      'Same CNN, same training rows, only the input channels change. Each row is the average of 2 seeds; '
      'the two individual seed scores are shown so you can see whether a difference is bigger than seed noise.\n',
      '| input channels | ROC (test photos) | seed 1 / seed 2 | copy-move | splice | noise | caught @5% false alarms |',
      '|---|---|---|---|---|---|---|']
for r in d.to_dict('records'):
    md.append(f"| {r['variant']} | **{r['roc_auc_test']:.4f}** | {r['roc_seed1']:.3f} / {r['roc_seed2']:.3f} | "
              f"{r['roc_copy_move']:.3f} | {r['roc_splice']:.3f} | {r['roc_noise']:.3f} | {r['caught_at_5pct_fa']:.1%} |")
open(f'{DST}/cnn_channels.md', 'w', encoding='utf-8').write('\n'.join(md))
print('\n'.join(md))
