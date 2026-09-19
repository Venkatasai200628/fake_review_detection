"""
compare_image_models.py -- which IMAGE models are best for our project?

PART A: photo-reuse embedding (provenance, guide M1/M3)
  Task: "is this the same root photo (lightly edited copy) or a different one?"
  Candidates: CLIP ViT-B/32 (current) | DINOv2 ViT-S/14 | DINOv2 ViT-B/14 | ResNet-50 (ImageNet)
  Metrics over all photo pairs: ROC-AUC, and -- the hard part -- true-positive
  rate at 0.1% false-positive rate against DIFFERENT photos of the SAME category.

PART B: edit detector on forensic maps (guide M4)
  Task: "was this photo edited?" -- train on train photos, test on unseen test photos
  Candidates: small CNN (current) | ResNet-18 (ImageNet-pretrained) | EfficientNet-B0 (pretrained)

Output: ../out/model_comparison/image_reuse_models.csv, image_edit_models.csv
    python compare_image_models.py            (both parts)
    python compare_image_models.py reuse      (part A only)
    python compare_image_models.py edit       (part B only)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score

OUT = '../out'
DST = f'{OUT}/model_comparison'
os.makedirs(DST, exist_ok=True)
torch.set_num_threads(8)
full = pd.read_csv(f'{OUT}/reviews_full.csv')


# =============================================================== PART A
def encode_timm(name, size=224, bs=32):
    import timm
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size) \
        if 'dinov2' in name else timm.create_model(name, pretrained=True, num_classes=0)
    m.eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    cfg['input_size'] = (3, size, size)
    tfm = timm.data.create_transform(**cfg)
    vecs = []
    with torch.no_grad():
        for i in range(0, len(full), bs):
            x = torch.stack([tfm(Image.open(os.path.join(OUT, f)).convert('RGB'))
                             for f in full.image_file.values[i:i + bs]])
            vecs.append(torch.nn.functional.normalize(m(x), dim=1).numpy())
            if i % (bs * 20) == 0:
                print(f"    {name}: {i}/{len(full)}", flush=True)
    return np.vstack(vecs)


def reuse_metrics(E):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    root = full.root_image_id.values
    cat = full.category.values
    iu = np.triu_indices(len(full), 1)
    s = S[iu]
    same = root[iu[0]] == root[iu[1]]
    samecat = cat[iu[0]] == cat[iu[1]]
    hard = (~same) & samecat
    auc_all = roc_auc_score(same, s)
    auc_hard = roc_auc_score(np.r_[np.ones(same.sum()), np.zeros(hard.sum())], np.r_[s[same], s[hard]])
    thr = np.quantile(s[hard], 0.999)                  # 0.1% false positives among hard negatives
    return {'roc_auc_all_pairs': auc_all, 'roc_auc_vs_same_category': auc_hard,
            'tpr_at_0.1pct_fpr_same_category': (s[same] > thr).mean(),
            'mean_sim_same_photo': s[same].mean(), 'mean_sim_other_photo_same_category': s[hard].mean()}


def part_a():
    rows = []
    cands = [('CLIP ViT-B/32 (current)', None),
             ('DINOv2 ViT-S/14', 'vit_small_patch14_dinov2.lvd142m'),
             ('DINOv2 ViT-B/14', 'vit_base_patch14_dinov2.lvd142m'),
             ('ResNet-50 ImageNet', 'resnet50.a1_in1k')]
    for label, name in cands:
        t0 = time.time()
        if name is None:
            E = np.load(f'{OUT}/clip_image_embeddings.npy')
        else:
            E = encode_timm(name)
            np.save(f"{DST}/img_emb_{name.split('.')[0]}.npy", E)
        r = {'model': label, **reuse_metrics(E), 'dim': E.shape[1], 'seconds': round(time.time() - t0)}
        print(f"  {label:26s} AUC(all) {r['roc_auc_all_pairs']:.4f}  AUC(vs same category) "
              f"{r['roc_auc_vs_same_category']:.4f}  TPR@0.1%FPR {r['tpr_at_0.1pct_fpr_same_category']:.3f}", flush=True)
        rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(f'{DST}/image_reuse_models.csv', index=False)


# =============================================================== PART B
def part_b():
    import cnn_forensics as CF
    import timm
    X = np.load(f'{OUT}/forensic_maps.npy')
    y = (full.fraud_type == 'visual_manipulation').astype(int).values
    ops = full.transform_chain.str.extract(r'(copy_move|splice|noise)$')[0].fillna('').values
    tr = (full.split == 'train').values

    class Pre(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.m = timm.create_model(name, pretrained=True, num_classes=1)

        def forward(self, x):
            return self.m(x).squeeze(1)

    cands = [('small CNN (current)', CF.ForensicCNN, 14, 2e-3),
             ('ResNet-18 pretrained', lambda: Pre('resnet18.a1_in1k'), 8, 5e-4),
             ('EfficientNet-B0 pretrained', lambda: Pre('efficientnet_b0.ra_in1k'), 8, 1e-3)]
    rows = []
    for label, make, epochs, lr in cands:
        t0 = time.time()
        CF.EPOCHS = epochs
        orig = CF.ForensicCNN
        CF.ForensicCNN = make
        # same training loop as cnn_forensics.train, but with this model and lr
        torch.manual_seed(0); np.random.seed(0)
        m = make()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        n_steps = int(epochs * ((tr.sum() + CF.BATCH - 1) // CF.BATCH))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=n_steps)
        pos = y[tr].sum()
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((tr.sum() - pos) / pos))
        Xt = torch.from_numpy(X[tr].astype(np.float32)); yt = torch.from_numpy(y[tr].astype(np.float32))
        m.train()
        for ep in range(epochs):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), CF.BATCH):
                b = perm[i:i + CF.BATCH]
                opt.zero_grad()
                lossf(m(CF.augment(Xt[b])), yt[b]).backward()
                opt.step(); sched.step()
            print(f"    {label}: epoch {ep + 1}/{epochs} ({time.time() - t0:.0f}s)", flush=True)
        CF.ForensicCNN = orig
        s = CF.predict(m, X[~tr])
        yt_, op_ = y[~tr], ops[~tr]
        neg = s[yt_ == 0]
        r = {'model': label, 'roc_auc_test': roc_auc_score(yt_, s)}
        for op in ['copy_move', 'splice', 'noise']:
            p = s[(yt_ == 1) & (op_ == op)]
            r[f'roc_auc_{op}'] = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(p))], np.r_[neg, p])
        r['caught_at_0.5'] = (s[yt_ == 1] >= 0.5).mean()
        r['false_alarm_at_0.5'] = (neg >= 0.5).mean()
        r['params_M'] = sum(p.numel() for p in m.parameters()) / 1e6
        r['seconds'] = round(time.time() - t0)
        print(f"  {label:28s} ROC {r['roc_auc_test']:.3f}  copy-move {r['roc_auc_copy_move']:.3f}  "
              f"splice {r['roc_auc_splice']:.3f}  noise {r['roc_auc_noise']:.3f}  "
              f"caught {r['caught_at_0.5']:.0%} / false alarms {r['false_alarm_at_0.5']:.0%}", flush=True)
        rows.append(r)
        pd.DataFrame(rows).round(4).to_csv(f'{DST}/image_edit_models.csv', index=False)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if what in ('both', 'reuse'):
        print('PART A -- photo-reuse embeddings'); part_a()
    if what in ('both', 'edit'):
        print('\nPART B -- edit detectors on forensic maps'); part_b()
