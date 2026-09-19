"""
reuse_embedding.py -- the photo fingerprint used to find REUSED photos
(provenance, guide M1/M3).

Chosen in compare_image_models.py (part A): ImageNet ResNet-50 recognises
"the same photo, lightly edited" better than CLIP ViT-B/32 --
at 0.1% false positives against other photos of the SAME product category it
catches 94.7% of copies vs CLIP's 83.0%. CLIP is still used for the
photo-vs-text (cross-modal) check, which only CLIP can do.

Used by BOTH training (features.py reads the saved .npy) and the extension
backend (encodes new photos) -- one function, so the two can never differ.

    python reuse_embedding.py     # writes ../out/reuse_embeddings_resnet50.npy (+ ids)
"""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

OUT = '../out'
MODEL = 'resnet50.a1_in1k'
THRESHOLD = 0.72          # max Youden's J on v6.1 pairs: TPR 0.9997, FPR 0.0015
_m = _tfm = None


def _init():
    global _m, _tfm
    if _m is None:
        import timm
        _m = timm.create_model(MODEL, pretrained=True, num_classes=0).eval()
        cfg = timm.data.resolve_data_config({}, model=_m)
        cfg['input_size'] = (3, 224, 224)
        _tfm = timm.data.create_transform(**cfg)


@torch.no_grad()
def encode(images, bs=32):
    """PIL images -> (n, 2048) L2-normalised float32 vectors."""
    _init()
    out = []
    for i in range(0, len(images), bs):
        x = torch.stack([_tfm(im.convert('RGB')) for im in images[i:i + bs]])
        out.append(torch.nn.functional.normalize(_m(x), dim=1).numpy())
    return np.vstack(out)


def main():
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    cached = f'{OUT}/model_comparison/img_emb_resnet50.npy'   # same encoder, same order
    if os.path.exists(cached):
        E = np.load(cached)
    else:
        E = np.vstack([encode([Image.open(os.path.join(OUT, f)) for f in df.image_file[i:i + 256]])
                       for i in range(0, len(df), 256)])
    assert len(E) == len(df)
    np.save(f'{OUT}/reuse_embeddings_resnet50.npy', E.astype(np.float16))
    df[['review_id']].to_csv(f'{OUT}/reuse_embeddings_ids.csv', index=False)
    print(f"saved reuse_embeddings_resnet50.npy {E.shape}")


if __name__ == '__main__':
    main()
