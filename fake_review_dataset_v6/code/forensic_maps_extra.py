"""
forensic_maps_extra.py -- Phase B: three MORE forensic maps, aimed at the weak spot.

The current 3 maps (forensic_maps.py) give the CNN ROC 0.953, but broken down by
manipulation type: splice 0.971, noise 0.970, **copy-move 0.907**. Copy-move is the
one the current channels cannot see: when a region is copied from the SAME photo, its
JPEG history and its noise floor are identical to the rest of the photo, so ELA and the
noise map show nothing. It has to be found by the thing that defines it -- a region that
looks like ANOTHER region of the same photo.

  channel 3  SRM residual energy    high-pass residual (KV kernel) squared. A pasted or
                                    re-rendered region has a different high-frequency
                                    micro-texture from camera pixels.
  channel 4  JPEG grid mismatch     8x8 block-boundary strength minus interior strength.
                                    A patch pasted at a non-multiple-of-8 offset carries
                                    its own, misaligned block grid.
  channel 5  copy-move self-match   for every 16x16 patch, the best match to any OTHER
                                    non-neighbouring patch of the same photo (DCT
                                    descriptor, cosine). A copied region lights up twice.

Same geometry as forensic_maps.py: computed at full resolution, area-averaged to
128x128, divided by the channel's own median, log1p. Nothing here reads a label.

Output: ../out/forensic_maps_extra.npy  (N, 3, 128, 128) float16, rows in
        reviews_full.csv order.
"""
import os
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd
from PIL import Image

OUT = '../out'
SIZE = 128

KV = np.array([[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0

# copy-move search settings
CM_SIDE = 256          # work size
CM_PATCH = 16
CM_STRIDE = 8
CM_MIN_GAP = 4         # ignore matches closer than this many strides (self + neighbours)
CM_KEEP = 6            # DCT coefficients kept per axis
CM_FLAT = 2.0          # patch std below this is featureless -> ignored
CM_SIM = 0.95          # a match must be near-exact to vote
CM_MIN_VOTES = 3       # an offset needs this many agreeing patches to count


def _down(x):
    return cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def _srm(g):
    r = cv2.filter2D(g, cv2.CV_32F, KV)
    return r * r


def _jpeg_grid(g):
    """block-boundary energy minus interior energy, measured locally."""
    dx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    dy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    h, w = g.shape
    cx = np.zeros(w, np.float32); cx[::8] = 1.0
    cy = np.zeros(h, np.float32); cy[::8] = 1.0
    on = dx * cx[None, :] + dy * cy[:, None]
    off = dx * (1 - cx)[None, :] + dy * (1 - cy)[:, None]
    k = (16, 16)
    return np.abs(cv2.blur(on, k) * 8.0 - cv2.blur(off, k) * (8.0 / 7.0))


def _copy_move(g):
    """copy-move detector by OFFSET CONSISTENCY.

    A copied region is an exact pixel copy pasted at one fixed translation, so dozens of
    neighbouring patches all find their best match at the SAME offset (dy, dx). One patch
    matching one other patch means nothing (flat sky matches flat sky); *many patches
    agreeing on one offset* is the signature. Each patch is scored by how many patches
    share its own offset, so genuine photos stay near zero.
    """
    s = cv2.resize(g, (CM_SIDE, CM_SIDE), interpolation=cv2.INTER_AREA)
    n = (CM_SIDE - CM_PATCH) // CM_STRIDE + 1
    D = np.empty((n * n, CM_KEEP * CM_KEEP - 1), np.float32)
    sd = np.empty(n * n, np.float32)
    for i in range(n):
        for j in range(n):
            p = s[i * CM_STRIDE:i * CM_STRIDE + CM_PATCH, j * CM_STRIDE:j * CM_STRIDE + CM_PATCH]
            D[i * n + j] = cv2.dct(p)[:CM_KEEP, :CM_KEEP].ravel()[1:]   # no DC -> brightness invariant
            sd[i * n + j] = p.std()
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-6)
    D[sd < CM_FLAT] = 0.0                    # featureless patches match everything: drop them
    S = D @ D.T
    ii, jj = np.divmod(np.arange(n * n), n)
    near = (np.abs(ii[:, None] - ii[None, :]) < CM_MIN_GAP) & (np.abs(jj[:, None] - jj[None, :]) < CM_MIN_GAP)
    S[near] = -1.0
    best = S.argmax(1)
    sim = S[np.arange(n * n), best]
    di, dj = ii[best] - ii, jj[best] - jj
    ok = sim >= CM_SIM                       # only near-exact matches count
    # how many patches agree on each offset (sign-folded: A->B and B->A are one offset)
    key = np.where((di > 0) | ((di == 0) & (dj > 0)), di * (2 * n) + dj, -di * (2 * n) - dj)
    votes = np.zeros(n * n, np.float32)
    if ok.any():
        u, c = np.unique(key[ok], return_counts=True)
        votes[ok] = c[np.searchsorted(u, key[ok])]
    votes[votes < CM_MIN_VOTES] = 0.0        # a couple of coincidences are not a copied region
    return cv2.resize(votes.reshape(n, n), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def maps_from_image(im):
    """Three extra maps for an in-memory photo (shared with the extension backend)."""
    g = cv2.cvtColor(np.asarray(im.convert('RGB')), cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = []
    for m in (_down(_srm(g)), _down(_jpeg_grid(g)), _copy_move(g)):
        out.append(np.log1p(m / (np.median(m) + 1e-3)))
    return np.stack(out).astype(np.float16)


def maps_for(path):
    return maps_from_image(Image.open(os.path.join(OUT, path)))


def main():
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
        maps = p.map(maps_for, df.image_file.tolist(), chunksize=16)
    arr = np.stack(maps)
    np.save(f'{OUT}/forensic_maps_extra.npy', arr)
    print(f"forensic_maps_extra.npy: {arr.shape}, {arr.nbytes / 1e6:.0f} MB")


if __name__ == '__main__':
    main()
