"""
forensics.py — manipulation evidence computed FROM PIXELS.

Replaces the leaking `manipulation_score` column, which was the generator's
ground truth rather than a measurement. Nothing here reads any label.

Two families of evidence, both classical and both appropriate for a course
project (PROJECT_SPEC.md section 5 / M4):

  ELA (Error Level Analysis)
      Recompress at a fixed JPEG quality and take the absolute difference.
      Regions that have been through a different compression history than the
      rest of the frame -- a spliced-in patch, a cloned region re-saved once
      more -- settle at a different error level than their surroundings.
      The signal is not the mean error, it is the INCONSISTENCY of the error
      across the frame.

  Noise residual
      Subtract a median-filtered version to isolate high-frequency residual,
      then measure how much local noise energy varies block to block. A camera
      produces a roughly uniform noise floor; a splice from another source, or
      a synthetic noise burst, does not.

Both are computed on the saved 512px JPEGs, i.e. exactly what a deployed system
would receive.
"""

import numpy as np
from PIL import Image, ImageChops, ImageFilter
import io

ELA_QUALITY = 90
BLOCK = 32


def _blocks(a, size=BLOCK):
    h, w = a.shape[:2]
    h2, w2 = (h // size) * size, (w // size) * size
    if h2 < size or w2 < size:
        return a.reshape(1, -1)
    a = a[:h2, :w2]
    return (a.reshape(h2 // size, size, w2 // size, size)
             .transpose(0, 2, 1, 3)
             .reshape(-1, size * size))


def ela_features(im, quality=ELA_QUALITY):
    """Error Level Analysis statistics. Higher inconsistency => more suspicious."""
    im = im.convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    recompressed.load()

    diff = ImageChops.difference(im, recompressed)
    a = np.asarray(diff).astype(np.float32).mean(axis=2)   # per-pixel error

    blk = _blocks(a)
    bmean = blk.mean(axis=1)
    if bmean.size < 2:
        bmean = np.array([a.mean(), a.mean()])

    denom = bmean.mean() + 1e-6
    return {
        'ela_mean': float(a.mean()),
        'ela_p99': float(np.percentile(a, 99)),
        'ela_block_std': float(bmean.std()),
        # scale-free: how uneven the error field is relative to its own level.
        # This is the discriminative one -- a uniformly bright or dark image
        # shifts ela_mean without implying manipulation.
        'ela_block_cv': float(bmean.std() / denom),
        'ela_block_max_ratio': float(bmean.max() / denom),
        'ela_block_skew': float(((bmean - bmean.mean()) ** 3).mean() /
                                (bmean.std() ** 3 + 1e-6)),
        'ela_hot_block_frac': float((bmean > bmean.mean() + 2 * bmean.std()).mean()),
    }


def noise_features(im):
    """Noise-residual inconsistency across the frame."""
    g = im.convert('L')
    med = g.filter(ImageFilter.MedianFilter(size=3))
    res = np.asarray(g).astype(np.float32) - np.asarray(med).astype(np.float32)

    blk = _blocks(res)
    bvar = blk.var(axis=1)
    if bvar.size < 2:
        bvar = np.array([res.var(), res.var()])
    denom = bvar.mean() + 1e-6

    return {
        'noise_mean_var': float(bvar.mean()),
        'noise_var_cv': float(bvar.std() / denom),
        'noise_max_ratio': float(bvar.max() / denom),
        'noise_hot_frac': float((bvar > bvar.mean() + 2 * bvar.std()).mean()),
    }


def copy_move_score(im, size=16, stride=16, top_k=5):
    """
    Cheap copy-move indicator: hash non-overlapping blocks by their coarse
    DCT-free signature and count near-identical non-adjacent pairs. A cloned
    region produces duplicate blocks far apart in the frame.
    """
    g = np.asarray(im.convert('L').resize((256, 256), Image.LANCZOS)).astype(np.float32)
    sigs, coords = [], []
    for y in range(0, 256 - size + 1, stride):
        for x in range(0, 256 - size + 1, stride):
            b = g[y:y + size, x:x + size]
            if b.std() < 3:               # skip flat blocks; they match trivially
                continue
            v = (b - b.mean()) / (b.std() + 1e-6)
            sigs.append(v.ravel()[::4])
            coords.append((y, x))
    if len(sigs) < 4:
        return {'cm_pairs': 0.0, 'cm_best_sim': 0.0}

    S = np.stack(sigs)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    M = S @ S.T
    np.fill_diagonal(M, -1)

    pairs, best = 0, 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if dist < 48:                 # adjacent blocks: ignore
                continue
            if M[i, j] > 0.97:
                pairs += 1
                best = max(best, float(M[i, j]))
    return {'cm_pairs': float(pairs), 'cm_best_sim': best}


def analyse(path):
    im = Image.open(path)
    im.load()
    out = {}
    out.update(ela_features(im))
    out.update(noise_features(im))
    out.update(copy_move_score(im))
    return out
