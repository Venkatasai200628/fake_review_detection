"""
gen_images.py — derived review images from the real root photo pool.

Layer 1 (input)  : 836 real product photos, 15 category folders (the GitHub repo)
Layer 2 (output) : ~3300 derived review images with ground-truth provenance

Every derived image records the root_image_id it came from. That field is
load-bearing for the train/test split (see PROJECT_SPEC.md 4.9) -- transforms of
the same root must never straddle the split.

SWAP POINT FOR REAL DATA: compute_embedding() is the only function that needs to
change when a real CLIP/ViT encoder becomes available. The schema is unaffected.
"""

import os
import io
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import imagehash

WORK_SIZE = 1280     # roots are downscaled to this before transforms
# 1024, not 512. Measured: at 512px every manipulation detector sits at chance
# (ELA 0.513, combined 0.554); at 1024px ELA reaches 0.739 and the combined
# family 0.758. The downscale was averaging away the evidence. Real platforms
# serve product photos at ~1000-1500px, so 1024 is also the more realistic
# choice. See experiment_forensics.py.
OUT_SIZE = 1024      # long edge of saved derived images
OUT_QUALITY = 88

MANIPULATION_THRESHOLD = 0.25   # PROJECT_SPEC.md 5/M4


# ----------------------------------------------------------------------------
# Root pool
# ----------------------------------------------------------------------------

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff')
Image.MAX_IMAGE_PIXELS = None    # some stock photos are ~90MP; they are trusted files


def root_long_edge(path):
    """Long edge in pixels, read from the header only (fast)."""
    with Image.open(path) as im:
        return max(im.size)


def load_root_pool(repo_dir, min_edge=0):
    """
    Scan the Layer 1 repo. Returns ({category: [(root_image_id, path), ...]},
    [skipped (path, long_edge)]).

    v6: only real image files are used (scripts / txt files in the repo are
    ignored) and roots whose long edge is below `min_edge` are skipped, because
    a source smaller than the 1024px output gets upscaled and the manipulation
    detectors cannot work on it (TECHNICAL_REPORT_01 / guide section 9.7).
    Root ids are built from the file name, so adding a new photo later does
    not renumber every other root.
    """
    pool, skipped = {}, []
    for cat in sorted(os.listdir(repo_dir)):
        cdir = os.path.join(repo_dir, cat)
        if not os.path.isdir(cdir) or cat.startswith('.'):
            continue
        files = sorted(f for f in os.listdir(cdir)
                       if not f.startswith('.') and f.lower().endswith(IMAGE_EXTS))
        entries = []
        for f in files:
            path = os.path.join(cdir, f)
            if min_edge and root_long_edge(path) < min_edge:
                skipped.append((path, root_long_edge(path)))
                continue
            stem = os.path.splitext(f)[0]
            root_id = f"{cat.replace(' ', '_')}__{stem}"
            entries.append((root_id, path))
        if entries:
            pool[cat] = entries
    return pool, skipped


def open_root(path):
    """Open any of jpg/png/webp/avif, normalise to RGB and a workable size."""
    im = Image.open(path)
    # For JPEGs, decode directly at reduced scale (DCT scaling). The result is
    # still >= WORK_SIZE and is downscaled with LANCZOS below, so the output is
    # the same kind of image -- this only avoids decoding 90MP files in full.
    try:
        im.draft('RGB', (WORK_SIZE, WORK_SIZE))
    except Exception:
        pass
    im.load()
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.thumbnail((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    return im


# ----------------------------------------------------------------------------
# Benign transforms -- these simulate a different customer re-uploading a
# near-identical shot of the same product. They must NOT by themselves make an
# image look fraudulent.
# ----------------------------------------------------------------------------

def _crop(im, rng):
    w, h = im.size
    f = rng.uniform(0.90, 0.99)
    nw, nh = int(w * f), int(h * f)
    x = rng.randint(0, w - nw)
    y = rng.randint(0, h - nh)
    return im.crop((x, y, x + nw, y + nh))


def _rotate(im, rng):
    return im.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC, expand=False)


def _brightness(im, rng):
    return ImageEnhance.Brightness(im).enhance(rng.uniform(0.82, 1.18))


def _contrast(im, rng):
    return ImageEnhance.Contrast(im).enhance(rng.uniform(0.85, 1.20))


def _saturation(im, rng):
    return ImageEnhance.Color(im).enhance(rng.uniform(0.80, 1.20))


def _blur(im, rng):
    return im.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))


def _recompress(im, rng, quality=None):
    q = quality if quality is not None else rng.randint(55, 80)
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


# NOTE: transform magnitudes are deliberately mild. A customer re-uploading a
# near-identical shot, or a campaign operator lightly editing to evade a hash,
# does not rotate by 7 degrees or crop 18 percent. Aggressive chains destroy the
# pHash fingerprint and collapse the reuse signal into the noise floor -- this
# was measured, not assumed (see validate.py check 1).
BENIGN_OPS = [
    ('crop', _crop),
    ('rotate', _rotate),
    ('brightness', _brightness),
    ('contrast', _contrast),
    ('saturation', _saturation),
    ('blur', _blur),
    ('recompress', _recompress),
]


def apply_transform_chain(im, rng, n_ops=None):
    """Apply a random benign chain. Returns (image, chain_description)."""
    n = n_ops if n_ops is not None else rng.randint(2, 4)
    ops = rng.sample(BENIGN_OPS, n)
    chain = []
    for name, fn in ops:
        im = fn(im, rng)
        chain.append(name)
    return im, '+'.join(chain)


# ----------------------------------------------------------------------------
# Manipulations -- real pixel edits, so ELA and forensic checks detect something
# rather than reading a label.
# ----------------------------------------------------------------------------

def _copy_move(im, rng):
    """Duplicate a patch elsewhere in the same image."""
    w, h = im.size
    pw, ph = int(w * rng.uniform(0.12, 0.22)), int(h * rng.uniform(0.12, 0.22))
    sx, sy = rng.randint(0, w - pw), rng.randint(0, h - ph)
    patch = im.crop((sx, sy, sx + pw, sy + ph))
    dx, dy = rng.randint(0, w - pw), rng.randint(0, h - ph)
    out = im.copy()
    out.paste(patch, (dx, dy))
    return out


def _splice(im, donor, rng):
    """Paste a region from a different image (donor) into this one."""
    w, h = im.size
    pw, ph = int(w * rng.uniform(0.15, 0.28)), int(h * rng.uniform(0.15, 0.28))
    donor = donor.resize((max(pw, 8), max(ph, 8)), Image.LANCZOS)
    dx, dy = rng.randint(0, w - pw), rng.randint(0, h - ph)
    out = im.copy()
    out.paste(donor, (dx, dy))
    return out


def _noise_inject(im, rng):
    """Local gaussian noise burst -- inconsistent noise floor is a classic tell."""
    arr = np.array(im).astype(np.int16)
    h, w, _ = arr.shape
    ph, pw = int(h * 0.25), int(w * 0.25)
    y, x = rng.randint(0, h - ph), rng.randint(0, w - pw)
    noise = np.random.normal(0, 18, (ph, pw, 3))
    arr[y:y + ph, x:x + pw] += noise.astype(np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_manipulation(im, rng, donor=None):
    """
    Apply one manipulation + a real JPEG recompression pass.
    Returns (image, op_name, manipulation_score).
    """
    choices = ['copy_move', 'noise']
    if donor is not None:
        choices.append('splice')
    op = rng.choice(choices)
    if op == 'copy_move':
        out = _copy_move(im, rng)
        score = rng.uniform(0.72, 0.93)
    elif op == 'splice':
        out = _splice(im, donor, rng)
        score = rng.uniform(0.80, 0.97)
    else:
        out = _noise_inject(im, rng)
        score = rng.uniform(0.65, 0.88)
    # NOTE: no recompression here. save_derived() applies exactly one pass to
    # EVERY row, so manipulated and genuine images share an identical
    # compression history. An extra pass here made the classes separable by
    # compression alone -- a pipeline artifact, not forensic evidence.
    return out, op, round(score, 3)


def apply_light_recompression(im, rng):
    """
    NEGATIVE CONTROL (PROJECT_SPEC.md 4.5): image merely re-saved, nothing edited.
    Must score BELOW MANIPULATION_THRESHOLD.
    """
    out = _recompress(im, rng, quality=rng.randint(72, 88))
    return out, 'recompress_only', round(rng.uniform(0.05, 0.20), 3)


# ----------------------------------------------------------------------------
# Forensic descriptors
# ----------------------------------------------------------------------------

def compute_phash(im):
    """Perceptual hash (DCT). Compared later by Hamming distance."""
    return str(imagehash.phash(im, hash_size=8))


def phash_distance(a, b):
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def compute_embedding(im, dim=64):
    """
    PLACEHOLDER EMBEDDING -- swap for CLIP/ViT here. Schema unchanged.

    Deliberately built from structure (gradient orientation on a downsampled
    grayscale grid) plus coarse colour, rather than raw colour histograms, so
    unrelated images do not all collapse to ~0.9 similarity through shared
    background tone.
    """
    g = np.array(im.convert('L').resize((32, 32), Image.LANCZOS)).astype(np.float32)
    gx = np.diff(g, axis=1)[:31, :]
    gy = np.diff(g, axis=0)[:, :31]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.arctan2(gy, gx)

    # 4x4 spatial cells x 8 orientation bins = 128 -> reduced below
    feats = []
    cell = 31 // 4
    for i in range(4):
        for j in range(4):
            m = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            a = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist, _ = np.histogram(a, bins=8, range=(-np.pi, np.pi), weights=m)
            feats.append(hist)
    v = np.concatenate(feats)

    small = np.array(im.resize((8, 8), Image.LANCZOS)).astype(np.float32).ravel() / 255.0
    v = np.concatenate([v / (np.linalg.norm(v) + 1e-8), 0.35 * small])

    rng = np.random.default_rng(0)          # fixed projection, reproducible
    P = rng.normal(size=(v.shape[0], dim))
    e = v @ P
    return e / (np.linalg.norm(e) + 1e-8)


def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------

def save_derived(im, out_dir, image_id, rng=None):
    """One and only one final JPEG pass, applied uniformly to every row."""
    os.makedirs(out_dir, exist_ok=True)
    im = im.copy()
    im.thumbnail((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    path = os.path.join(out_dir, f"{image_id}.jpg")
    im.save(path, format='JPEG', quality=OUT_QUALITY)
    return path
