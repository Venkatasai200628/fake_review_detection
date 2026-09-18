"""
forensic_maps.py -- turn every review photo into three small forensic MAPS
(not summary numbers) for the CNN in cnn_forensics.py.

forensics.py squeezes the whole photo into 13 numbers (mean ELA, block std, ...).
That throws away WHERE the odd region is and what shape it has -- exactly what
a pasted patch or a noise burst looks like. Here we keep a 128x128 picture of
the evidence instead:

  channel 0  ELA level        |photo - photo re-saved at JPEG q90|, local mean
  channel 1  ELA roughness    local standard deviation of that error
  channel 2  noise energy     (photo - 3x3 median)^2, local mean -- a camera
                              gives a fairly even noise floor, a splice or a
                              noise burst does not

Each channel is computed at the photo's full saved resolution (<=1024px) and
only then area-averaged down to 128x128, so fine evidence is summarised, not
blurred away first. Each channel is divided by its own median: the CNN should
see "this region is unlike the rest of THIS photo", not "this photo is bright".

Nothing here reads a label.
Output: ../out/forensic_maps.npy  (N, 3, 128, 128) float16, rows in
        reviews_full.csv order, plus forensic_maps_ids.csv
"""
import io
import os
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd
from PIL import Image

OUT = '../out'
SIZE = 128
ELA_QUALITY = 90


def maps_for(path):
    return maps_from_image(Image.open(os.path.join(OUT, path)))


def maps_from_image(im):
    """Same maps for an in-memory photo (used by the extension backend)."""
    im = im.convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=ELA_QUALITY)
    buf.seek(0)
    re = Image.open(buf).convert('RGB')
    a = np.asarray(im, dtype=np.float32)
    ela = np.abs(a - np.asarray(re, dtype=np.float32)).mean(axis=2)

    g = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
    noise = (g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)) ** 2

    def down(x):
        return cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    m1 = down(ela)
    m2 = np.sqrt(np.maximum(down(ela ** 2) - m1 ** 2, 0))
    m3 = down(noise)
    out = []
    for m in (m1, m2, m3):
        out.append(np.log1p(m / (np.median(m) + 1e-3)))
    return np.stack(out).astype(np.float16)


def main():
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
        maps = p.map(maps_for, df.image_file.tolist(), chunksize=16)
    arr = np.stack(maps)
    np.save(f'{OUT}/forensic_maps.npy', arr)
    df[['review_id']].to_csv(f'{OUT}/forensic_maps_ids.csv', index=False)
    print(f"forensic_maps.npy: {arr.shape}, {arr.nbytes / 1e6:.0f} MB")


if __name__ == '__main__':
    main()
