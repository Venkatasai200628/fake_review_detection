"""
refresh_clip_text.py -- re-encode CLIP *text* vectors after regen_text.py, and nothing else.

WHY THIS EXISTS
clip_text_embeddings.npy and clip_cross_modal.csv are built from review_text. regen_text.py
(v6.2) rewrote every review, but those files were left at their 19 Sep contents, so
`clip_cross_modal` was still comparing each photo against the text of a review that no longer
exists. The same stale-cache mistake hit text_model.py (fixed there with a content hash).

The PHOTOS did not change, so clip_image_embeddings.npy is still correct and is reused as-is
rather than spending minutes re-encoding 3,348 images. Only the text side is recomputed, and
the cross-modal score is the dot product of the two, exactly as recompute_embeddings.py does it.

    python refresh_clip_text.py
"""
import numpy as np
import pandas as pd

import clip_embedding as CE

OUT = '../out'

df = pd.read_csv(f'{OUT}/reviews_full.csv')
img = np.load(f'{OUT}/clip_image_embeddings.npy')
assert len(img) == len(df), 'image embeddings out of sync with reviews_full.csv'

old_t = np.load(f'{OUT}/clip_text_embeddings.npy')
old_xm = np.einsum('ij,ij->i', img, old_t)

print(f'encoding {len(df)} review texts with CLIP ...')
tvecs = CE.encode_text(df.review_text.fillna('').tolist())
xm = np.einsum('ij,ij->i', img, tvecs)

np.save(f'{OUT}/clip_text_embeddings.npy', tvecs)
pd.DataFrame({'review_id': df.review_id, 'clip_cross_modal': xm}).to_csv(
    f'{OUT}/clip_cross_modal.csv', index=False)

print(f'clip_cross_modal  old mean {old_xm.mean():.4f}  new mean {xm.mean():.4f}  '
      f'mean |change| {np.abs(xm - old_xm).mean():.4f}  max |change| {np.abs(xm - old_xm).max():.4f}')
print('wrote clip_text_embeddings.npy and clip_cross_modal.csv')
