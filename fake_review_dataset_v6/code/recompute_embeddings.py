"""
recompute_embeddings.py — swap the embedding column in place, on the existing
images, without regenerating the dataset.

WHY IN PLACE rather than re-running build_dataset.py:
regenerating would redraw every random transform, every reviewer assignment and
every timestamp, so a before/after comparison would be confounded by dataset
churn. Recomputing embeddings on the SAME image files changes exactly one
variable, which makes the comparison against the recorded baseline
(PR-AUC 0.818 +/- 0.022, TECHNICAL_REPORT_02) exact.

Usage:
    python recompute_embeddings.py                 # CLIP, all CSVs
    python recompute_embeddings.py --dry-run       # check paths only

Writes:
    reviews_full.csv, train.csv, holdout_realistic.csv   (embedding column)
    clip_text_embeddings.npy                             (for cross-modal work)
    embeddings_backup_placeholder.csv                    (rollback)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image

OUT = '../out'
CSVS = ['reviews_full.csv', 'reviews_balanced.csv', 'train.csv',
        'test_balanced.csv', 'holdout_realistic.csv',
        'needs_verification_controls.csv']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-text', action='store_true')
    args = ap.parse_args()

    full_path = os.path.join(args.out, 'reviews_full.csv')
    df = pd.read_csv(full_path)
    print(f"{len(df)} rows, {df.image_id.nunique()} images")

    missing = [f for f in df.image_file
               if not os.path.exists(os.path.join(args.out, f))][:5]
    if missing:
        print(f"ERROR: image files not found, e.g. {missing}")
        print("Regenerate images first: python build_dataset.py")
        sys.exit(1)
    print("all image files present")

    if args.dry_run:
        print("dry run OK -- rerun without --dry-run to encode")
        return

    # rollback copy of the old embeddings
    backup = os.path.join(args.out, 'embeddings_backup_placeholder.csv')
    if not os.path.exists(backup):
        df[['review_id', 'embedding']].to_csv(backup, index=False)
        print(f"placeholder embeddings backed up -> {os.path.basename(backup)}")

    import clip_embedding as CE

    # ---- image embeddings -------------------------------------------
    t0 = time.time()
    vecs = np.zeros((len(df), 512), dtype=np.float32)
    B = args.batch_size
    for i in range(0, len(df), B):
        chunk = df.iloc[i:i + B]
        ims = []
        for f in chunk.image_file:
            im = Image.open(os.path.join(args.out, f))
            im.load()
            ims.append(im.convert('RGB'))
        vecs[i:i + len(ims)] = CE.encode_images(ims, batch_size=B)
        if i and i % (B * 10) == 0:
            el = time.time() - t0
            print(f"  {i}/{len(df)}  {el:.0f}s  "
                  f"(eta {el / i * (len(df) - i):.0f}s)", flush=True)
    print(f"image encoding done in {time.time()-t0:.0f}s")

    np.save(os.path.join(args.out, 'clip_image_embeddings.npy'), vecs)
    emb_map = dict(zip(df.review_id,
                       [json.dumps([round(float(x), 5) for x in v])
                        for v in vecs]))

    # ---- text embeddings (same space -- enables cross-modal score) ---
    if not args.skip_text:
        t1 = time.time()
        tvecs = CE.encode_text(df.review_text.fillna('').tolist())
        np.save(os.path.join(args.out, 'clip_text_embeddings.npy'), tvecs)
        xm = np.einsum('ij,ij->i', vecs, tvecs)
        pd.DataFrame({'review_id': df.review_id,
                      'clip_cross_modal': xm}).to_csv(
            os.path.join(args.out, 'clip_cross_modal.csv'), index=False)
        print(f"text encoding done in {time.time()-t1:.0f}s")
        print(f"  cross-modal similarity: mean={xm.mean():.3f} "
              f"min={xm.min():.3f} max={xm.max():.3f}")

    # ---- rewrite the CSVs -------------------------------------------
    for name in CSVS:
        p = os.path.join(args.out, name)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d['embedding'] = d.review_id.map(emb_map)
        n_null = d.embedding.isna().sum()
        d.to_csv(p, index=False)
        print(f"  {name}: {len(d)} rows updated"
              + (f"  WARNING {n_null} unmapped" if n_null else ""))

    print("\nembeddings are now 512-d CLIP.")
    print("NEXT: python tune_thresholds.py   "
          "(COSINE_REUSE_THRESHOLD=0.94 was tuned for the placeholder "
          "and will be wrong for CLIP)")


if __name__ == '__main__':
    main()
