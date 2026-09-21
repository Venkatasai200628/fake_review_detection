"""
make_bundle.py -- copy the ~75 MB the server actually reads into deploy/bundle/.

The full out/ folder is 1.8 GB, but most of it is things the backend never opens: 349 MB of
dataset photos (only /demo uses those) and 314 MB of forensic maps (only training uses those).
The image memory works from stored pHashes and vectors, not from the photos themselves, which
is why a hosted copy needs so little.

    python deploy/make_bundle.py            build deploy/bundle/
    python deploy/make_bundle.py --demo     also copy the photos, so /demo works when hosted
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'fake_review_dataset_v6', 'out')
DST = os.path.join(HERE, 'bundle')

# Every file Pipeline.__init__ and features.py open. Missing one only shows up as a crash on
# the first request, so this list is deliberately explicit rather than a glob.
NEEDED = [
    'reviews_full.csv',                 # the image memory: ids, hashes, embeddings, timestamps
    'forensic_features.csv',            # ELA / noise numbers for the memory rows
    'clip_cross_modal.csv',             # photo-vs-text agreement for the memory rows
    'cnn_manip_scores_split.csv',       # CNN edit score for the memory rows
    'text_model_scores_split.csv',      # MiniLM text score for the memory rows
    'reuse_embeddings_resnet50.npy',    # ResNet-50 vectors used to spot a reused photo
    'reuse_embeddings_ids.csv',
    'clip_image_embeddings.npy',
    'clip_text_embeddings.npy',
    'test_balanced.csv',                # /demo picks its 14 held-out reviews from here
]

os.makedirs(DST, exist_ok=True)
total = 0
missing = []
for name in NEEDED:
    src = os.path.join(OUT, name)
    if not os.path.exists(src):
        missing.append(name)
        continue
    shutil.copy2(src, os.path.join(DST, name))
    mb = os.path.getsize(src) / 1e6
    total += mb
    print(f'  {mb:8.1f} MB  {name}')

if '--demo' in sys.argv:
    src = os.path.join(OUT, 'images')
    dst = os.path.join(DST, 'images')
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    mb = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / 1e6
    total += mb
    print(f'  {mb:8.1f} MB  images/ ({len(os.listdir(dst))} photos, for /demo)')

print(f'\n  {total:8.1f} MB  total in {DST}')
if missing:
    print('\nMISSING (the server will fail to start without these):')
    for m in missing:
        print('  -', m)
    sys.exit(1)
