"""Paths and settings for the local backend. Change MODEL_DIR to plug in a new model."""
import os

# PROJECT is the repository root: backend/ sits directly inside it, next to extension/,
# papers/ and the dataset folders. (Before 21 Sep 2026 backend/ lived one level above the
# repo, so this walked up twice and then back down through 'dataset'.)
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V6 = os.path.join(PROJECT, 'fake_review_dataset_v6')
CODE_DIR = os.path.join(V6, 'code')          # features.py, forensics.py, ... (same code as training)
OUT_DIR = os.path.join(V6, 'out')            # image memory = the v6 dataset
MODEL_DIR = os.path.join(OUT_DIR, 'model')   # <-- drop a better model here and restart

# On a laptop this listens on 127.0.0.1 and nothing else can reach it. A hosted copy has to
# bind 0.0.0.0, which is why the two guards below exist -- see deploy/README.md.
HOST = os.environ.get('RPF_HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT', os.environ.get('RPF_PORT', 8000)))

# --- guards that only matter once this is reachable by anyone else ----------------------
# 1. A shared token on /analyze. Empty means open, which is right for 127.0.0.1 and wrong
#    anywhere else; deploy/ sets it from the environment.
API_TOKEN = os.environ.get('RPF_TOKEN', '')

# 2. load_image() fetches whatever URL the caller sends. On a laptop that is harmless. On a
#    public host it is a server-side request forgery hole: a stranger could make the server
#    fetch internal addresses, including a cloud metadata endpoint (169.254.169.254 on Azure
#    and AWS) that can hand out credentials. Only these hosts are fetchable, and any host
#    that resolves to a private, loopback or link-local address is refused regardless.
ALLOWED_IMAGE_HOSTS = (
    'm.media-amazon.com', 'images-na.ssl-images-amazon.com', 'images-eu.ssl-images-amazon.com',
    'rukminim1.flixcart.com', 'rukminim2.flixcart.com', 'rukminim3.flixcart.com',
    'images.meesho.com',
)
ALLOW_ANY_IMAGE_HOST = os.environ.get('RPF_ALLOW_ANY_HOST', '') == '1'   # testing only
OUT_SIZE = 1024          # same as gen_images.OUT_SIZE
OUT_QUALITY = 88         # same as gen_images.OUT_QUALITY
MIN_FORENSIC_EDGE = 1000 # below this, ELA / CNN evidence is weakened (guide 9.7)

# Below MIN_CNN_EDGE the tamper evidence is DISCARDED, not just labelled unreliable: the
# forensic features are replaced by the memory median so they cannot move the risk score,
# and no "signs of editing" reason is shown. Measured with diagnose_small_photos.py by
# shrinking test photos and re-scoring them with the same CNN:
#     <=1024px ROC 0.947 | 800px 0.898 | 600px 0.835 | 480px 0.734 | 360px 0.629 | 240px 0.577
# 800px still carries real evidence, so a blanket 1000px cutoff would throw away signal.
# The collapse happens below ~600px, which is where marketplace review thumbnails live.
MIN_CNN_EDGE = 600
# Between MIN_CNN_EDGE and MIN_FORENSIC_EDGE the evidence is kept but the claim needs a
# higher score before it is worth telling the user about.
WEAK_CNN_MIN_SCORE = 0.65
LISTING_MATCH_HAMMING = 10
DEMO_N_REVIEWS = 14
DEMO_SEED = 7
