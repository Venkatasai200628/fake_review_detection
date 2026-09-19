"""
features.py — turns raw review rows into the module feature blocks defined in
PROJECT_SPEC.md section 5.

Feature blocks, kept separate so ablations are exact:
    TEXT      text deception signal
    IMAGE     manipulation + provenance (the novel visual-forensic block)
    BEHAVIOUR rating and timing
    GRAPH     visual coordination graph features

IMPORTANT ON LEAKAGE
Provenance and graph features are computed over the union of train and holdout.
That is legitimate and intentional: a real platform sees every image on the site
when it indexes for reuse, so restricting the index to training rows would
understate the deployed system. It is *transductive on unlabelled structure
only* -- no label from the holdout is ever touched. Splits remain by
root_image_id, so a holdout row's near-duplicates live in holdout too.
"""

import json
import os
import numpy as np
import pandas as pd
import networkx as nx
import imagehash

PHASH_REUSE_THRESHOLD = 20      # Hamming; validated margin sits at ~18 vs 24
CLIP_REUSE_THRESHOLD = 0.749     # CLIP ViT-B/32; max Youden J on v6 data (TPR 0.997, FPR 0.0039). Placeholder value was 0.94.
RESNET_REUSE_THRESHOLD = 0.72    # ResNet-50 reuse vectors; max Youden J on v6.1 (TPR 0.9997, FPR 0.0015)
COSINE_REUSE_THRESHOLD = CLIP_REUSE_THRESHOLD   # set by _reuse_index() to whichever vectors are in use
REUSE_EMB_NPY = '../out/reuse_embeddings_resnet50.npy'   # see reuse_embedding.py
REUSE_IDS_CSV = '../out/reuse_embeddings_ids.csv'
BURST_WINDOW_HOURS = 6
CNN_SCORES_CSV = '../out/cnn_manip_scores.csv'   # see cnn_forensics.py
TEXT_SCORES_CSV = '../out/text_model_scores.csv'  # see text_model.py

GENERIC_TOKENS = {
    'amazing', 'excellent', 'superb', 'perfect', 'best', 'outstanding',
    'fantastic', 'wonderful', 'awful', 'terrible', 'worst', 'great',
    'highly', 'recommend', 'recommended', 'love', 'quality', 'value',
}
SPECIFIC_MARKERS = {
    'after', 'within', 'weeks', 'month', 'months', 'year', 'half',
    'started', 'stopped', 'developed', 'came', 'went', 'using', 'used',
}


# ---------------------------------------------------------------- TEXT

def text_features(df, overrides=None):
    t = df.review_text.fillna('')
    words = t.str.lower().str.findall(r"[a-z']+")
    n = words.apply(len).replace(0, 1)
    out = pd.DataFrame(index=df.index)
    out['txt_len'] = words.apply(len)
    out['txt_unique_ratio'] = words.apply(lambda w: len(set(w))) / n
    out['txt_generic_ratio'] = words.apply(
        lambda w: sum(x in GENERIC_TOKENS for x in w)) / n
    out['txt_specific_ratio'] = words.apply(
        lambda w: sum(x in SPECIFIC_MARKERS for x in w)) / n
    out['txt_has_number'] = t.str.contains(r'\d').astype(int)
    out['txt_exclam'] = t.str.count('!')
    out['txt_sent_count'] = t.str.count(r'[.!?]')
    # near-duplicate text across the corpus: the campaign-template signal
    counts = t.str.lower().str.strip().value_counts()
    out['txt_exact_dup_count'] = t.str.lower().str.strip().map(counts).fillna(1)

    # MiniLM sentence-embedding model (text_model.py): out-of-fold "this text
    # is deceptive" probability. crossval.py uses the cross-fitted file;
    # train_model.py / the backend use the train-only file or an override.
    ov = overrides or {}
    try:
        if 'text_model' in ov:
            ts = ov['text_model']
        else:
            ts = pd.read_csv(TEXT_SCORES_CSV).set_index('review_id')
        tc = ts.reindex(df.review_id.values)
        tc.index = df.index
        out = pd.concat([out, tc], axis=1)
    except FileNotFoundError:
        pass
    return out


# ----------------------------------------------------- IMAGE (forensic)

def _load_reuse_embeddings():
    """review_id -> ResNet-50 reuse vector (reuse_embedding.py), or None."""
    if not os.path.exists(REUSE_EMB_NPY):
        return None
    E = np.load(REUSE_EMB_NPY).astype(np.float32)
    ids = pd.read_csv(REUSE_IDS_CSV).review_id.values
    return dict(zip(ids, E))


def _reuse_index(df, overrides=None):
    """Pairwise near-duplicate structure over the whole corpus.

    v6.1: the similarity side uses ResNet-50 reuse vectors when available
    (better at "same photo, lightly edited" than CLIP -- see
    reuse_embedding.py); otherwise the CLIP `embedding` column as before.
    """
    global COSINE_REUSE_THRESHOLD
    hashes = {r.image_id: imagehash.hex_to_hash(r.phash) for r in df.itertuples()}
    ov = overrides or {}
    rmap = ov.get('reuse_emb') if 'reuse_emb' in ov else _load_reuse_embeddings()
    if rmap is not None:
        embs = {r.image_id: rmap[r.review_id] for r in df.itertuples()}
        COSINE_REUSE_THRESHOLD = RESNET_REUSE_THRESHOLD
    else:
        embs = {r.image_id: np.array(json.loads(r.embedding)) for r in df.itertuples()}
        COSINE_REUSE_THRESHOLD = CLIP_REUSE_THRESHOLD
    ids = list(hashes)
    E = np.stack([embs[i] for i in ids])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    C = E @ E.T                                  # cosine matrix

    # Vectorised Hamming distance. The nested Python loop was O(n^2) with
    # per-pair object overhead and did not scale past a few hundred rows.
    # Unpack each 64-bit hash to a bit vector, then XOR-count via matrix algebra:
    #   hamming(a,b) = popcount(a XOR b) = a.(1-b) + (1-a).b
    B = np.stack([h.hash.flatten().astype(np.int8) for h in hashes.values()])
    H = (B @ (1 - B).T) + ((1 - B) @ B.T)
    H = H.astype(np.int16)
    np.fill_diagonal(H, 64)
    return ids, H, C


def image_features(df, index, forensic_csv='../out/forensic_features.csv',
                   overrides=None):
    """
    NOTE ON THE MANIPULATION FEATURE
    Earlier versions used the `manipulation_score` COLUMN, which is the
    generator's ground truth, not a measurement -- the module was reading the
    answer key and scoring 1.000. It is now replaced by ELA and noise-residual
    statistics computed from the saved JPEGs by forensics.py. Nothing in this
    block touches a label.
    """
    ids, H, C = index
    # overrides: {'forensic': df, 'xmodal': df, 'cnn': df}, each indexed by
    # review_id. Used by the extension backend to pass features of brand-new
    # photos; training leaves it None and reads the CSVs as before.
    ov = overrides or {}
    if 'forensic' in ov:
        forensic = ov['forensic']
    else:
        forensic = pd.read_csv(forensic_csv).set_index('review_id')
    pos = {k: i for i, k in enumerate(ids)}
    rows = []
    reviewer = df.set_index('image_id').reviewer_id.to_dict()
    prod = df.set_index('image_id').product_id.to_dict()
    ts = df.set_index('image_id').timestamp.apply(pd.Timestamp).to_dict()

    for r in df.itertuples():
        i = pos[r.image_id]
        h_row, c_row = H[i], C[i]
        near = np.where((h_row <= PHASH_REUSE_THRESHOLD) |
                        (c_row >= COSINE_REUSE_THRESHOLD))[0]
        near = [j for j in near if j != i]
        near_ids = [ids[j] for j in near]

        # who and when reused it -- reuse alone is not fraud
        distinct_reviewers = len({reviewer[k] for k in near_ids})
        distinct_products = len({prod[k] for k in near_ids})
        if near_ids:
            times = [ts[k] for k in near_ids] + [ts[r.image_id]]
            span_h = (max(times) - min(times)).total_seconds() / 3600
            in_burst = sum(
                abs((ts[k] - ts[r.image_id]).total_seconds()) / 3600
                <= BURST_WINDOW_HOURS for k in near_ids)
        else:
            span_h, in_burst = 0.0, 0

        rows.append({
            'img_reuse_count': len(near),
            'img_min_phash_dist': int(h_row.min()),
            'img_max_cosine': float(np.max(np.delete(c_row, i))),
            'img_reuse_distinct_reviewers': distinct_reviewers,
            'img_reuse_distinct_products': distinct_products,
            'img_reuse_span_hours': span_h,
            'img_reuse_in_burst': in_burst,
            'img_burst_ratio': in_burst / max(len(near), 1),
        })
    out = pd.DataFrame(rows, index=df.index)
    fcols = forensic.reindex(df.review_id.values)
    fcols.index = df.index
    parts = [out, fcols]

    # Cross-modal consistency: CLIP image-embedding . CLIP text-embedding.
    # Only available once recompute_embeddings.py has run locally -- CLIP
    # cannot be fetched in the sandbox. Absent, the pipeline runs unchanged
    # with the placeholder descriptor, so results stay reproducible either way.
    try:
        if 'xmodal' in ov:
            xm = ov['xmodal']
        else:
            xm = pd.read_csv('../out/clip_cross_modal.csv').set_index('review_id')
        xc = xm.reindex(df.review_id.values)
        xc.index = df.index
        parts.append(xc)
    except FileNotFoundError:
        pass

    # CNN on forensic maps (cnn_forensics.py): out-of-fold "photo was edited"
    # probability. crossval.py uses the cross-fitted file; train_model.py
    # switches CNN_SCORES_CSV to the train-roots-only file.
    try:
        if 'cnn' in ov:
            cs = ov['cnn']
        else:
            cs = pd.read_csv(CNN_SCORES_CSV).set_index('review_id')
        cc = cs.reindex(df.review_id.values)
        cc.index = df.index
        parts.append(cc)
    except FileNotFoundError:
        pass

    return pd.concat(parts, axis=1)


# ------------------------------------------------------------ BEHAVIOUR

def behaviour_features(df):
    out = pd.DataFrame(index=df.index)
    out['beh_rating'] = df.rating
    out['beh_is_extreme'] = df.rating.isin([1, 5]).astype(int)
    t = df.timestamp.apply(pd.Timestamp)
    rc = df.reviewer_id.value_counts()
    out['beh_reviewer_count'] = df.reviewer_id.map(rc)
    out['beh_hour'] = t.dt.hour
    # reviewer's own posting spread
    spread = df.assign(_t=t).groupby('reviewer_id')._t.agg(
        lambda s: (s.max() - s.min()).total_seconds() / 3600 if len(s) > 1 else -1)
    out['beh_reviewer_span_hours'] = df.reviewer_id.map(spread)
    return out


# ---------------------------------------------------------------- GRAPH

def graph_features(df, index):
    """
    Visual coordination graph (PROJECT_SPEC.md M5): nodes are reviewers,
    products and images; an image-similarity edge links reviewers whose images
    are near-duplicates. Louvain communities + clustering coefficient.
    """
    ids, H, C = index
    pos = {k: i for i, k in enumerate(ids)}
    reviewer = df.set_index('image_id').reviewer_id.to_dict()

    # Nodes and edges are added in SORTED order: Louvain's result depends on
    # insertion order, so without this the same graph gave different community
    # sizes when rows arrived in a different order (found by the backend
    # parity test -- g_community_size differed by 173).
    G = nx.Graph()
    G.add_nodes_from(sorted(df.reviewer_id.unique()))
    A = np.triu((H <= PHASH_REUSE_THRESHOLD) | (C >= COSINE_REUSE_THRESHOLD), k=1)
    edges = set()
    for a, b in zip(*np.nonzero(A)):
        ra, rb = reviewer[ids[a]], reviewer[ids[b]]
        if ra != rb:
            edges.add((min(ra, rb), max(ra, rb)))
    G.add_edges_from(sorted(edges), weight=1.0)

    clust = nx.clustering(G)
    deg = dict(G.degree())
    try:
        comms = nx.community.louvain_communities(G, seed=0)
    except Exception:
        comms = []
    comm_of, comm_size = {}, {}
    for ci, c in enumerate(comms):
        for node in c:
            comm_of[node] = ci
            comm_size[node] = len(c)

    out = pd.DataFrame(index=df.index)
    out['g_degree'] = df.reviewer_id.map(deg).fillna(0)
    out['g_clustering'] = df.reviewer_id.map(clust).fillna(0)
    out['g_community_size'] = df.reviewer_id.map(comm_size).fillna(1)
    out['g_in_community'] = (out.g_community_size > 2).astype(int)
    return out


# ------------------------------------------------------------ ASSEMBLY

BLOCKS = {
    'TEXT': text_features,
    'IMAGE': image_features,
    'BEHAVIOUR': behaviour_features,
    'GRAPH': graph_features,
}


def build_all(df, overrides=None):
    index = _reuse_index(df, overrides)
    return {
        'TEXT': text_features(df, overrides=overrides),
        'IMAGE': image_features(df, index, overrides=overrides),
        'BEHAVIOUR': behaviour_features(df),
        'GRAPH': graph_features(df, index),
    }
