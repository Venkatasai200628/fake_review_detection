"""
build_dataset.py (v6 -- BALANCED) -- builds the Layer 2 review dataset from the
Layer 1 root photo pool (the `dataset/` folder that is pushed to GitHub).

WHAT CHANGED FROM v5
  * Genuine and Fake are EXACTLY 50/50 -- overall, inside every split, and
    inside every product category of every split. (v5 was ~40% Fake.)
  * Uses the new 836-photo pool (15 categories x ~55). Roots whose long edge is
    under 1000px are skipped (10 backpack photos) -- see gen_images.py.
  * The split is decided FIRST (by root photo, per category), then each split
    is generated to be balanced. Nothing is thrown away afterwards to force a
    balance, and no root photo ever appears on both sides.
  * Every root photo is used for both Genuine and Fake rows (roots are handed
    out round-robin). So "which photo is this?" can never predict the label --
    only the evidence (edits, reuse pattern, text, timing) can.
  * review_id / image_id are assigned AFTER shuffling, so the row order and the
    id numbers carry no information about the label. (In v5 all genuine rows
    came first, so R00001..R01100 were all genuine.)
  * One reviewer-id namespace (U#####) for everybody. In v5 campaign accounts
    were C##### and genuine accounts U#####, so the id prefix alone gave away
    campaign membership.
  * Images are rendered in parallel (one worker per root photo).

OUTPUTS (in OUT_DIR)
  images/                          derived review images, one JPEG per row
  reviews_full.csv                 every row (Genuine, Fake, Needs_Verification)
  reviews_balanced.csv             Genuine + Fake only, exactly 50/50
  train.csv                        train roots, exactly 50/50
  test_balanced.csv                test roots, exactly 50/50
  holdout_realistic.csv            test roots, ~15% Fake (honest reporting)
  needs_verification_controls.csv  the two abstain-band negative controls
  loco_folds.json                  leave-one-category-out folds (5 x 3 cats)
  build_summary.json               counts, for the README / report
  README.md                        data dictionary

Run from this folder:   python build_dataset.py
"""

import os
import json
import random
import datetime as dt
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import pandas as pd

import gen_images as GI
import gen_text as GT

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 20260919
REPO_DIR = '../..'                  # the Layer 1 image pool = root of the GitHub repo
OUT_DIR = '../out'
MIN_ROOT_EDGE = 1000                # px, long edge (guide section 9.7)
TEST_ROOT_FRACTION = 0.25           # share of each category's roots held out

# Fake rows PER CATEGORY (train + test together). One third per mechanism, so
# the 2x2 "which modality can see it" grid is fully populated.
FAKE_PER_CATEGORY = OrderedDict([
    ('visual_manipulation', 37),    # image is the only tell
    ('text_deception', 37),         # text is the only tell
    ('coordinated_reuse', 36),      # reuse + timing + templated text
])
CAMPAIGN_MEAN_SIZE = 5              # members per campaign ~4-6

# Genuine negative control that sits INSIDE the balanced data:
# the manufacturer photo re-uploaded by real buyers over weeks. High reuse,
# still Genuine. Rows count towards the Genuine half of their category.
LEGIT_REUSE_ROWS = {'train': 20, 'test': 12}

# Abstain-band controls (label Needs_Verification). NOT part of the balanced
# Genuine/Fake data; kept in reviews_full.csv + their own file.
N_RECOMPRESS = 24
N_AMBIGUOUS_PAIRS = 12

LAZY_GENUINE_FRACTION = 0.18        # real customers writing generic text
SOPHISTICATED_FAKE_FRACTION = 0.30  # paid reviewers fabricating detail
TARGET_HOLDOUT_PREVALENCE = 0.15

N_GENUINE_ACCOUNTS = 1800
N_CAMPAIGN_ACCOUNTS = 330

BASE_DATE = dt.datetime(2026, 1, 15, 9, 0, 0)

SIGNAL_CELL = {                     # the 2x2 grid from the project definition
    'genuine': 'none',
    'visual_manipulation': 'image_only',
    'text_deception': 'text_only',
    'coordinated_reuse': 'both',
    'borderline_recompress': 'weak_image',
    'borderline_ambiguous_reuse': 'weak_image',
}

rng = random.Random(SEED)


# --------------------------------------------------------------------------
# Helpers used while PLANNING rows (single process, deterministic)
# --------------------------------------------------------------------------

def make_accounts():
    """One shared id namespace; which ids are campaign accounts is hidden."""
    ids = [f"U{i:05d}" for i in range(1, N_GENUINE_ACCOUNTS + N_CAMPAIGN_ACCOUNTS + 1)]
    rng.shuffle(ids)
    return ids[:N_GENUINE_ACCOUNTS], ids[N_GENUINE_ACCOUNTS:]


def spread_timestamp():
    """Ordinary posting: spread across ~180 days, ordinary hours."""
    return BASE_DATE + dt.timedelta(days=rng.randint(0, 180),
                                    hours=rng.randint(0, 14),
                                    minutes=rng.randint(0, 59))


def burst_timestamps(n):
    """Campaign posting: all members inside a ~3 hour window."""
    start = BASE_DATE + dt.timedelta(days=rng.randint(0, 180),
                                     hours=rng.randint(0, 20))
    return [start + dt.timedelta(minutes=rng.randint(0, 180)) for _ in range(n)]


class RootCycler:
    """Hands out roots round-robin (reshuffled every pass) so every root in a
    (split, category) cell gets used about equally, for Genuine AND Fake."""

    def __init__(self, roots):
        self.roots = list(roots)
        self.queue = []

    def next(self, exclude=()):
        for _ in range(len(self.roots) * 2 + 2):
            if not self.queue:
                self.queue = self.roots[:]
                rng.shuffle(self.queue)
            r = self.queue.pop()
            if r not in exclude:
                return r
        return rng.choice([r for r in self.roots if r not in exclude])


def campaign_sizes(total):
    """Split `total` campaign rows into campaigns of ~4-6 members."""
    if total <= 0:
        return []
    n = max(1, round(total / CAMPAIGN_MEAN_SIZE))
    base, extra = divmod(total, n)
    sizes = [base + (1 if i < extra else 0) for i in range(n)]
    rng.shuffle(sizes)
    return sizes


def split_quota(total, split):
    """Per-category quota for one split (train gets 75%, test 25%)."""
    test = round(total * TEST_ROOT_FRACTION)
    return total - test if split == 'train' else test


# --------------------------------------------------------------------------
# Rendering (runs in worker processes, one task per root photo)
# --------------------------------------------------------------------------

_DONOR_CACHE = OrderedDict()


def _donor(path):
    if path not in _DONOR_CACHE:
        im = GI.open_root(path)
        im.thumbnail((640, 640))
        _DONOR_CACHE[path] = im
        if len(_DONOR_CACHE) > 32:
            _DONOR_CACHE.popitem(last=False)
    return _DONOR_CACHE[path]


def render_root(task):
    """Render every planned row that derives from one root photo."""
    root_path, specs, img_dir = task
    base = GI.open_root(root_path)
    results = []
    for s in specs:
        r = random.Random(s['seed'])              # per-row RNG: order-independent
        np.random.seed(s['seed'] % (2 ** 32))     # used by the noise manipulation
        kind = s['render']
        score = s['manipulation_score']
        if kind == 'benign':
            im, chain = GI.apply_transform_chain(base, r, n_ops=s['n_ops'])
        elif kind == 'manipulate':
            im, chain = GI.apply_transform_chain(base, r, n_ops=2)
            im, op, score = GI.apply_manipulation(im, r, donor=_donor(s['donor_path']))
            chain = f"{chain}+{op}"
        elif kind == 'recompress':
            im, chain, score = GI.apply_light_recompression(base, r)
        else:
            raise ValueError(kind)
        GI.save_derived(im, img_dir, s['image_id'])
        small = im.copy()
        small.thumbnail((GI.OUT_SIZE, GI.OUT_SIZE))
        results.append({
            'review_id': s['review_id'],
            'transform_chain': chain,
            'manipulation_score': score,
            'phash': GI.compute_phash(small),
            'embedding': json.dumps([round(float(x), 5)
                                     for x in GI.compute_embedding(small)]),
        })
    return results


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def plan_rows(pool):
    """Decide every row (label, root, text, rating, time, reviewer) up front."""
    genuine_accts, campaign_accts = make_accounts()
    categories = sorted(pool)

    # ---- split roots per category (stratified, by root photo) -------------
    split_of = {}
    cells = {}                          # (split, category) -> [root ids]
    for cat in categories:
        rids = [r for r, _ in pool[cat]]
        rng.shuffle(rids)
        k = max(1, round(len(rids) * TEST_ROOT_FRACTION))
        for i, rid in enumerate(rids):
            split_of[rid] = 'test' if i < k else 'train'
        cells[('test', cat)] = rids[:k]
        cells[('train', cat)] = rids[k:]
    path_of = {rid: p for cat in categories for rid, p in pool[cat]}
    cat_of = {rid: cat for cat in categories for rid, _ in pool[cat]}

    specs = []

    def add(**kw):
        kw.setdefault('campaign_id', '')
        kw.setdefault('notes', '')
        kw.setdefault('n_ops', None)
        kw.setdefault('donor_path', None)
        kw.setdefault('manipulation_score', round(rng.uniform(0.01, 0.12), 3))
        kw['split'] = split_of[kw['root_image_id']]
        kw['category'] = cat_of[kw['root_image_id']]
        specs.append(kw)

    # ---- legitimate stock-photo reuse: one root per split ----------------
    legit_root = {}
    for split in ('train', 'test'):
        cat = rng.choice(categories)
        legit_root[split] = rng.choice(cells[(split, cat)])

    camp_no = 0
    for split in ('train', 'test'):
        for cat in categories:
            cyc = RootCycler(cells[(split, cat)])
            n_fake = 0

            # visual_manipulation -- authentic-sounding text, edited pixels
            for _ in range(split_quota(FAKE_PER_CATEGORY['visual_manipulation'], split)):
                rid = cyc.next()
                donor_pool = [r for c in categories for r in cells[(split, c)] if r != rid]
                text, _ = GT.gen_genuine_text(cat, rng, sentiment='pos')
                add(root_image_id=rid, fraud_type='visual_manipulation', label='Fake',
                    render='manipulate', donor_path=path_of[rng.choice(donor_pool)],
                    reviewer_id=rng.choice(genuine_accts),
                    timestamp=spread_timestamp(), review_text=text,
                    rating=GT.gen_rating('visual_manipulation', rng, 'pos'))
                n_fake += 1

            # text_deception -- clean real photo, deceptive text
            for _ in range(split_quota(FAKE_PER_CATEGORY['text_deception'], split)):
                rid = cyc.next()
                if rng.random() < SOPHISTICATED_FAKE_FRACTION:
                    text, pol = GT.gen_sophisticated_deceptive_text(cat, rng), 'pos'
                    note = 'OVERLAP_sophisticated_fake'
                else:
                    (text, pol), note = GT.gen_deceptive_text(rng), ''
                add(root_image_id=rid, fraud_type='text_deception', label='Fake',
                    render='benign', reviewer_id=rng.choice(genuine_accts),
                    timestamp=spread_timestamp(), review_text=text,
                    rating=GT.gen_rating('text_deception', rng, pol), notes=note)
                n_fake += 1

            # coordinated_reuse -- one root, many accounts, ~3h burst
            for size in campaign_sizes(split_quota(FAKE_PER_CATEGORY['coordinated_reuse'], split)):
                camp_no += 1
                cid = f"CAMP{camp_no:03d}"
                rid = cyc.next(exclude={legit_root[split]})
                stamps = burst_timestamps(size)
                members = rng.sample(campaign_accts, size)
                for k in range(size):
                    add(root_image_id=rid, fraud_type='coordinated_reuse', label='Fake',
                        render='benign', n_ops=rng.randint(2, 3),
                        manipulation_score=round(rng.uniform(0.02, 0.15), 3),
                        reviewer_id=members[k], timestamp=stamps[k],
                        review_text=GT.gen_campaign_text(cat, rng, campaign_seed=camp_no),
                        rating=GT.gen_campaign_rating(rng),
                        campaign_id=cid, notes='campaign_member')
                    n_fake += 1

            # genuine -- EXACTLY as many as the fakes in this (split, category)
            n_genuine = n_fake
            lr = legit_root[split]
            if cat_of[lr] == cat:
                for _ in range(LEGIT_REUSE_ROWS[split]):
                    text, sent = GT.gen_genuine_text(cat, rng)
                    add(root_image_id=lr, fraud_type='genuine', label='Genuine',
                        render='benign', n_ops=2,
                        manipulation_score=round(rng.uniform(0.01, 0.10), 3),
                        reviewer_id=rng.choice(genuine_accts),
                        timestamp=spread_timestamp(), review_text=text,
                        rating=GT.gen_rating('genuine', rng, sent),
                        notes='NEGATIVE_CONTROL_legit_stock_reuse')
                n_genuine -= LEGIT_REUSE_ROWS[split]
            for _ in range(n_genuine):
                rid = cyc.next()
                if rng.random() < LAZY_GENUINE_FRACTION:
                    text, sent = GT.gen_lazy_genuine_text(rng), 'pos'
                    note = 'OVERLAP_lazy_genuine'
                else:
                    (text, sent), note = GT.gen_genuine_text(cat, rng), ''
                add(root_image_id=rid, fraud_type='genuine', label='Genuine',
                    render='benign', reviewer_id=rng.choice(genuine_accts),
                    timestamp=spread_timestamp(), review_text=text,
                    rating=GT.gen_rating('genuine', rng, sent), notes=note)

    # ---- abstain-band controls (Needs_Verification) ----------------------
    all_roots = list(split_of)
    for _ in range(N_RECOMPRESS):
        rid = rng.choice(all_roots)
        text, sent = GT.gen_genuine_text(cat_of[rid], rng)
        add(root_image_id=rid, fraud_type='borderline_recompress',
            label='Needs_Verification', render='recompress',
            reviewer_id=rng.choice(genuine_accts), timestamp=spread_timestamp(),
            review_text=text, rating=GT.gen_rating('genuine', rng, sent),
            notes='NEGATIVE_CONTROL_recompress_only_below_threshold')
    for _ in range(N_AMBIGUOUS_PAIRS):
        rid = rng.choice(all_roots)
        pair = rng.sample(genuine_accts, 2)
        for k in range(2):
            text, sent = GT.gen_genuine_text(cat_of[rid], rng)
            add(root_image_id=rid, fraud_type='borderline_ambiguous_reuse',
                label='Needs_Verification', render='benign', n_ops=2,
                manipulation_score=round(rng.uniform(0.02, 0.14), 3),
                reviewer_id=pair[k], timestamp=spread_timestamp(),
                review_text=text, rating=GT.gen_rating('genuine', rng, sent),
                notes='NEGATIVE_CONTROL_weak_reuse_not_campaign')

    # ---- shuffle, THEN number (ids carry no label information) -----------
    rng.shuffle(specs)
    for i, s in enumerate(specs, start=1):
        s['review_id'] = f"R{i:05d}"
        s['image_id'] = f"IMG{i:05d}"
        s['seed'] = SEED * 100000 + i
        s['root_path'] = path_of[s['root_image_id']]
    return specs


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = os.path.join(OUT_DIR, 'images')
    os.makedirs(img_dir, exist_ok=True)

    pool, skipped = GI.load_root_pool(REPO_DIR, min_edge=MIN_ROOT_EDGE)
    n_roots = sum(len(v) for v in pool.values())
    print(f"Root pool: {len(pool)} categories, {n_roots} usable root photos "
          f"({len(skipped)} skipped for long edge < {MIN_ROOT_EDGE}px)")
    for p, e in skipped:
        print(f"   skipped {p}  ({e}px)")

    specs = plan_rows(pool)
    print(f"Planned {len(specs)} rows. Rendering images "
          f"with {max(1, mp.cpu_count() - 1)} workers...")

    by_root = OrderedDict()
    for s in specs:
        by_root.setdefault(s['root_path'], []).append(s)
    tasks = [(path, rows, img_dir) for path, rows in by_root.items()]

    rendered = {}
    with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
        for i, res in enumerate(p.imap_unordered(render_root, tasks, chunksize=1), 1):
            for r in res:
                rendered[r['review_id']] = r
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} root photos rendered", flush=True)

    rows = []
    for s in specs:
        r = rendered[s['review_id']]
        rows.append({
            'review_id': s['review_id'],
            'product_id': f"P_{s['category'].replace(' ', '_')}",
            'category': s['category'],
            'reviewer_id': s['reviewer_id'],
            'review_text': s['review_text'],
            'rating': s['rating'],
            'timestamp': s['timestamp'].isoformat(),
            'image_file': f"images/{s['image_id']}.jpg",
            'image_id': s['image_id'],
            'root_image_id': s['root_image_id'],
            'root_file': os.path.relpath(s['root_path'], REPO_DIR).replace('\\', '/'),
            'split': s['split'],
            'label': s['label'],
            'is_fake': int(s['label'] == 'Fake'),
            'fraud_type': s['fraud_type'],
            'signal_cell': SIGNAL_CELL[s['fraud_type']],
            'campaign_id': s['campaign_id'],
            'transform_chain': r['transform_chain'],
            'manipulation_score': r['manipulation_score'],
            'phash': r['phash'],
            'embedding': r['embedding'],
            'notes': s['notes'],
        })
    df = pd.DataFrame(rows).sort_values('review_id').reset_index(drop=True)
    write_outputs(df, pool, skipped)
    return df


def write_outputs(df, pool, skipped):
    out = lambda name: os.path.join(OUT_DIR, name)
    df.to_csv(out('reviews_full.csv'), index=False)

    gf = df[df.label.isin(['Genuine', 'Fake'])]
    nv = df[df.label == 'Needs_Verification']
    train = gf[gf.split == 'train']
    test = gf[gf.split == 'test']
    gf.to_csv(out('reviews_balanced.csv'), index=False)
    train.to_csv(out('train.csv'), index=False)
    test.to_csv(out('test_balanced.csv'), index=False)
    nv.to_csv(out('needs_verification_controls.csv'), index=False)

    # realistic-prevalence holdout: all test Genuine + fakes thinned to ~15%,
    # stratified by fraud_type so every mechanism stays measurable
    gen = test[test.label == 'Genuine']
    fake = test[test.label == 'Fake']
    n_target = int(round(len(gen) * TARGET_HOLDOUT_PREVALENCE /
                         (1 - TARGET_HOLDOUT_PREVALENCE)))
    types = sorted(fake.fraud_type.unique())
    per = n_target // len(types)
    parts = [fake[fake.fraud_type == t].sample(
        n=min(per, (fake.fraud_type == t).sum()), random_state=SEED) for t in types]
    holdout = pd.concat([gen] + parts).sample(frac=1, random_state=SEED)
    holdout.to_csv(out('holdout_realistic.csv'), index=False)

    # leave-one-category-out: 5 folds x 3 categories, every category held out once
    cats = sorted(df.category.unique())
    rng.shuffle(cats)
    folds = [{'fold': i + 1, 'held_out_categories': cats[i::5]} for i in range(5)]
    with open(out('loco_folds.json'), 'w') as f:
        json.dump({'note': 'train on all other categories; test on these',
                   'folds': folds}, f, indent=2)

    # ---- integrity assertions -----------------------------------------
    assert (gf.label == 'Genuine').sum() == (gf.label == 'Fake').sum(), 'NOT BALANCED'
    for name, part in [('train', train), ('test', test)]:
        c = part.groupby(['category', 'label']).size().unstack(fill_value=0)
        assert (c['Genuine'] == c['Fake']).all(), f'{name} not balanced per category'
    overlap = set(train.root_image_id) & set(test.root_image_id)
    assert not overlap, f'ROOT LEAKAGE: {list(overlap)[:5]}'
    assert set(nv[nv.split == 'train'].root_image_id).isdisjoint(test.root_image_id)
    assert df.image_id.is_unique and df.review_id.is_unique
    missing = [f for f in df.image_file if not os.path.exists(out(f))]
    assert not missing, f'missing images: {missing[:3]}'

    summary = {
        'seed': SEED,
        'root_photos_used': int(df.root_image_id.nunique()),
        'root_photos_skipped_small': [os.path.relpath(p, REPO_DIR).replace('\\', '/')
                                      for p, _ in skipped],
        'rows_total': len(df),
        'rows_balanced': len(gf),
        'genuine': int((gf.label == 'Genuine').sum()),
        'fake': int((gf.label == 'Fake').sum()),
        'needs_verification': len(nv),
        'fraud_type_counts': gf.fraud_type.value_counts().to_dict(),
        'train_rows': len(train), 'train_fake_pct': round(100 * train.is_fake.mean(), 2),
        'test_rows': len(test), 'test_fake_pct': round(100 * test.is_fake.mean(), 2),
        'holdout_rows': len(holdout),
        'holdout_fake_pct': round(100 * holdout.is_fake.mean(), 2),
        'train_roots': int(train.root_image_id.nunique()),
        'test_roots': int(test.root_image_id.nunique()),
        'campaigns': int(df.campaign_id.replace('', np.nan).nunique()),
        'per_category': gf.groupby(['category', 'label']).size().unstack().to_dict('index'),
    }
    with open(out('build_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=int)

    print(f"\nGenerated {len(df)} rows / {df.image_id.nunique()} images")
    print(f"  balanced set : {len(gf)} rows  Genuine={summary['genuine']}  "
          f"Fake={summary['fake']}")
    print(f"  train.csv            {len(train):5d} rows  fake={summary['train_fake_pct']}%")
    print(f"  test_balanced.csv    {len(test):5d} rows  fake={summary['test_fake_pct']}%")
    print(f"  holdout_realistic    {len(holdout):5d} rows  fake={summary['holdout_fake_pct']}%")
    print(f"  needs_verification   {len(nv):5d} rows (kept separate)")
    print("  integrity: balanced overall + per category + per split; "
          "no root leakage; all images present")


if __name__ == '__main__':
    main()
