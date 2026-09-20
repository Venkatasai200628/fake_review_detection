"""
regen_text.py -- v6.2: rewrite ONLY the review text, using the large space in gen_text2.py.

WHY ONLY THE TEXT
The photos, the train/test split, the review ids, the reviewer accounts, the timestamps and
the campaigns are all fine and were expensive to build (forensic maps, ResNet-50 reuse
embeddings, CLIP vectors all index these exact rows). The one defect measured in
TEXT_DIFFICULTY_AUDIT.md is the review text: 46.2% of test reviews are word-for-word copies
of a training review. So this pass touches the text column and nothing else. Every image
artefact on disk stays valid.

HOW EACH ROW IS REWRITTEN
Each row keeps its style and its sentiment, so the dataset means the same thing afterwards:

  style      taken from fraud_type + notes, exactly as build_dataset.py assigned it
             (notes marks OVERLAP_lazy_genuine and OVERLAP_sophisticated_fake; rows where
             build_dataset.py did not record the lazy/detailed coin-flip -- the
             visual_manipulation rows -- re-draw it at the same 18% rate)
  sentiment  read back from the row's EXISTING rating (5,4 -> positive, 3 -> mixed,
             2,1 -> negative), so text and stars still agree and the v6.1 fix that removed
             the rating shortcut is preserved. Ratings are not modified.
  rng        seeded from the review_id, so the result is reproducible and does not depend
             on row order.

HARD CONSTRAINT
No text may appear in more than one row, except inside a single campaign, where members are
supposed to look alike. Enforced by rejection sampling, then asserted.

    python regen_text.py            rewrite and report
    python regen_text.py --check    report only, write nothing
"""
import hashlib
import random
import sys

import pandas as pd

import gen_text2 as G

OUT = '../out'
LAZY_GENUINE_FRACTION = 0.18          # same rates as build_dataset.py
SOPHISTICATED_FAKE_FRACTION = 0.30
MAX_TRIES = 400


def rng_for(review_id, salt='v6.2'):
    h = hashlib.sha256(f'{salt}:{review_id}'.encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def sentiment_of(rating):
    return 'pos' if rating >= 4 else ('mixed' if rating == 3 else 'neg')


def style_of(row, rng):
    """Which of the five writing styles this row was written in."""
    notes = str(row.notes) if pd.notna(row.notes) else ''
    ft = row.fraud_type
    if ft == 'coordinated_reuse':
        return 'campaign'
    if ft == 'text_deception':
        if 'sophisticated' in notes:
            return 'sophisticated'
        return 'deceptive'
    # genuine, visual_manipulation, borderline_* : all written as genuine customers
    if 'lazy_genuine' in notes:
        return 'lazy'
    # build_dataset.py recorded the lazy/detailed coin-flip in `notes` for genuine rows,
    # so no note means detailed. It did NOT record it for visual_manipulation rows
    # (they carry no note at all), so those re-draw it at the same 18% rate.
    if ft != 'genuine' and row.rating >= 4 and rng.random() < LAZY_GENUINE_FRACTION:
        return 'lazy'
    return 'genuine'


def make(style, row, rng):
    sent = sentiment_of(row.rating)
    if style == 'campaign':
        return G.campaign_text(row.category, int(str(row.campaign_id)[4:]), rng)
    if style == 'deceptive':
        return G.deceptive_generic(rng, 'neg' if sent == 'neg' else 'pos')
    if style == 'sophisticated':
        return G.sophisticated_deceptive(row.category, rng)
    if style == 'lazy':
        return G.lazy_genuine(rng)
    return G.genuine_detailed(row.category, rng, sent)


def main(write=True):
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    used = {}                                    # text -> campaign_id or None
    texts, styles = [], []
    collisions = 0
    for row in df.itertuples(index=False):
        rng = rng_for(row.review_id)
        st = style_of(row, rng)
        cid = row.campaign_id if pd.notna(row.campaign_id) else None
        for attempt in range(MAX_TRIES):
            t = make(st, row, rng)
            if t not in used or (cid is not None and used[t] == cid):
                break                            # unused, or a repeat inside its own campaign
            collisions += 1
        else:
            raise RuntimeError(f'{row.review_id}: no free {st} text after {MAX_TRIES} tries')
        used.setdefault(t, cid)
        texts.append(t)
        styles.append(st)

    df['review_text_v62'] = texts
    print(f'rewrote {len(df)} reviews  ({collisions} redraws to avoid cross-row duplicates)')
    print('\nstyle counts:')
    print(pd.Series(styles).value_counts().to_string())

    # ---- checks ----------------------------------------------------------
    d = df.assign(style=styles)
    nc = d[d.campaign_id.isna()]
    dup = nc.review_text_v62.duplicated().sum()
    print(f'\nduplicate texts outside campaigns: {dup}   (must be 0)')
    tr = set(d[d.split == 'train'].review_text_v62)
    te_rows = d[d.split == 'test']
    cross = te_rows.review_text_v62.isin(tr)
    print(f'test texts that also appear in train: {int(cross.sum())} / {len(te_rows)} '
          f'({cross.mean():.1%})   (was 46.2%)')
    for ft in ['genuine', 'text_deception', 'coordinated_reuse', 'visual_manipulation']:
        m = te_rows.fraud_type == ft
        if m.any():
            print(f'   {ft:22s} {int(cross[m].sum())}/{int(m.sum())} ({cross[m].mean():.1%})')
    per_camp = d[d.campaign_id.notna()].groupby('campaign_id').review_text_v62.nunique()
    print(f'\ncampaigns: {len(per_camp)}, distinct texts within a campaign: '
          f'{per_camp.min()}-{per_camp.max()} (members stay near-identical by design)')
    print(f'unique texts overall: {d.review_text_v62.nunique()} / {len(d)} '
          f'(was 2027 / 3348)')
    assert dup == 0, 'duplicate text outside a campaign'

    if not write:
        print('\n--check: nothing written')
        return
    # keep the old text so this pass is reversible and the audit can be repeated
    bak = f'{OUT}/review_text_v61.csv'
    df[['review_id', 'review_text']].to_csv(bak, index=False)
    print(f'\nold text saved to {bak}')
    df['review_text'] = df.pop('review_text_v62')
    df.to_csv(f'{OUT}/reviews_full.csv', index=False)
    # the split files carry their own copy of the text
    for name in ['train.csv', 'test_balanced.csv', 'holdout_realistic.csv']:
        p = f'{OUT}/{name}'
        s = pd.read_csv(p)
        s['review_text'] = s.review_id.map(dict(zip(df.review_id, df.review_text)))
        s.to_csv(p, index=False)
        print(f'updated {name} ({len(s)} rows)')
    print('\nreviews_full.csv and the split files now carry v6.2 text.')
    print('NEXT: text_model.py -> train_model.py (image artefacts are untouched and stay valid)')


if __name__ == '__main__':
    main(write='--check' not in sys.argv)
