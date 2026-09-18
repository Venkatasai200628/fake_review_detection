"""validate.py — sanity checks on the generated dataset before modelling."""
import json, itertools, numpy as np, pandas as pd
import gen_images as GI

OUT = '../out'
df = pd.read_csv(f'{OUT}/reviews_full.csv')
print(f"rows={len(df)}  images={df.image_id.nunique()}  "
      f"roots_used={df.root_image_id.nunique()}  categories={df.category.nunique()}\n")

print("=== label distribution ===")
print(df.label.value_counts().to_string())
print(f"\n=== fraud_type distribution ===")
print(df.fraud_type.value_counts().to_string())

print("\n=== 1. pHash separation: same-root reuse vs unrelated ===")
same, diff = [], []
by_root = df.groupby('root_image_id')
for rid, g in by_root:
    if len(g) < 2: continue
    for a, b in itertools.islice(itertools.combinations(g.phash.tolist(), 2), 12):
        same.append(GI.phash_distance(a, b))
rows = df.sample(n=400, random_state=1)
for a, b in itertools.islice(itertools.combinations(
        rows[['phash', 'root_image_id']].values.tolist(), 2), 4000):
    if a[1] != b[1]:
        diff.append(GI.phash_distance(a[0], b[0]))
same, diff = np.array(same), np.array(diff)
print(f"  same root      n={len(same):5d}  mean={same.mean():5.1f}  "
      f"p95={np.percentile(same,95):4.0f}  max={same.max()}")
print(f"  different root n={len(diff):5d}  mean={diff.mean():5.1f}  "
      f"p5={np.percentile(diff,5):4.0f}   min={diff.min()}")
gap = np.percentile(diff, 5) - np.percentile(same, 95)
print(f"  margin (diff p5 - same p95) = {gap:.0f}  "
      f"{'CLEAN' if gap > 0 else 'OVERLAP -- provenance threshold will misfire'}")

print("\n=== 2. Embedding cosine separation ===")
emb = {r.image_id: np.array(json.loads(r.embedding)) for r in df.itertuples()}
s2, d2 = [], []
for rid, g in by_root:
    ids = g.image_id.tolist()
    if len(ids) < 2: continue
    for a, b in itertools.islice(itertools.combinations(ids, 2), 12):
        s2.append(GI.cosine(emb[a], emb[b]))
samp = df.sample(n=250, random_state=2)[['image_id', 'root_image_id']].values.tolist()
for a, b in itertools.islice(itertools.combinations(samp, 2), 3000):
    if a[1] != b[1]:
        d2.append(GI.cosine(emb[a[0]], emb[b[0]]))
s2, d2 = np.array(s2), np.array(d2)
print(f"  same root      mean={s2.mean():.3f}  p5={np.percentile(s2,5):.3f}")
print(f"  different root mean={d2.mean():.3f}  p95={np.percentile(d2,95):.3f}")

print("\n=== 3. Manipulation score vs threshold (0.25) ===")
for ft in ['genuine', 'visual_manipulation', 'text_deception',
           'coordinated_reuse', 'borderline_recompress']:
    s = df[df.fraud_type == ft].manipulation_score
    if len(s):
        print(f"  {ft:24s} n={len(s):3d}  mean={s.mean():.3f}  "
              f"range=[{s.min():.3f},{s.max():.3f}]")

print("\n=== 4. Negative controls ===")
ctrl = df[df.notes.astype(str).str.startswith('NEGATIVE_CONTROL')]
for note, g in ctrl.groupby('notes'):
    print(f"  {note[17:]:42s} n={len(g):2d}  labels={dict(g.label.value_counts())}")
rec = df[df.fraud_type == 'borderline_recompress']
bad = (rec.manipulation_score >= GI.MANIPULATION_THRESHOLD).sum()
print(f"  recompress-only rows above threshold: {bad}  "
      f"{'OK' if bad == 0 else 'FAIL'}")
legit = df[df.notes.astype(str).str.contains('legit_stock_reuse')]
print(f"  legit stock reuse all labelled Genuine: "
      f"{'OK' if (legit.label == 'Genuine').all() else 'FAIL'}  "
      f"(shares root with {df[df.root_image_id.isin(legit.root_image_id)].shape[0]} rows total)")

print("\n=== 5. Campaign integrity (burst timing) ===")
camp = df[df.campaign_id.notna() & (df.campaign_id != '')]
spans = []
for cid, g in camp.groupby('campaign_id'):
    t = pd.to_datetime(g.timestamp)
    span_h = (t.max() - t.min()).total_seconds() / 3600
    spans.append(span_h)
    assert g.root_image_id.nunique() == 1, f"{cid} spans multiple roots"
print(f"  campaigns={len(spans)}  members={len(camp)}  "
      f"max span={max(spans):.2f}h  (design target <=3h)")
gen_t = pd.to_datetime(df[df.fraud_type == 'genuine'].timestamp)
print(f"  genuine posting spread: {(gen_t.max()-gen_t.min()).days} days")

print("\n=== 6. Rating skew ===")
print(pd.crosstab(df.fraud_type, df.rating).to_string())

print("\n=== 7. Text specificity gap (proxy: distinct-token ratio) ===")
for ft in ['genuine', 'text_deception', 'coordinated_reuse']:
    t = df[df.fraud_type == ft].review_text
    uniq = t.nunique() / len(t)
    print(f"  {ft:20s} unique_text_ratio={uniq:.2f}  "
          f"mean_len={t.str.split().str.len().mean():.1f} words")

print("\n=== 8. Split integrity ===")
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
print(f"  train         {len(tr)} rows, fraud {100*(tr.label=='Fake').mean():.1f}%")
print(f"  test_balanced {len(te)} rows, fraud {100*(te.label=='Fake').mean():.1f}%")
print(f"  holdout       {len(ho)} rows, fraud {100*(ho.label=='Fake').mean():.1f}%")
ov = len(set(tr.root_image_id) & set(te.root_image_id))
print(f"  root overlap train/test: {ov} (must be 0)  {'OK' if ov == 0 else 'FAIL'}")
print(f"  categories in train {tr.category.nunique()}, test {te.category.nunique()}")

print("\n=== 9. Balance (v6) ===")
bal = pd.read_csv(f'{OUT}/reviews_balanced.csv')
g, f = (bal.label == 'Genuine').sum(), (bal.label == 'Fake').sum()
print(f"  overall   Genuine={g}  Fake={f}  {'OK' if g == f else 'FAIL'}")
for name, part in [('train', tr), ('test', te)]:
    c = part.groupby(['category', 'label']).size().unstack(fill_value=0)
    ok = (c['Genuine'] == c['Fake']).all()
    print(f"  {name:5s} per-category Genuine==Fake in all {len(c)} categories: "
          f"{'OK' if ok else 'FAIL'}")
# the photo itself must not predict the label: roots used by both classes
both = bal.groupby('root_image_id').label.nunique()
print(f"  roots used by BOTH Genuine and Fake rows: {(both == 2).sum()}/{len(both)} "
      f"({100*(both == 2).mean():.0f}%)")
print(f"  fraud_type split of the Fake half: "
      f"{bal[bal.label=='Fake'].fraud_type.value_counts().to_dict()}")
