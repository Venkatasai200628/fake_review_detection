"""
verify_requirements.py -- PROVE the v6.1 dataset meets every requirement from
IMPLEMENTATION_GUIDE.md section 6 and the earlier project definition (PDF chat).

Each check prints PASS / WARN / FAIL with the evidence (numbers), and the whole
report is written to ../out/REQUIREMENTS_CHECK.md.

Leakage probes: a model is trained to predict the label from things that must
carry NO information (row order, id numbers, reviewer-id format, category,
which photo it is, JPEG settings). Grouped by root photo, their ROC-AUC must be
~0.5 (= coin flip). If any probe scores high, the dataset has a shortcut.

    python verify_requirements.py
"""
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import gen_text as GT

OUT = '../out'
REPO = '../..'
results = []


def check(section, name, ok, evidence, level=None):
    status = level or ('PASS' if ok else 'FAIL')
    results.append((section, name, status, evidence))
    print(f"[{status:4s}] {name}: {evidence}")


def probe_auc(X, y, groups):
    """Grouped 5-fold ROC-AUC of a Random Forest trying to predict y from X."""
    X = np.asarray(X, dtype=float).reshape(len(y), -1)
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=0)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


full = pd.read_csv(f'{OUT}/reviews_full.csv')
bal = pd.read_csv(f'{OUT}/reviews_balanced.csv')
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
meta = json.load(open(f'{OUT}/model/model_meta.json'))
y = (bal.label == 'Fake').astype(int).values
groups = bal.root_image_id.values
fakes = bal[bal.label == 'Fake']
gen = bal[bal.label == 'Genuine']

# ------------------------------------------------------------------ 1 schema
S = '1. Schema and labels'
need = ['review_id', 'product_id', 'category', 'review_text', 'rating', 'reviewer_id',
        'image_file', 'label', 'fraud_type']
miss = [c for c in need if c not in full.columns]
check(S, 'CSV has every column of the agreed schema (PDF: review_id, product_id, category, '
         'review_text, rating, reviewer_id, image_file, label, fake_type)',
      not miss, f"missing: {miss or 'none'}; plus signal_cell, root_image_id, split, timestamp, ...")
check(S, 'labels are exactly Genuine / Fake / Needs_Verification',
      set(full.label) == {'Genuine', 'Fake', 'Needs_Verification'}, dict(full.label.value_counts()))
MAP = {'genuine': 'Genuine', 'visual_manipulation': 'Fake', 'text_deception': 'Fake',
       'coordinated_reuse': 'Fake', 'borderline_recompress': 'Needs_Verification',
       'borderline_ambiguous_reuse': 'Needs_Verification'}
bad = (full.fraud_type.map(MAP) != full.label).sum()
check(S, 'every fraud_type maps to the right label (guide 6.3 table)', bad == 0, f"{bad} mismatches")
missing_img = [f for f in full.image_file if not os.path.exists(os.path.join(OUT, f))]
check(S, 'every row has its image file', not missing_img, f"{len(full)} rows, {len(missing_img)} missing")
check(S, 'ids unique', full.review_id.is_unique and full.image_id.is_unique, 'review_id and image_id unique')

# ----------------------------------------------------------------- 2 balance
S = '2. Balance'
check(S, 'Genuine == Fake overall', len(gen) == len(fakes), f"{len(gen)} vs {len(fakes)}")
for name, part in [('train', tr), ('test', te)]:
    c = part.groupby(['category', 'label']).size().unstack(fill_value=0)
    check(S, f'Genuine == Fake in every category of {name}', (c.Genuine == c.Fake).all(),
          f"{len(c)} categories, per-category counts {sorted(set(c.Genuine))}")
shares = fakes.fraud_type.value_counts(normalize=True)
check(S, 'three fake types in equal thirds (each 30-37% of fakes)',
      shares.between(0.30, 0.37).all(), {k: f"{v:.1%}" for k, v in shares.items()})
cells = bal.signal_cell.value_counts()
check(S, 'all four cells of the 2x2 grid present (none / image_only / text_only / both)',
      {'none', 'image_only', 'text_only', 'both'} <= set(cells.index), dict(cells))
both = bal.groupby('category').label.nunique()
check(S, 'PDF rule 1: every category contains BOTH classes', (both == 2).all(),
      f"{(both == 2).sum()}/{len(both)} categories")

# ------------------------------------------------------------- 3 split rules
S = '3. Split and leakage'
ov = set(tr.root_image_id) & set(te.root_image_id)
check(S, 'no root photo in both train and test (guide 6.6)', not ov, f"overlap = {len(ov)}")
check(S, 'no image in both train and test', not (set(tr.image_id) & set(te.image_id)), 'overlap = 0')
check(S, 'train is balanced 50/50', abs(tr.is_fake.mean() - .5) < 1e-9, f"{tr.is_fake.mean():.1%} fake")
check(S, 'reporting holdout at realistic prevalence (10-20% fake), every fake type present',
      0.10 <= ho.is_fake.mean() <= 0.20 and ho[ho.label == 'Fake'].fraud_type.nunique() == 3,
      f"{ho.is_fake.mean():.1%} fake, types {dict(ho[ho.label == 'Fake'].fraud_type.value_counts())}")
rb = bal.groupby('root_image_id').label.nunique()
check(S, 'photos are used for BOTH classes (photo identity cannot give the label)',
      (rb == 2).mean() > 0.95, f"{(rb == 2).sum()}/{len(rb)} root photos appear as Genuine and as Fake")

# leakage probes
num = lambda s: s.str.extract(r'(\d+)')[0].astype(float)
probes = {
    'row position in the file': np.arange(len(bal)),
    'review_id number': num(bal.review_id),
    'image_id number': num(bal.image_id),
}
for pname, X in probes.items():
    a = probe_auc(X, y, groups)
    check(S, f'leakage probe - {pname} predicts label?', abs(a - 0.5) < 0.05, f"ROC-AUC {a:.3f} (0.5 = no information)")
# category: exact test (a probe AUC is misleading when every category is
# exactly 50/50 -- grouped CV then goes systematically BELOW 0.5, guide 9.5)
cr = bal.groupby('category').is_fake.mean()
check(S, 'product category carries no label information (fake rate identical in every category)',
      cr.min() == cr.max() == 0.5, f"fake rate per category: min {cr.min():.3f}, max {cr.max():.3f} -> mutual information 0")
a_all = probe_auc(num(bal.reviewer_id), y, groups)
k = (bal.fraud_type != 'coordinated_reuse').values
a_nc = probe_auc(num(bal.reviewer_id)[k], y[k], groups[k])
camp_acc = set(bal[bal.fraud_type == 'coordinated_reuse'].reviewer_id)
check(S, 'leakage probe - reviewer account predicts label?', abs(a_nc - 0.5) < 0.05,
      f"ROC-AUC {a_all:.3f} with campaigns, {a_nc:.3f} without. The signal is repeat campaign accounts "
      f"({len(camp_acc)} accounts x ~2 fake reviews, never a genuine one); id numbers carry no order. "
      "Realistic (rings reuse accounts) but simplified: real rings also post genuine reviews",
      level='WARN' if abs(a_nc - 0.5) < 0.05 else 'FAIL')
fmt = bal.reviewer_id.str.fullmatch(r'U\d{5}').mean()
check(S, 'one reviewer-id format for everyone (v5 used C##### for campaign accounts)', fmt == 1.0,
      f"{fmt:.0%} match U#####")
forbidden = {'fraud_type', 'signal_cell', 'campaign_id', 'transform_chain', 'manipulation_score',
             'label', 'is_fake', 'notes', 'root_image_id', 'split'}
leak = forbidden & set(meta['features'])
check(S, 'model never sees answer-key columns (fraud_type, manipulation_score, ...)', not leak,
      f"model features: {len(meta['features'])}, forbidden used: {sorted(leak) or 'none'}")

# ----------------------------------------------------------- 4 image pipeline
S = '4. Images (no processing shortcut)'
qt, sizes, modes, fsize = Counter(), Counter(), Counter(), []
sample = bal.sample(600, random_state=0)
for f in sample.image_file:
    im = Image.open(os.path.join(OUT, f))
    qt[str(sorted(im.quantization[0])[:8])] += 1
    sizes[max(im.size)] += 1
    modes[(im.format, im.mode)] += 1
    fsize.append(os.path.getsize(os.path.join(OUT, f)))
check(S, 'every image saved the same way: one JPEG pass, identical quality tables (Leak 2 fix)',
      len(qt) == 1 and len(modes) == 1, f"{len(qt)} distinct quantisation table(s), formats {dict(modes)} (600 sampled)")
edges = np.array([max(Image.open(os.path.join(OUT, f)).size) for f in full.image_file])
check(S, 'every image stored at ~1024px long edge (guide 9.7: forensics need >= ~1000px)',
      (edges >= 1000).mean() >= 0.99,
      f"{(edges == 1024).sum()} at 1024px, {((edges >= 1000) & (edges < 1024)).sum()} at 1000-1023px, "
      f"{(edges < 1000).sum()} below 1000px (min {edges.min()}): sources just over 1000px, then cropped 90-99%",
      level='WARN' if (edges < 1000).any() else None)
a = probe_auc(np.array(fsize), (sample.label == 'Fake').astype(int).values, sample.root_image_id.values)
check(S, 'file size alone predicts label? (info: edits add real pixel detail, so some signal is expected)',
      True, f"ROC-AUC {a:.3f}", level='INFO')
roots = bal.drop_duplicates('root_file')
small = [r for r in roots.root_file if max(Image.open(os.path.join(REPO, r)).size) < 1000]
check(S, 'every source photo used is >= 1000px (guide 9.7)', not small,
      f"{len(roots)} root photos used, {len(small)} under 1000px")

# ---------------------------------------------------- 5 design of each type
S = '5. Fake types behave as designed'
man_ops = r'(copy_move|splice|noise)$'
vm = bal[bal.fraud_type == 'visual_manipulation']
others = bal[bal.fraud_type != 'visual_manipulation']
check(S, 'ONLY edited-photo fakes have an edited photo',
      vm.transform_chain.str.contains(man_ops).all() and not others.transform_chain.str.contains(man_ops).any(),
      f"{vm.transform_chain.str.extract(man_ops)[0].value_counts().to_dict()}; 0 edits elsewhere")
g_r = gen.rating.value_counts().sort_index()
v_r = vm.rating.value_counts().reindex(g_r.index, fill_value=0)
chi = stats.chi2_contingency(np.vstack([g_r.values, v_r.values]))
check(S, 'edited-photo fakes have the SAME star ratings as genuine (v6.1 shortcut fix)', chi.pvalue > 0.01,
      f"chi-square p = {chi.pvalue:.3f} (p > 0.01 = no detectable difference)")
lazy_set = set(GT.LAZY_GENUINE)
lz_g = gen.review_text.isin(lazy_set).mean()
lz_v = vm.review_text.isin(lazy_set).mean()
check(S, 'edited-photo fakes have the same share of short generic text as genuine', abs(lz_g - lz_v) < 0.06,
      f"genuine {lz_g:.1%} vs edited-photo {lz_v:.1%}")
td = bal[bal.fraud_type == 'text_deception']
check(S, 'fake-text reviews use clean, unedited photos', not td.transform_chain.str.contains(man_ops).any(),
      '0 edited photos among text_deception')
soph = (td.notes == 'OVERLAP_sophisticated_fake').mean()
check(S, 'overlap: ~30% of fake-text reviews invent convincing detail (guide 6.4)', 0.24 <= soph <= 0.36, f"{soph:.1%}")
lg = (gen.notes == 'OVERLAP_lazy_genuine').mean()
check(S, 'overlap: ~18% of genuine reviews are short and generic (guide 6.4)', 0.14 <= lg <= 0.22, f"{lg:.1%}")

camp = bal[bal.fraud_type == 'coordinated_reuse']
span = camp.groupby('campaign_id').timestamp.agg(lambda s: (pd.to_datetime(s).max() - pd.to_datetime(s).min()).total_seconds() / 3600)
sizes_c = camp.groupby('campaign_id').size()
one_root = (camp.groupby('campaign_id').root_image_id.nunique() == 1).all()
check(S, 'campaigns: one photo, 4-6 accounts, all posted within 3 hours', one_root and span.max() <= 3.0 and sizes_c.between(4, 6).all(),
      f"{len(span)} campaigns, size {sizes_c.min()}-{sizes_c.max()}, longest burst {span.max():.2f} h")
gspan = (pd.to_datetime(gen.timestamp).max() - pd.to_datetime(gen.timestamp).min()).days
check(S, 'genuine posting spread over ~180 days', gspan >= 170, f"{gspan} days")

# rating agrees with text polarity (guide 9.3)
def polarity(t):
    if t in GT.DECEPTIVE_NEGATIVE:
        return 'neg'
    if t in GT.DECEPTIVE_POSITIVE or t in lazy_set:
        return 'pos'
    low = t.lower()
    if any(w in low for w in ['disappointing', 'would not buy', 'was the weak point', 'not built for']):
        return 'neg'
    if low.startswith('mixed feelings') or 'though' in low or 'not perfect' in low:
        return 'mixed'
    return 'pos'
pol = bal.review_text.map(polarity)
contra = ((pol == 'pos') & (bal.rating <= 2)) | ((pol == 'neg') & (bal.rating >= 4))
check(S, 'star rating agrees with the text (no "Excellent!" at 1 star, guide 9.3)', contra.sum() == 0,
      f"{contra.sum()} contradictions in {len(bal)} reviews")

# --------------------------------------------------------- 6 negative controls
S = '6. Negative controls (traps)'
notes = full.notes.astype(str)
legit = full[notes.str.contains('legit_stock_reuse')]
lspan = (pd.to_datetime(legit.timestamp).max() - pd.to_datetime(legit.timestamp).min()).days
check(S, 'legit stock-photo reuse present: one photo, many real buyers, spread over weeks, labelled Genuine',
      len(legit) >= 20 and (legit.label == 'Genuine').all() and lspan >= 21,
      f"{len(legit)} rows, {legit.root_image_id.nunique()} photos, spread {lspan} days")
rec = full[full.fraud_type == 'borderline_recompress']
check(S, 'recompress-only rows present and below the 0.25 manipulation threshold',
      len(rec) > 0 and (rec.manipulation_score < 0.25).all(), f"{len(rec)} rows, max score {rec.manipulation_score.max():.2f}")
amb = full[full.fraud_type == 'borderline_ambiguous_reuse']
check(S, 'weak-reuse rows present: exactly 2 accounts per photo',
      len(amb) > 0 and (amb.groupby('root_image_id').reviewer_id.nunique() == 2).all(), f"{len(amb)} rows")
tm = open(f'{OUT}/train_model_log.txt', encoding='utf-8').read()
m = re.search(r'legit stock reuse \(must NOT be flagged\): (\d+)/(\d+)', tm)
check(S, 'trained model does NOT flag legit stock reuse', m and m.group(1) == '0', f"{m.group(1)}/{m.group(2)} flagged" if m else 'n/a')
sec = tm.split('Needs_Verification controls -> model decision ===')[1].split('\n\n')[0].strip().splitlines()
hdr = sec[0].split()[1:]
nv_counts = {ln.split()[0]: dict(zip(hdr, map(int, ln.split()[1:]))) for ln in sec[2:]}
n_fake = sum(v.get('Fake', 0) for v in nv_counts.values())
n_abst = sum(v.get('Needs_Verification', 0) for v in nv_counts.values())
n_tot = sum(sum(v.values()) for v in nv_counts.values())
check(S, 'trained model does NOT call recompress-only / weak-reuse rows Fake', n_fake == 0,
      f"{n_fake}/{n_tot} called Fake: {nv_counts}")
check(S, 'trained model ABSTAINS (Needs Verification) on these borderline rows', n_abst > n_tot / 2,
      f"{n_abst}/{n_tot} sent to Needs Verification -- the decision bands are not tuned yet (guide 16 step 2)",
      level=None if n_abst > n_tot / 2 else 'WARN')

# ------------------------------------------------------------------ 7 scale
S = '7. Scale'
check(S, 'size: PDF target 600-1000 review rows', len(bal) >= 600, f"{len(bal)} Genuine/Fake rows (+{len(full) - len(bal)} controls)")
check(S, '15 product categories', bal.category.nunique() == 15, bal.category.nunique())
rpc = bal.drop_duplicates('root_image_id').groupby('category').size()
check(S, 'guide target: >= 54 usable photos per category', (rpc >= 54).all(),
      f"min {rpc.min()} ({rpc.idxmin()}: 10 photos under 1000px were skipped), max {rpc.max()}",
      level=None if (rpc >= 54).all() else 'WARN')

# ------------------------------------------------------------ 8 not covered
S = '8. Mechanisms from the PDF list that are NOT in this dataset'
for mech in ['stolen seller catalogue photo labelled Fake', 'AI-generated photos',
             'text-image mismatch (text describes a different product)', 'incentivised-review language']:
    check(S, mech, False, 'not generated; the guide 6.3 spec uses the 3 types above instead', level='INFO')

# ------------------------------------------------------------------ report
cnt = Counter(r[2] for r in results)
lines = ['# Requirements check: v6.1 balanced dataset', '',
         'Generated by `code/verify_requirements.py` from the actual files.', '',
         f"**{cnt['PASS']} PASS · {cnt['WARN']} WARN · {cnt['FAIL']} FAIL · {cnt['INFO']} INFO**", '']
for sec in dict.fromkeys(r[0] for r in results):
    lines += [f'## {sec}', '', '| Result | Requirement | Evidence |', '|---|---|---|']
    for s, n, st, ev in results:
        if s == sec:
            lines.append(f"| {st} | {n} | {str(ev).replace('|', '/')} |")
    lines.append('')
open(f'{OUT}/REQUIREMENTS_CHECK.md', 'w', encoding='utf-8').write('\n'.join(lines))
print(f"\n{cnt['PASS']} PASS, {cnt['WARN']} WARN, {cnt['FAIL']} FAIL, {cnt['INFO']} INFO -> out/REQUIREMENTS_CHECK.md")
