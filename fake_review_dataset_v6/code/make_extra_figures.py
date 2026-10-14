"""
make_extra_figures.py -- the five "should-have" paper figures.

  fig14_leakage_shuffle      shuffled-label test (no feature leaks the answer)
  fig15_robustness_caps      result survives capping variants per root photo
  fig16_ela_resolution       NEW measured sweep: manipulation evidence vs output resolution
  fig17_pipeline             architecture / pipeline diagram
  fig18_qualitative          original vs edited photo + ELA map + reuse burst timeline
  fig19_review_examples      genuine vs deceptive review text examples

fig14/fig15 values come from the recorded verification run and IMPLEMENTATION_GUIDE
sec 8.5. fig16 is a fresh diagnostic computed here (GroupKFold on root photo, so
no root can appear in train and test together). fig18 uses real dataset rows.

    python make_extra_figures.py
"""
import io
import os
import random
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import forensics as FX
import gen_images as GI

OUT = '../out'
REPO = '../..'
FIGDIR = '../../paper_figures'
os.makedirs(FIGDIR, exist_ok=True)

sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'figure.facecolor': 'white'})
PAL = sns.color_palette('colorblind')
BLUE, ORANGE, GREEN, RED, PURPLE, BROWN = (PAL[0], PAL[1], PAL[2],
                                           PAL[3], PAL[4], PAL[5])


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}')


# ================================================ fig14 label-shuffle test
# from code/verify_results.py run (../out/VERIFY_RESULTS.md, 2026-10-14)
shuffled = [0.536, 0.521, 0.444, 0.616, 0.428]
real_auc = 0.967

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar([f'run {i+1}' for i in range(len(shuffled))], shuffled,
       color=ORANGE, width=.6, label='shuffled labels (5 runs)')
ax.axhline(0.5, color='grey', ls='--', lw=1.2, label='chance (0.5)')
ax.axhline(real_auc, color=GREEN, lw=2, label=f'real labels (ROC-AUC {real_auc})')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Test ROC-AUC')
ax.set_title('Leakage test: retrain on randomly shuffled labels')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.96), frameon=False)
for i, v in enumerate(shuffled):
    ax.text(i, v + .02, f'{v:.3f}', ha='center', fontsize=8)
fig.tight_layout()
save(fig, 'fig14_leakage_shuffle')

# ================================================ fig15 robustness under caps
# IMPLEMENTATION_GUIDE sec 8.5 (PR-AUC, 5x5 grouped CV)
cap = pd.DataFrame({
    'cap': ['3 variants/root', '5 variants/root', 'uncapped (~10/root)'],
    'rows': [590, 963, 1926],
    'fusion': [0.785, 0.799, 0.826],
    'base': [0.740, 0.773, 0.780],
    'p': [0.0006, 0.018, '4e-09']})

fig, ax = plt.subplots(figsize=(6.2, 3.4))
x = np.arange(len(cap))
ax.bar(x - .2, cap.fusion, .4, label='Fusion (ours)', color=GREEN)
ax.bar(x + .2, cap.base, .4, label='Base-paper equivalent', color=BLUE)
for i, r in cap.iterrows():
    gap = r.fusion - r.base
    ax.text(i, max(r.fusion, r.base) + .02,
            f'+{gap:.3f}\np={r.p}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\n({n} rows)' for c, n in zip(cap.cap, cap.rows)])
ax.set_ylim(0.5, 0.95)
ax.set_ylabel('PR-AUC (5x5 grouped CV)')
ax.set_title('The fusion gap survives capping variants per root photo')
ax.legend(loc='lower right', frameon=False)
fig.tight_layout()
save(fig, 'fig15_robustness_caps')

# ================================================ fig16 ELA vs resolution
RESOLUTIONS = [256, 384, 512, 768, 1024]
N_ROOTS = 50
OUT_Q = 88   # same JPEG quality the dataset uses (gen_images.OUT_QUALITY)


def long_resize(im, r):
    w, h = im.size
    s = r / max(w, h)
    return im.resize((max(8, round(w * s)), max(8, round(h * s))), Image.LANCZOS)


def jpeg_roundtrip(im, q=OUT_Q):
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out


def feats(im):
    d = {**FX.ela_features(im), **FX.noise_features(im)}
    return list(d.values())


# --- collect roots big enough to be downscaled to every resolution
if os.path.exists('ela_sweep_cache.csv'):
    sweep = pd.read_csv('ela_sweep_cache.csv')
    print('loaded cached resolution sweep', sweep.shape)
else:
    roots = []
    for dirpath, _, files in os.walk(f'{REPO}/source_photos'):
        for fn in files:
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                p = os.path.join(dirpath, fn)
                try:
                    with Image.open(p) as im:
                        if max(im.size) >= 1024:
                            roots.append(p)
                except Exception:
                    pass
    rng = np.random.default_rng(0)
    roots = list(rng.choice(roots, size=min(N_ROOTS, len(roots)), replace=False))
    print(f'{len(roots)} roots for resolution sweep')

    rows = []
    for r_i, rp in enumerate(roots):
        with Image.open(rp) as im0:
            im0 = im0.convert('RGB')
            for R in RESOLUTIONS:
                base = long_resize(im0, R)
                # genuine: resized + one JPEG save (identical history to dataset genuine rows)
                rows.append((R, 0, r_i, feats(jpeg_roundtrip(base))))
                # manipulated: same resize, then a dataset manipulation, then one save
                manip_rng = random.Random(10_000 * r_i + R)
                manipulated, op, _ = GI.apply_manipulation(base.copy(), manip_rng)
                rows.append((R, 1, r_i, feats(jpeg_roundtrip(manipulated))))
        if (r_i + 1) % 10 == 0:
            print(f'  {r_i+1}/{len(roots)} roots done')

    sweep = pd.DataFrame([(R, y, g, *f) for R, y, g, f in rows],
                         columns=['res', 'y', 'group'] +
                                 [f'f{i}' for i in range(len(rows[0][3]))])
    sweep.to_csv('ela_sweep_cache.csv', index=False)

N_ELA, N_NOISE = 7, 4
ela_cols = [f'f{i}' for i in range(N_ELA)]
noise_cols = [f'f{i}' for i in range(N_ELA, N_ELA + N_NOISE)]
all_cols = ela_cols + noise_cols

result = {}
for label, cols in [('ELA', ela_cols), ('Noise residual', noise_cols),
                    ('Combined', all_cols)]:
    aucs = []
    for R in RESOLUTIONS:
        d = sweep[sweep.res == R]
        y, X, g = d.y.values, d[cols].values, d.group.values
        fold_auc = []
        for tr, te in GroupKFold(5).split(X, y, g):
            m = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
            m.fit(X[tr], y[tr])
            fold_auc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        aucs.append(np.mean(fold_auc))
    result[label] = aucs

fig, ax = plt.subplots(figsize=(6.6, 3.8))
for (label, aucs), c, mk in zip(result.items(), [BLUE, ORANGE, GREEN], ['o', 's', 'D']):
    ax.plot(RESOLUTIONS, aucs, marker=mk, color=c, lw=1.8, label=label)
ax.axhline(0.5, color='grey', ls='--', lw=1, label='chance (0.5)')
ax.axvspan(200, 512, color='red', alpha=.07)
ax.text(300, 0.95, 'evidence\ncollapses', ha='center', fontsize=8, color=RED)
ax.axvline(1024, color=GREEN, ls=':', lw=1.5)
ax.text(1000, 0.60, 'this dataset\n(1024px)', ha='right', fontsize=8, color=GREEN)
ax.set_xscale('log', base=2)
ax.set_xticks(RESOLUTIONS)
ax.set_xticklabels(RESOLUTIONS)
ax.set_ylim(0.4, 1.0)
ax.set_xlabel('Output image resolution (long edge, px)')
ax.set_ylabel('ROC-AUC (GroupKFold on root photo)')
ax.set_title('Manipulation evidence disappears as images are downscaled')
ax.legend(loc='lower left', frameon=False)
fig.tight_layout()
save(fig, 'fig16_ela_resolution')

# ================================================ fig17 pipeline diagram
fig, ax = plt.subplots(figsize=(12.4, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')


def box(x0, x1, y0, y1, text, fc, fontsize=8.5, bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle='round,pad=0.4,rounding_size=1.2',
        facecolor=fc, edgecolor='black', linewidth=1.0))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold' if bold else 'normal')


def arrow(x0, x1, y, color='black'):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.6))


# column 1
box(1, 16, 19, 31,
    'LAYER 1\n836 real product photos\n15 categories\n(826 usable, >=1000px)',
    '#dfeaf6', bold=False)
# column 2
box(20, 38, 33, 45, 'Genuine (1,650)\nreal photo\nhonest review text', '#dff3e4')
box(20, 38, 7, 26,
    'Fake (1,650)\n- visual manipulation\n  (copy-move, splice, noise)\n'
    '- text deception\n- coordinated reuse\n  (105 campaigns)', '#fbdedd')
# column 3 feature families
box(42, 60, 37, 46, 'Text (9 feats)\nMiniLM score, lexical', '#dfeaf6')
box(42, 60, 28.5, 36.5, 'Image forensics (23)\nELA, noise, CNN, pHash', '#f6e7d8')
box(42, 60, 20, 28, 'Behaviour (5)\nrating, burst, timing', '#e6e0f2')
box(42, 60, 11.5, 19.5, 'Reuse + graph (4)\nbipartite communities,\nCLIP cross-modal', '#d8f0ea')
box(42, 60, 3, 11, 'Split by root photo\n(no photo in train and test)', '#eeeeee')
# column 4
box(64, 79, 17, 33,
    'FUSION MODEL\nExtra Trees / Random\nForest, 41 features', '#dff3e4', bold=True)
# column 5
box(83, 99, 27, 45,
    '3-way output\nGenuine\nFake\nNeeds verification', '#dfeaf6', bold=False)
box(83, 99, 8, 23,
    'FastAPI backend\n(127.0.0.1:8000)\n-> Chrome extension\nAmazon / Flipkart / Meesho',
    '#f6e7d8')

arrow(16.5, 19.5, 29)
arrow(16.5, 19.5, 14)
arrow(38.5, 41.5, 39)
arrow(38.5, 41.5, 32.5)
arrow(38.5, 41.5, 24)
arrow(38.5, 41.5, 15.5)
arrow(60.5, 63.5, 29)
arrow(60.5, 63.5, 15)
arrow(79.5, 82.5, 36)
arrow(91, 91, 26.5, color='#555555')
ax.set_title('Pipeline: constructed dataset -> 41 forensic/behavioural features -> fusion -> browser extension',
             fontsize=11, pad=8)
save(fig, 'fig17_pipeline')

# ================================================ fig18 qualitative examples
rev = pd.read_csv(f'{OUT}/reviews_balanced.csv')
manip_row = rev[(rev.fraud_type == 'visual_manipulation') &
                rev.root_file.notna()].iloc[0]

edited_path = f"{OUT}/{manip_row.image_file}"
original_path = f"{REPO}/source_photos/{manip_row.root_file}"
if not os.path.exists(original_path):   # fall back to any root that exists
    for cand in rev.root_file.dropna().unique():
        p = f'{REPO}/source_photos/{cand}'
        if os.path.exists(p):
            original_path, manip_row = p, rev[rev.root_file == cand].iloc[0]
            edited_path = f"{OUT}/{manip_row.image_file}"
            break


def ela_map(im):
    im = im.convert('RGB')
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=90)
    buf.seek(0)
    rec = Image.open(buf)
    rec.load()
    diff = np.abs(np.asarray(im).astype(np.int16) -
                  np.asarray(rec).astype(np.int16)).mean(axis=2)
    return diff


def thumb(path_or_im, maxpx=420):
    im = Image.open(path_or_im) if isinstance(path_or_im, str) else path_or_im
    im = im.convert('RGB')
    im.thumbnail((maxpx, maxpx), Image.LANCZOS)
    return im


with Image.open(original_path) as o, Image.open(edited_path) as e:
    o, e = o.convert('RGB'), e.convert('RGB')
    ela = ela_map(e)
    o_th, e_th = thumb(o), thumb(e)

# reuse burst: biggest campaign sharing one photo
camp = (rev[rev.fraud_type == 'coordinated_reuse']
        .groupby('campaign_id').size().sort_values(ascending=False).index[0])
burst = rev[rev.campaign_id == camp].copy()
burst['t'] = pd.to_datetime(burst.timestamp)
burst = burst.sort_values('t')
span_h = (burst.t.max() - burst.t.min()).total_seconds() / 3600

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
axes = axes.ravel()

axes[0].imshow(o_th)
axes[0].set_title(f'(a) Original product photo\n({manip_row.category}, real photo)', fontsize=9)
axes[1].imshow(e_th)
axes[1].set_title(f'(b) Edited variant\n({manip_row.transform_chain})', fontsize=9)
im2 = axes[2].imshow(ela, cmap='inferno')
axes[2].set_title('(c) ELA error map of (b)\nbright = suspicious error level', fontsize=9)
fig.colorbar(im2, ax=axes[2], fraction=.046, pad=.04)

ax = axes[3]
ax.scatter(burst.t, range(len(burst)), color=PURPLE, s=60, zorder=3)
for i, (_, r) in enumerate(burst.iterrows()):
    ax.annotate(r.reviewer_id, (r.t, i), textcoords='offset points',
                xytext=(8, -3), fontsize=7)
ax.axvspan(burst.t.min(), burst.t.max(), color=PURPLE, alpha=.10)
ax.set_title(f'(d) Coordinated reuse: one photo,\n{len(burst)} accounts within {span_h:.1f} h', fontsize=9)
ax.set_ylabel('Account (posting order)')
ax.set_xlabel('Time')
ax.tick_params(axis='x', rotation=20, labelsize=8)
ax.set_yticks([])

for ax in axes[:3]:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('The three fraud mechanisms, as the model sees them', fontsize=12, y=1.0)
fig.tight_layout()
save(fig, 'fig18_qualitative')

# ================================================ fig19 review text examples
cat = manip_row.category
genuine = rev[(rev.is_fake == 0) & (rev.category == cat)].drop_duplicates(
    'review_text').head(3)
deceptive = rev[(rev.fraud_type == 'text_deception') & (rev.category == cat)].drop_duplicates(
    'review_text').head(3)

fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax.axis('off')
ax.text(0.24, 0.97, 'GENUINE reviews', ha='center', fontsize=11,
        fontweight='bold', color=GREEN)
ax.text(0.76, 0.97, 'FAKE reviews (clean photo, deceptive text)', ha='center',
        fontsize=11, fontweight='bold', color=RED)
for i in range(3):
    y = 0.88 - i * 0.29
    ax.text(0.02, y, textwrap.fill(genuine.iloc[i].review_text, 52),
            fontsize=8.5, va='top', color='#1a1a1a')
    ax.text(0.52, y, textwrap.fill(deceptive.iloc[i].review_text, 52),
            fontsize=8.5, va='top', color='#1a1a1a')
for i in range(2):
    ax.axhline(0.62 - i * 0.29, color='grey', lw=.6)
ax.axvline(0.5, color='grey', lw=.6)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f'Example review text, {cat} category (template-generated, see limitations)',
             fontsize=10)
save(fig, 'fig19_review_examples')

print('extra figures done ->', os.path.abspath(FIGDIR))
