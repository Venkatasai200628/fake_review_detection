"""
make_paper_figures.py -- generate every figure needed for the paper.

Reads only files already in ../out (no retraining), plus one fast pass of the
saved fusion model to draw ROC/PR curves and confusion matrices.

Output: ../../paper_figures/figNN_*.png  (+ .pdf versions)

    python make_paper_figures.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

import features as F

OUT = '../out'
MC = f'{OUT}/model_comparison'
FIGDIR = '../../paper_figures'
os.makedirs(FIGDIR, exist_ok=True)

# ----------------------------------------------------------------- style
sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'figure.facecolor': 'white'})

PAL = sns.color_palette('colorblind')
BLUE, ORANGE, GREEN, RED, PURPLE = PAL[0], PAL[1], PAL[2], PAL[4], PAL[5]


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(f'{FIGDIR}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}')


def modality_of(f):
    if f.startswith('txt_'):
        return 'Text'
    if f.startswith('beh_'):
        return 'Behaviour'
    if f.startswith('ela_') or f.startswith('noise_') or f == 'img_cnn_manip':
        return 'Image forensics'
    if f.startswith('g_'):
        return 'Reuse graph'
    if f.startswith('img_') or f.startswith('cm_') or f.startswith('clip_'):
        return 'Image reuse / CLIP'
    return 'Other'


# =============================================== 1. dataset composition
rev = pd.read_csv(f'{OUT}/reviews_balanced.csv')
fake = rev[rev.is_fake == 1]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

# (a) class balance
counts = rev.label.value_counts().reindex(['Genuine', 'Fake'])
axes[0].bar(counts.index, counts.values, color=[GREEN, RED], width=.55)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontsize=10)
axes[0].set_title('(a) Class balance')
axes[0].set_ylabel('Reviews')

# (b) fake sub-types
ft = fake.fraud_type.value_counts().reindex(
    ['visual_manipulation', 'text_deception', 'coordinated_reuse'])
axes[1].bar([s.replace('_', '\n') for s in ft.index], ft.values, color=[ORANGE, PURPLE, BLUE], width=.55)
for i, v in enumerate(ft.values):
    axes[1].text(i, v + 12, str(v), ha='center', fontsize=10)
axes[1].set_title('(b) Fake sub-types (equal thirds)')
axes[1].set_ylabel('Reviews')

# (c) categories
cat = rev.category.value_counts().sort_values()
axes[2].barh(cat.index, cat.values, color=TEAL if (TEAL := sns.color_palette()[3]) else BLUE)
axes[2].set_title('(c) Reviews per product category')
axes[2].set_xlabel('Reviews')
axes[2].tick_params(axis='y', labelsize=7)

fig.tight_layout()
save(fig, 'fig01_dataset_composition')

# =============================================== 2. feature importance
imp = pd.read_csv(f'{OUT}/model/feature_importance.csv', index_col=0)
imp = imp.sort_values('importance', ascending=False).head(20)
imp['modality'] = [modality_of(i) for i in imp.index]

fig, ax = plt.subplots(figsize=(7, 4.8))
order = imp.index[::-1]
sns.barplot(data=imp.reset_index(), y='index', x='importance', hue='modality',
            order=order, dodge=False, ax=ax)
ax.set_xlabel('Mean decrease in impurity (importance)')
ax.set_ylabel('')
ax.set_title('Top 20 of 41 fusion features')
ax.legend(title='Modality', bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=False)
ax.tick_params(axis='y', labelsize=8)
fig.tight_layout()
save(fig, 'fig02_feature_importance')

# =============================================== 3. modality ablation
abl = pd.read_csv(f'{OUT}/model/modality_ablation_final.csv')
abl = abl.rename(columns={'features': 'Features set'})
abl['short'] = (abl['Features set']
                .str.replace(' only', '', regex=False)
                .str.replace('ALL (text+image+behaviour+graph)', 'ALL 41 (ours)', regex=False)
                .str.replace('TEXT + BEHAVIOUR + GRAPH (no image)', 'Text+Behav+Graph', regex=False))

fig, ax = plt.subplots(figsize=(7.2, 3.6))
x = np.arange(len(abl))
ax.bar(x - .2, abl.accuracy, .4, label='Accuracy', color=BLUE)
ax.bar(x + .2, abl.pr_auc_holdout15, .4, label='PR-AUC (realistic 15% fake)', color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(abl.short, rotation=30, ha='right')
ax.set_ylim(0, 1.02)
ax.set_ylabel('Score')
ax.set_title('What each modality contributes')
ax.legend()
fig.tight_layout()
save(fig, 'fig03_modality_ablation')

# =============================================== 4. cross-val modalities
cv = pd.read_csv(f'{OUT}/crossval_results_final.csv')
cv['short'] = cv.config.str.replace('FUSION (ours)', 'Fusion (ours)', regex=False) \
                     .str.replace('-only', '', regex=False) \
                     .str.replace('base-paper-equivalent', 'Base-paper equiv.', regex=False)

fig, ax = plt.subplots(figsize=(6.6, 3.6))
yerr = cv['PR-AUC'].str.extract(r'\+/-\s*([\d.]+)')[0].astype(float)  # std over folds
ax.bar(cv.short, cv['PR-AUC_mean'], yerr=yerr, capsize=3,
       color=[GREEN if 'ours' in s else BLUE for s in cv.short])
ax.set_ylabel('PR-AUC (25-fold CV)')
ax.set_title('25-fold cross-validation: fusion beats every single modality')
ax.set_ylim(0, 1.0)
for i, m in enumerate(cv['PR-AUC_mean']):
    ax.text(i, m + .03, f'{m:.3f}', ha='center', fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
fig.tight_layout()
save(fig, 'fig04_crossval_modalities')

# =============================================== 5. per fraud type heatmap
pf = pd.read_csv(f'{OUT}/crossval_per_fraud_type_final.csv', index_col=0)
pf.index = pf.index.str.replace('FUSION (ours)', 'Fusion (ours)', regex=False) \
                   .str.replace('-only', '', regex=False) \
                   .str.replace('base-paper-equivalent', 'Base-paper equiv.', regex=False)

fig, ax = plt.subplots(figsize=(5.4, 3.4))
sns.heatmap(pf, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Detection rate'})
ax.set_title('Detection rate per fake type (CV)')
ax.set_xlabel('Fake type')
ax.set_ylabel('')
fig.tight_layout()
save(fig, 'fig05_per_fraud_type')

# =============================================== 6. fusion model variants
fm = pd.read_csv(f'{MC}/fusion_models.csv')
fm = fm.rename(columns={fm.columns[0]: 'model'})
fm['short'] = fm.model.str.replace(r' 500.*', '', regex=True) \
                     .str.replace(r' 1000.*', '', regex=True) \
                     .str.replace('Soft vote ET+RF+HGB', 'Soft vote (3)', regex=False) \
                     .str.replace('Logistic regression', 'Logistic Reg.', regex=False) \
                     .str.replace('Stacking (RF+XGB+HGB -> LR)', 'Stacking', regex=False) \
                     .str.replace('HistGradientBoosting', 'HistGB', regex=False) \
                     .str.replace('XGBoost deep', 'XGB deep', regex=False)

fig, ax = plt.subplots(figsize=(7.4, 3.6))
x = np.arange(len(fm))
ax.bar(x - .2, fm.accuracy, .4, label='Accuracy (test)', color=BLUE)
ax.bar(x + .2, fm.roc_auc, .4, label='ROC-AUC (test)', color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(fm.short, rotation=30, ha='right')
ax.set_ylim(.8, 1.0)
ax.set_title('Fusion model selection: all tree ensembles cluster together')
ax.set_ylabel('Score')
ax.legend()
fig.tight_layout()
save(fig, 'fig06_fusion_models')

# =============================================== 7. rebuilt published models
pm = pd.read_csv(f'{MC}/paper_models_on_our_data.csv')
pm = pm.sort_values('accuracy')
labels = [f"{r.key} {r.model.split(' +')[0].split('(')[0].strip()[:22]}" for _, r in pm.iterrows()]

fig, ax = plt.subplots(figsize=(7, 4.2))
y = np.arange(len(pm))
ax.barh(y - .2, pm.accuracy, .4, label='Accuracy', color=BLUE)
ax.barh(y + .2, pm.f1, .4, label='F1', color=ORANGE)
ours_acc, ours_f1 = 0.943, 0.943   # from modality_ablation_final.csv, ALL row
ax.axvline(ours_acc, color=GREEN, ls='--', lw=2, label=f'Ours (acc {ours_acc:.3f})')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Score')
ax.set_title('7 re-built published models vs our fusion (same data, same split)')
ax.set_xlim(0, 1.06)
ax.legend(loc='upper right')
fig.tight_layout()
save(fig, 'fig07_paper_models_vs_ours')

# =============================================== 8. baselines vs ours
bl = pd.read_csv(f'{MC}/baselines_on_our_data.csv')
bl['short'] = bl.method.str.replace(r'^(B\d) .*', r'\1', regex=True) \
                       .str.replace('OURS forensic fusion', 'OURS', regex=False)

fig, ax = plt.subplots(figsize=(6.8, 3.6))
x = np.arange(len(bl))
ax.bar(x - .2, bl.accuracy, .4, label='Accuracy', color=BLUE)
ax.bar(x + .2, bl.pr_auc_holdout15, .4, label='PR-AUC (15% fake)', color=ORANGE)
ax.set_xticks(x)
ax.set_xticklabels(bl.short)
for i, r in bl.iterrows():
    ax.text(i - .2, r.accuracy + .01, f'{r.accuracy:.2f}', ha='center', fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Baseline families (B1-B5) vs our forensic fusion')
ax.legend()
fig.tight_layout()
save(fig, 'fig08_baselines_vs_ours')

# =============================================== 9. LOCO generalisation
loco = pd.read_csv(f'{MC}/loco.csv').sort_values('ROC-AUC')

fig, ax = plt.subplots(figsize=(7, 4.4))
y = np.arange(len(loco))
ax.barh(y - .22, loco['ROC-AUC'], .44, label='ROC-AUC (fusion)', color=BLUE)
ax.barh(y + .22, loco.accuracy, .44, label='Accuracy', color=GREEN)
ax.set_yticks(y)
ax.set_yticklabels(loco['held-out category'], fontsize=8)
ax.axvline(loco['ROC-AUC'].mean(), color=RED, ls=':', lw=1.5,
           label=f'mean ROC-AUC {loco["ROC-AUC"].mean():.3f}')
ax.set_xlim(.6, 1.0)
ax.set_xlabel('Score')
ax.set_title('Leave-one-category-out: unseen product categories')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
fig.tight_layout()
save(fig, 'fig09_loco_generalisation')

# ============================ 10-12. curves from the saved fusion model
print('building features for ROC/PR/confusion ...')
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores_split.csv'
full = pd.read_csv(f'{OUT}/reviews_full.csv')
X = pd.concat(F.build_all(full).values(), axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
X.index = full.review_id.values
cols = json.load(open(f'{OUT}/model/model_meta.json'))['features']
model = joblib.load(f'{OUT}/model/fusion_rf.joblib')

sets = [('test_balanced', pd.read_csv(f'{OUT}/test_balanced.csv')),
        ('holdout_realistic', pd.read_csv(f'{OUT}/holdout_realistic.csv'))]
pred = {n: model.predict_proba(X.loc[p.review_id, cols].values)[:, 1] for n, p in sets}

# --- fig 10: ROC + PR
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
for (n, p), c in zip(sets, [BLUE, ORANGE]):
    y, s = p.is_fake.values, pred[n]
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    axes[0].plot(fpr, tpr, color=c, lw=1.8,
                 label=f'{n.replace("_", " ")} (AUC={roc_auc_score(y, s):.3f})')
    axes[1].plot(rec, prec, color=c, lw=1.8,
                 label=f'{n.replace("_", " ")} (AP={average_precision_score(y, s):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=.8)
axes[0].set_xlabel('False positive rate'); axes[0].set_ylabel('True positive rate')
axes[0].set_title('(a) ROC'); axes[0].legend(loc='lower right')
axes[1].axhline(sets[1][1].is_fake.mean(), color='grey', ls=':', lw=1, label='random (15% fake)')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('(b) Precision–Recall'); axes[1].legend(loc='upper right')
fig.tight_layout()
save(fig, 'fig10_roc_pr_curves')

# --- fig 11: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
for ax, (n, p) in zip(axes, sets):
    cm = confusion_matrix(p.is_fake.values, (pred[n] >= .5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Genuine', 'Fake'], yticklabels=['Genuine', 'Fake'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{n.replace("_", " ")} (threshold 0.5)')
fig.tight_layout()
save(fig, 'fig11_confusion_matrices')

# --- fig 12: threshold sweep on realistic holdout
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for (n, p), c in zip(sets, [BLUE, ORANGE]):
    prec, rec, thr = precision_recall_curve(p.is_fake.values, pred[n])
    m = min(len(thr), len(prec), len(rec))   # align lengths across sklearn versions
    ax.plot(thr[:m], prec[:m], color=c, ls='-', label=f'{n.replace("_", " ")} precision')
    ax.plot(thr[:m], rec[:m], color=c, ls='--', label=f'{n.replace("_", " ")} recall')
ax.set_xlabel('Decision threshold')
ax.set_ylabel('Score')
ax.set_title('Precision / recall vs threshold (trade-off for deployment)')
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
save(fig, 'fig12_threshold_tradeoff')

# =============================================== 13. stability + CI summary
# values from verify_results.py run (../out/VERIFY_RESULTS.md)
ci = pd.DataFrame({
    'metric': ['Accuracy', 'F1', 'ROC-AUC', 'PR-AUC'] * 2,
    'value': [.932, .929, .967, .976, .973, .908, .980, .955],
    'lo':    [.914, .907, .953, .966, .954, .845, .955, .913],
    'hi':    [.949, .948, .978, .984, .988, .958, .997, .986],
    'set':   ['Balanced test (50% fake)'] * 4 + ['Realistic holdout (15% fake)'] * 4})

fig, ax = plt.subplots(figsize=(6.6, 3.4))
metrics4 = ['Accuracy', 'F1', 'ROC-AUC', 'PR-AUC']
for m_i, m in enumerate(metrics4):
    for s_name, off, col in [('Balanced test (50% fake)', -0.15, BLUE),
                             ('Realistic holdout (15% fake)', 0.15, ORANGE)]:
        r = ci[(ci.metric == m) & (ci['set'] == s_name)].iloc[0]
        ax.errorbar(r.value, m_i + off, xerr=[[r.value - r.lo], [r.hi - r.value]],
                    fmt='o', color=col, capsize=3)
ax.set_yticks(range(4))
ax.set_yticklabels(metrics4)
ax.set_xlim(.8, 1.02)
ax.set_xlabel('Metric value with 95% bootstrap CI (2,000 resamples, grouped by photo)')
ax.set_title('Uncertainty of the final model')
ax.legend(handles=[plt.Line2D([], [], marker='o', color=BLUE, ls='', label='Balanced test'),
                   plt.Line2D([], [], marker='o', color=ORANGE, ls='', label='Realistic holdout')],
          loc='upper left')
fig.tight_layout()
save(fig, 'fig13_confidence_intervals')

print('all figures in', os.path.abspath(FIGDIR))
