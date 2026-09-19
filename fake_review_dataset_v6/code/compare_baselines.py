"""
compare_baselines.py -- FAIR comparison with the literature: the published
METHODS re-implemented and run on OUR dataset, same train photos, same unseen
test photos, same metrics. (Numbers copied from papers are measured on their own
datasets and cannot be ranked against ours -- guide section 12.6.)

  B1  TF-IDF n-grams + linear SVM          papers 17, 23, 27, 29 (classic text ML)
  B2  fine-tuned transformer (DistilRoBERTa) papers 09, 13, 26, 30, 38, 40 (BERT/RoBERTa/DeBERTa)
  B3  sentence embeddings + behaviour       papers 12, 15, 31 (BERT + behaviour features)
  B4  He et al. 2022 base paper             paper 33: network + metadata + text + its image
                                            feature (similarity to the product's other photos)
  B5  multimodal embedding fusion           paper 11 FRIDRC style: text encoder + image
                                            encoder (CLIP ViT), concatenated -> classifier
  OURS  forensic fusion (train_model.py)

Output: ../out/model_comparison/baselines_on_our_data.csv (+ .md table)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.svm import LinearSVC
from scipy.sparse import hstack

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores_split.csv'

full = pd.read_csv(f'{OUT}/reviews_full.csv')
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
ytr = tr.is_fake.values
pos = {r: i for i, r in enumerate(full.review_id)}
I = lambda part: np.array([pos[r] for r in part.review_id])
rows = []


def score(name, papers, p_te, p_ho):
    yh = (p_te >= 0.5).astype(int)
    t = te.assign(f=yh)
    r = {'method': name, 'represents papers': papers,
         'accuracy': accuracy_score(te.is_fake, yh), 'f1': f1_score(te.is_fake, yh),
         'roc_auc': roc_auc_score(te.is_fake, p_te),
         'pr_auc_holdout15': average_precision_score(ho.is_fake, p_ho),
         'edited_photo_caught': t[t.fraud_type == 'visual_manipulation'].f.mean(),
         'fake_text_caught': t[t.fraud_type == 'text_deception'].f.mean(),
         'coordinated_caught': t[t.fraud_type == 'coordinated_reuse'].f.mean(),
         'false_alarms': t[t.fraud_type == 'genuine'].f.mean()}
    rows.append(r)
    print(f"  {name:42s} acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  ROC {r['roc_auc']:.3f}  "
          f"edited-photo {r['edited_photo_caught']:.0%}  fake-text {r['fake_text_caught']:.0%}", flush=True)


blocks = F.build_all(full)
for k in blocks:
    blocks[k] = blocks[k].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    blocks[k].index = full.review_id.values
X = lambda names, part: pd.concat([blocks[b] for b in names], axis=1).loc[part.review_id].values

# B1 -- TF-IDF + linear SVM
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)
Xt = hstack([w.fit_transform(tr.review_text), c.fit_transform(tr.review_text)])
tf = lambda s: hstack([w.transform(s), c.transform(s)])
svm = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight='balanced'), cv=5).fit(Xt, ytr)
score('B1 TF-IDF + linear SVM', '17, 23, 27, 29',
      svm.predict_proba(tf(te.review_text))[:, 1], svm.predict_proba(tf(ho.review_text))[:, 1])

# B2 -- fine-tuned transformer (result from compare_text_models.py, same split)
tm = pd.read_csv(f'{DST}/text_models.csv').set_index('model').loc['5 DistilRoBERTa fine-tuned']
rows.append({'method': 'B2 fine-tuned transformer (DistilRoBERTa)', 'represents papers': '09, 13, 26, 30, 38, 40',
             'accuracy': tm.accuracy, 'f1': tm.f1, 'roc_auc': tm.roc_auc, 'pr_auc_holdout15': tm.pr_auc_holdout15,
             'edited_photo_caught': tm.caught_edited_photo, 'fake_text_caught': tm.caught_text_deception,
             'coordinated_caught': tm.caught_coordinated, 'false_alarms': tm.false_alarm_genuine})
print(f"  {'B2 fine-tuned transformer (DistilRoBERTa)':42s} acc {tm.accuracy:.3f}  (from compare_text_models.py)")

# B3 -- sentence embeddings + behaviour features
T = np.load(f'{DST}/text_emb_all-MiniLM-L6-v2.npy')
Bt = lambda part: np.hstack([T[I(part)], X(['BEHAVIOUR'], part)])
rf = RandomForestClassifier(500, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=0)
rf.fit(Bt(tr), ytr)
score('B3 sentence embeddings + behaviour', '12, 15, 31',
      rf.predict_proba(Bt(te))[:, 1], rf.predict_proba(Bt(ho))[:, 1])

# B4 -- He et al. 2022: network + metadata + text + THEIR image feature
#       (mean CLIP similarity of a review photo to the other photos of the same product)
C = np.load(f'{OUT}/clip_image_embeddings.npy')
C = C / np.linalg.norm(C, axis=1, keepdims=True)
he_img = np.zeros(len(full))
for prod, idx in full.groupby('product_id').indices.items():
    S = C[idx] @ C[idx].T
    np.fill_diagonal(S, np.nan)
    he_img[idx] = np.nanmean(S, axis=1)
text8 = [c for c in blocks['TEXT'].columns if c != 'txt_minilm_p']      # their text = hand-made features
Hb = lambda part: np.hstack([blocks['TEXT'][text8].loc[part.review_id].values,
                             X(['BEHAVIOUR', 'GRAPH'], part), he_img[I(part)][:, None]])
rf = RandomForestClassifier(500, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=0)
rf.fit(Hb(tr), ytr)
score('B4 He et al. 2022 (base paper, with image sim.)', '33',
      rf.predict_proba(Hb(te))[:, 1], rf.predict_proba(Hb(ho))[:, 1])

# B5 -- FRIDRC-style multimodal embedding fusion: text encoder + image encoder (ViT)
Mm = lambda part: np.hstack([T[I(part)], C[I(part)]])
lr = LogisticRegression(max_iter=5000, C=2.0, class_weight='balanced').fit(Mm(tr), ytr)
score('B5 multimodal embeddings (FRIDRC-style)', '11',
      lr.predict_proba(Mm(te))[:, 1], lr.predict_proba(Mm(ho))[:, 1])

# OURS
model = joblib.load(f'{OUT}/model/fusion_rf.joblib')
cols = json.load(open(f'{OUT}/model/model_meta.json'))['features']
Xall = pd.concat(blocks.values(), axis=1)
score('OURS forensic fusion', '-',
      model.predict_proba(Xall.loc[te.review_id, cols].values)[:, 1],
      model.predict_proba(Xall.loc[ho.review_id, cols].values)[:, 1])

d = pd.DataFrame(rows)
d.round(3).to_csv(f'{DST}/baselines_on_our_data.csv', index=False)
md = ['| Method | Represents papers | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | Coordinated caught | False alarms |',
      '|---|---|---|---|---|---|---|---|---|---|']
for r in d.itertuples():
    md.append(f"| {r.method} | {r._2} | {r.accuracy:.3f} | {r.f1:.3f} | {r.roc_auc:.3f} | {r.pr_auc_holdout15:.3f} | "
              f"{r.edited_photo_caught:.0%} | {r.fake_text_caught:.0%} | {r.coordinated_caught:.0%} | {r.false_alarms:.1%} |")
open(f'{DST}/baselines_on_our_data.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n' + '\n'.join(md))
