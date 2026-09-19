"""
compare_text_models.py -- which TEXT model is best for our project?

Same train photos -> same unseen test photos for every candidate. Text only.
  1. hand-made 8 features + Random Forest            (what we use now)
  2. TF-IDF word + character n-grams + Logistic Regression
  3. MiniLM-L6 sentence embeddings + Logistic Regression
  4. MPNet-base sentence embeddings + Logistic Regression
  5. DistilRoBERTa fine-tuned end-to-end

Remember the ceiling: 135 of the 405 test fakes (edited photos) have genuine
text by design, so NO text model can exceed accuracy 0.833 here.
Honest caveat: review text is template-generated (guide 11 item 8). A model
that memorises our templates scores high here but would transfer less well
to real reviews -- the fine-tuned transformer is the most likely to do this.

Output: ../out/model_comparison/text_models.csv
"""
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score)
from scipy.sparse import hstack

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
CEIL = 0.8333
torch.set_num_threads(8)
os.makedirs(DST, exist_ok=True)

full = pd.read_csv(f'{OUT}/reviews_full.csv')
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
ytr, yte, yho = tr.is_fake.values, te.is_fake.values, ho.is_fake.values


def score(name, p_te, p_ho, secs):
    yh = (p_te >= 0.5).astype(int)
    t = te.assign(f=yh)
    r = {'model': name,
         'accuracy': accuracy_score(yte, yh), 'f1': f1_score(yte, yh),
         'roc_auc': roc_auc_score(yte, p_te), 'pr_auc_holdout15': average_precision_score(yho, p_ho),
         'share_of_ceiling': accuracy_score(yte, yh) / CEIL,
         'caught_text_deception': t[t.fraud_type == 'text_deception'].f.mean(),
         'caught_coordinated': t[t.fraud_type == 'coordinated_reuse'].f.mean(),
         'caught_edited_photo': t[t.fraud_type == 'visual_manipulation'].f.mean(),
         'false_alarm_genuine': t[t.fraud_type == 'genuine'].f.mean(),
         'seconds': round(secs)}
    print(f"  {name:34s} acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  ROC {r['roc_auc']:.3f}  "
          f"({r['share_of_ceiling']:.0%} of ceiling)  fake-text caught {r['caught_text_deception']:.0%}  "
          f"false alarms {r['false_alarm_genuine']:.0%}  [{secs:.0f}s]", flush=True)
    return r


rows = []

# 1 -- hand-made features (the current TEXT block)
t0 = time.time()
TF = F.text_features(full)
TF.index = full.review_id.values
m = RandomForestClassifier(500, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=0)
m.fit(TF.loc[tr.review_id].values, ytr)
rows.append(score('1 hand-made 8 features + RF', m.predict_proba(TF.loc[te.review_id].values)[:, 1],
                  m.predict_proba(TF.loc[ho.review_id].values)[:, 1], time.time() - t0))

# 2 -- TF-IDF
t0 = time.time()
w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, sublinear_tf=True)
Xtr = hstack([w.fit_transform(tr.review_text), c.fit_transform(tr.review_text)])
lr = LogisticRegression(max_iter=3000, C=2.0, class_weight='balanced').fit(Xtr, ytr)
tf = lambda s: hstack([w.transform(s), c.transform(s)])
rows.append(score('2 TF-IDF words+chars + LogReg', lr.predict_proba(tf(te.review_text))[:, 1],
                  lr.predict_proba(tf(ho.review_text))[:, 1], time.time() - t0))


# 3/4 -- sentence embeddings
def embed(model_name, texts, bs=64):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=96, return_tensors='pt')
            h = mdl(**b).last_hidden_state
            msk = b['attention_mask'].unsqueeze(-1).float()
            v = (h * msk).sum(1) / msk.sum(1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy())
    return np.vstack(out)


for tag, name in [('3 MiniLM-L6 embeddings + LogReg', 'sentence-transformers/all-MiniLM-L6-v2'),
                  ('4 MPNet-base embeddings + LogReg', 'sentence-transformers/all-mpnet-base-v2')]:
    t0 = time.time()
    E = embed(name, full.review_text.fillna('').tolist())
    E = pd.DataFrame(E, index=full.review_id.values)
    np.save(f"{DST}/text_emb_{name.split('/')[-1]}.npy", E.values)
    lr = LogisticRegression(max_iter=3000, C=4.0, class_weight='balanced').fit(E.loc[tr.review_id].values, ytr)
    rows.append(score(tag, lr.predict_proba(E.loc[te.review_id].values)[:, 1],
                      lr.predict_proba(E.loc[ho.review_id].values)[:, 1], time.time() - t0))

# 5 -- DistilRoBERTa fine-tuned
t0 = time.time()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
name = 'distilroberta-base'
tok = AutoTokenizer.from_pretrained(name)
mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
torch.manual_seed(0)
opt = torch.optim.AdamW(mdl.parameters(), lr=3e-5, weight_decay=0.01)
enc = lambda s: tok(list(s), padding=True, truncation=True, max_length=64, return_tensors='pt')
EPOCHS, BS = 3, 16
steps = EPOCHS * ((len(tr) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=steps)
mdl.train()
for ep in range(EPOCHS):
    perm = np.random.default_rng(ep).permutation(len(tr))
    for i in range(0, len(tr), BS):
        idx = perm[i:i + BS]
        b = enc(tr.review_text.values[idx])
        loss = mdl(**b, labels=torch.tensor(ytr[idx])).loss
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
    print(f"    distilroberta epoch {ep + 1}/{EPOCHS} done ({time.time() - t0:.0f}s)", flush=True)
mdl.eval()


@torch.no_grad()
def prob(s):
    out = []
    for i in range(0, len(s), 64):
        out.append(torch.softmax(mdl(**enc(s[i:i + 64])).logits, -1)[:, 1].numpy())
    return np.concatenate(out)


rows.append(score('5 DistilRoBERTa fine-tuned', prob(te.review_text.values), prob(ho.review_text.values),
                  time.time() - t0))

d = pd.DataFrame(rows).round(3)
d.to_csv(f'{DST}/text_models.csv', index=False)
print(f"\nsaved {DST}/text_models.csv   (ceiling for any text model: accuracy {CEIL:.3f})")
