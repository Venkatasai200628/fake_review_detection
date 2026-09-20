"""
compare_paper_models.py -- the SPECIFIC models from the surveyed papers,
re-built and trained/tested on OUR dataset (same train reviews, same unseen
test photos, same metrics as our model). CPU-only, so a few pieces are
simplified; every simplification is listed in NOTES and in the output table.

  P38 FRD-LSTM      Qayyum 2023: contextual word representations (DCWR) + BiLSTM
  P30 BSTC          Lu 2023: BERT + TextCNN (SKEP sentiment branch omitted)
  P40 DeBERTa       Geetha 2025: fine-tuned DeBERTa-v3 (Monarch-Butterfly tuning omitted)
  P15 A-LSTM        Xu 2024: attention BiLSTM on text + behavioural features
  P39 DHMFRD-TER    Duma 2024: text CNN + emotion + rating deep hybrid
  P11 FRIDRC        Hou 2025: fine-tuned text encoder + ViT image (CLIP ViT, frozen), fused
  P02 Shan          Shan 2021: rating-sentiment / content / language inconsistency + Random Forest

Output: ../out/model_comparison/paper_models_on_our_data.csv / .md
    python compare_paper_models.py            (all)
    python compare_paper_models.py P38 P15    (only some)
"""
import os
os.environ.setdefault("USE_TF", "0")          # PyTorch only; the installed Keras 3 breaks TF loading
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import json
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import features as F

OUT = '../out'
DST = f'{OUT}/model_comparison'
torch.set_num_threads(8)
F.CNN_SCORES_CSV = f'{OUT}/cnn_manip_scores_split.csv'
F.TEXT_SCORES_CSV = f'{OUT}/text_model_scores_split.csv'

full = pd.read_csv(f'{OUT}/reviews_full.csv')
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
ho = pd.read_csv(f'{OUT}/holdout_realistic.csv')
POS = {r: i for i, r in enumerate(full.review_id)}
I = lambda part: np.array([POS[r] for r in part.review_id])
ytr = tr.is_fake.values
TEXTS = full.review_text.fillna('').tolist()
RES_CSV = f'{DST}/paper_models_on_our_data.csv'

NOTES = {
    'P38': 'DCWR approximated by frozen DistilBERT contextual token vectors',
    'P30': 'SKEP sentiment-knowledge branch omitted (not available in English/PyTorch)',
    'P40': 'deberta-v3-small; Monarch-Butterfly hyper-parameter search omitted',
    'P15': 'word embeddings trained from scratch',
    'P39': 'emotion from a pretrained English emotion classifier (7 emotions)',
    'P11': 'image encoder = CLIP ViT-B/32, frozen (CPU); text encoder DistilRoBERTa fine-tuned; profile info not available',
    'P02': 'sentiment from DistilBERT-SST2; 9 inconsistency features + 8 base text features',
}

# ------------------------------------------------------------ our model
blocks = F.build_all(full)
for k in blocks:
    blocks[k] = blocks[k].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    blocks[k].index = full.review_id.values
Xall = pd.concat(blocks.values(), axis=1)
cols = json.load(open(f'{OUT}/model/model_meta.json'))['features']
ours_model = joblib.load(f'{OUT}/model/fusion_rf.joblib')
OURS_TE = ours_model.predict_proba(Xall.loc[te.review_id, cols].values)[:, 1]
OURS_HO = ours_model.predict_proba(Xall.loc[ho.review_id, cols].values)[:, 1]
BEH = StandardScaler().fit(blocks['BEHAVIOUR'].loc[tr.review_id]).transform(blocks['BEHAVIOUR'].values)


def score(key, name, paper, p_te, p_ho, secs):
    yh = (p_te >= 0.5).astype(int)
    t = te.assign(f=yh)
    y = te.is_fake.values
    ours_ok = (OURS_TE >= 0.5).astype(int) == y
    them_ok = yh == y
    n01, n10 = int((ours_ok & ~them_ok).sum()), int((~ours_ok & them_ok).sum())
    r = {'key': key, 'model': name, 'paper': paper,
         'accuracy': accuracy_score(y, yh), 'f1': f1_score(y, yh), 'roc_auc': roc_auc_score(y, p_te),
         'pr_auc_holdout15': average_precision_score(ho.is_fake, p_ho),
         'edited_photo_caught': t[t.fraud_type == 'visual_manipulation'].f.mean(),
         'fake_text_caught': t[t.fraud_type == 'text_deception'].f.mean(),
         'coordinated_caught': t[t.fraud_type == 'coordinated_reuse'].f.mean(),
         'false_alarms': t[t.fraud_type == 'genuine'].f.mean(),
         'ours_better_on': n01, 'it_better_on': n10,
         'mcnemar_p_vs_ours': binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0,
         'seconds': round(secs), 'simplification': NOTES.get(key, '')}
    print(f"  => {name:32s} acc {r['accuracy']:.3f}  F1 {r['f1']:.3f}  ROC {r['roc_auc']:.3f}  "
          f"edited-photo {r['edited_photo_caught']:.0%}  fake-text {r['fake_text_caught']:.0%}  "
          f"false alarms {r['false_alarms']:.1%}  [{secs:.0f}s]", flush=True)
    # Save per-row probabilities. Without these, re-checking a comparison (confidence
    # intervals, a different McNemar pairing, a threshold sweep) means retraining all seven
    # models for ~2.5 h. With them it is a file read.
    os.makedirs(f'{DST}/preds', exist_ok=True)
    pd.DataFrame({'review_id': te.review_id, 'p_fake': p_te}).to_csv(f'{DST}/preds/{key}_test.csv', index=False)
    pd.DataFrame({'review_id': ho.review_id, 'p_fake': p_ho}).to_csv(f'{DST}/preds/{key}_holdout.csv', index=False)

    old = pd.read_csv(RES_CSV) if os.path.exists(RES_CSV) else pd.DataFrame()
    if len(old):
        old = old[old.key != key]
    pd.concat([old, pd.DataFrame([r])], ignore_index=True).to_csv(RES_CSV, index=False)


# ------------------------------------------------------------ helpers
def hf(name, cls=None):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(name)
    mdl = (cls or AutoModel).from_pretrained(name)
    return tok, mdl


def fit_torch(model, make_batch, n, epochs, lr, bs=16, log=''):
    """Generic training loop. make_batch(idx) -> (inputs dict/tuple, labels)."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, 1 - s / steps))
    lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for i, idx in enumerate(np.array_split(np.random.default_rng(ep).permutation(n), max(1, n // bs))):
            xb, yb = make_batch(idx)
            loss = lossf(model(*xb), torch.tensor(yb, dtype=torch.float32))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        print(f"    {log} epoch {ep + 1}/{epochs} ({time.time() - t0:.0f}s)", flush=True)
    model.eval()


@torch.no_grad()
def predict_torch(model, make_x, rows, bs=64):
    out = []
    for i in range(0, len(rows), bs):
        out.append(torch.sigmoid(model(*make_x(rows[i:i + bs]))).numpy())
    return np.concatenate(out)


class Vocab:
    def __init__(self, texts, min_count=2, max_len=64):
        from collections import Counter
        c = Counter(w for t in texts for w in self.tok(t))
        self.itos = ['<pad>', '<unk>'] + [w for w, n in c.items() if n >= min_count]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_len = max_len

    @staticmethod
    def tok(t):
        return re.findall(r"[a-z']+|[!?.]", t.lower())

    def encode(self, texts):
        M = np.zeros((len(texts), self.max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.stoi.get(w, 1) for w in self.tok(t)][:self.max_len]
            M[i, :len(ids)] = ids
        return M


# ------------------------------------------------------------ P38 FRD-LSTM
def p38():
    t0 = time.time()
    tok, enc = hf('distilbert-base-uncased')
    enc.eval()
    H = np.zeros((len(TEXTS), 64, 768), dtype=np.float16)
    Mk = np.zeros((len(TEXTS), 64), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(TEXTS), 64):
            b = tok(TEXTS[i:i + 64], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
            H[i:i + 64] = enc(**b).last_hidden_state.numpy().astype(np.float16)
            Mk[i:i + 64] = b['attention_mask'].numpy()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
            self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))

        def forward(self, h, m):
            o, _ = self.lstm(h)
            mean = (o * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True)
            mx = o.masked_fill(m.unsqueeze(-1) == 0, -1e4).max(1).values
            return self.out(torch.cat([mean, mx], 1)).squeeze(1)

    net = Net()
    torch.manual_seed(0)
    x = lambda rows: (torch.tensor(H[rows].astype(np.float32)), torch.tensor(Mk[rows]))
    itr = I(tr)
    fit_torch(net, lambda idx: (x(itr[idx]), ytr[idx]), len(itr), 8, 1e-3, bs=32, log='FRD-LSTM')
    score('P38', 'FRD-LSTM (DCWR + BiLSTM)', '38 Qayyum 2023',
          predict_torch(net, x, I(te)), predict_torch(net, x, I(ho)), time.time() - t0)


# ------------------------------------------------------------ transformer fine-tunes
def fine_tune(key, name, paper, model_name, head='cls', epochs=3, lr=3e-5, extra=None):
    """head: 'cls' mean-pool+linear | 'textcnn' | 'fusion' (concat extra vectors)"""
    t0 = time.time()
    tok, enc = hf(model_name)
    hid = enc.config.hidden_size
    torch.manual_seed(0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            if head == 'textcnn':
                self.convs = nn.ModuleList([nn.Conv1d(hid, 100, k) for k in (2, 3, 4)])
                self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(300, 1))
            elif head == 'fusion':
                self.img = nn.Sequential(nn.Linear(extra.shape[1], 256), nn.ReLU())
                self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(hid + 256, 256), nn.ReLU(), nn.Linear(256, 1))
            else:
                self.out = nn.Sequential(nn.Dropout(0.1), nn.Linear(hid, 1))

        def forward(self, ids, mask, ex=None):
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            if head == 'textcnn':
                z = h.transpose(1, 2)
                z = torch.cat([torch.relu(c(z)).max(2).values for c in self.convs], 1)
                return self.out(z).squeeze(1)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            if head == 'fusion':
                pooled = torch.cat([pooled, self.img(ex)], 1)
            return self.out(pooled).squeeze(1)

    net = Net()
    enc_all = tok(TEXTS, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    EX = torch.tensor(extra, dtype=torch.float32) if extra is not None else None

    def x(rows):
        rows = torch.tensor(rows)
        args = (enc_all['input_ids'][rows], enc_all['attention_mask'][rows])
        return args + ((EX[rows],) if EX is not None else ())

    itr = I(tr)
    fit_torch(net, lambda idx: (x(itr[idx]), ytr[idx]), len(itr), epochs, lr, bs=16, log=key)
    score(key, name, paper, predict_torch(net, x, I(te)), predict_torch(net, x, I(ho)), time.time() - t0)


def p30():
    fine_tune('P30', 'BSTC (BERT + TextCNN)', '30 Lu 2023', 'bert-base-uncased', head='textcnn', epochs=2, lr=3e-5)


def p40():
    fine_tune('P40', 'DeBERTa-v3 fine-tuned', '40 Geetha 2025', 'microsoft/deberta-v3-small', epochs=3, lr=4e-5)


def p11():
    C = np.load(f'{OUT}/clip_image_embeddings.npy')
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    fine_tune('P11', 'FRIDRC (text enc. + ViT, fused)', '11 Hou 2025', 'distilroberta-base',
              head='fusion', epochs=3, lr=3e-5, extra=C)


# ------------------------------------------------------------ P15 A-LSTM + behaviour
def p15():
    t0 = time.time()
    V = Vocab(tr.review_text.tolist())
    W = V.encode(TEXTS)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(V.itos), 128, padding_idx=0)
            self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
            self.att = nn.Linear(128, 1)
            self.out = nn.Sequential(nn.Linear(128 + BEH.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, w, b):
            o, _ = self.lstm(self.emb(w))
            a = self.att(o).squeeze(-1).masked_fill(w == 0, -1e4).softmax(1)
            return self.out(torch.cat([(o * a.unsqueeze(-1)).sum(1), b], 1)).squeeze(1)

    net = Net()
    torch.manual_seed(0)
    x = lambda rows: (torch.tensor(W[rows]), torch.tensor(BEH[rows], dtype=torch.float32))
    itr = I(tr)
    fit_torch(net, lambda idx: (x(itr[idx]), ytr[idx]), len(itr), 12, 2e-3, bs=32, log='A-LSTM')
    score('P15', 'A-LSTM + behaviour', '15 Xu 2024', predict_torch(net, x, I(te)), predict_torch(net, x, I(ho)),
          time.time() - t0)


# ------------------------------------------------------------ P39 DHMFRD-TER
def p39():
    t0 = time.time()
    from transformers import pipeline
    emo = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base',
                   top_k=None, truncation=True)
    E = []
    for i in range(0, len(TEXTS), 64):
        for r in emo(TEXTS[i:i + 64]):
            E.append([d['score'] for d in sorted(r, key=lambda d: d['label'])])
    E = np.array(E, dtype=np.float32)
    R = np.eye(5, dtype=np.float32)[full.rating.clip(1, 5).values - 1]
    V = Vocab(tr.review_text.tolist())
    W = V.encode(TEXTS)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(len(V.itos), 128, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(128, 64, k, padding=k // 2) for k in (3, 4, 5)])
            self.out = nn.Sequential(nn.Linear(192 + E.shape[1] + 5, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, w, e, r):
            z = self.emb(w).transpose(1, 2)
            z = torch.cat([torch.relu(c(z)).max(2).values for c in self.convs], 1)
            return self.out(torch.cat([z, e, r], 1)).squeeze(1)

    net = Net()
    torch.manual_seed(0)
    x = lambda rows: (torch.tensor(W[rows]), torch.tensor(E[rows]), torch.tensor(R[rows]))
    itr = I(tr)
    fit_torch(net, lambda idx: (x(itr[idx]), ytr[idx]), len(itr), 12, 2e-3, bs=32, log='DHMFRD-TER')
    score('P39', 'DHMFRD-TER (text + emotion + rating)', '39 Duma 2024',
          predict_torch(net, x, I(te)), predict_torch(net, x, I(ho)), time.time() - t0)


# ------------------------------------------------------------ P02 Shan inconsistency
def p02():
    t0 = time.time()
    from transformers import pipeline
    sent = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', truncation=True)
    pos_p = lambda texts: np.array([r['score'] if r['label'] == 'POSITIVE' else 1 - r['score']
                                    for r in sent(texts, batch_size=64)])
    s_review = pos_p(TEXTS)
    # sentence-level sentiment -> language inconsistency (mixed polarity inside one review)
    sents = [[s for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()] or [t] for t in TEXTS]
    flat = [s for ss in sents for s in ss]
    fp = pos_p(flat)
    k, s_std, s_range = 0, [], []
    for ss in sents:
        v = fp[k:k + len(ss)]; k += len(ss)
        s_std.append(v.std()); s_range.append(v.max() - v.min())
    rating_n = (full.rating.values - 1) / 4
    # content inconsistency: how different this review is from the product's other reviews
    T = np.load(f'{DST}/text_emb_all-MiniLM-L6-v2.npy')
    cont = np.zeros(len(full)); cont_max = np.zeros(len(full))
    for _, idx in full.groupby('product_id').indices.items():
        S = T[idx] @ T[idx].T
        np.fill_diagonal(S, np.nan)
        cont[idx] = 1 - np.nanmean(S, 1); cont_max[idx] = np.nanmax(S, 1)
    prod_mean = full.groupby('product_id').rating.transform('mean').values
    Xs = np.column_stack([
        rating_n, s_review, np.abs(rating_n - s_review),                  # rating-sentiment inconsistency
        (rating_n - 0.5) * (s_review - 0.5) < 0,                          # opposite polarity flag
        np.array(s_std), np.array(s_range),                               # language inconsistency
        cont, cont_max,                                                   # content inconsistency
        np.abs(full.rating.values - prod_mean),                           # rating vs product consensus
        blocks['TEXT'][[c for c in blocks['TEXT'].columns if c != 'txt_minilm_p']].values])
    rf = RandomForestClassifier(500, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=0)
    rf.fit(Xs[I(tr)], ytr)
    score('P02', 'Shan inconsistency + RF', '02 Shan 2021',
          rf.predict_proba(Xs[I(te)])[:, 1], rf.predict_proba(Xs[I(ho)])[:, 1], time.time() - t0)


def write_md():
    d = pd.read_csv(RES_CSV)
    order = ['P02', 'P15', 'P38', 'P39', 'P30', 'P40', 'P11']
    d['o'] = d.key.map({k: i for i, k in enumerate(order)})
    d = d.sort_values('o')
    y = te.is_fake.values
    yh = (OURS_TE >= 0.5).astype(int)
    t = te.assign(f=yh)
    md = ['| Paper | Re-built model | Accuracy | F1 | ROC-AUC | PR-AUC @15% | Edited photo caught | Fake text caught | False alarms | Ours better on / it better on (McNemar p) |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in d.itertuples():
        md.append(f"| {r.paper} | {r.model} | {r.accuracy:.3f} | {r.f1:.3f} | {r.roc_auc:.3f} | {r.pr_auc_holdout15:.3f} | "
                  f"{r.edited_photo_caught:.0%} | {r.fake_text_caught:.0%} | {r.false_alarms:.1%} | "
                  f"{r.ours_better_on} / {r.it_better_on} (p={r.mcnemar_p_vs_ours:.1e}) |")
    md.append(f"| **ours** | **forensic fusion** | **{accuracy_score(y, yh):.3f}** | **{f1_score(y, yh):.3f}** | "
              f"**{roc_auc_score(y, OURS_TE):.3f}** | **{average_precision_score(ho.is_fake, OURS_HO):.3f}** | "
              f"**{t[t.fraud_type == 'visual_manipulation'].f.mean():.0%}** | {t[t.fraud_type == 'text_deception'].f.mean():.0%} | "
              f"{t[t.fraud_type == 'genuine'].f.mean():.1%} | – |")
    md += ['', 'Simplifications (CPU-only re-builds):'] + [f"- {r.paper}: {r.simplification}" for r in d.itertuples()]
    open(f'{DST}/paper_models_on_our_data.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n' + '\n'.join(md))


if __name__ == '__main__':
    todo = sys.argv[1:] or ['P02', 'P15', 'P39', 'P38', 'P40', 'P11', 'P30']
    fns = {'P02': p02, 'P15': p15, 'P38': p38, 'P39': p39, 'P30': p30, 'P40': p40, 'P11': p11}
    for k in todo:
        print(f"\n### {k}: {NOTES[k]}", flush=True)
        try:
            fns[k]()
        except Exception as e:
            print(f"  !! {k} failed: {type(e).__name__}: {e}", flush=True)
    write_md()
