"""
text_model.py -- stronger TEXT evidence: MiniLM sentence embeddings + Logistic
Regression, turned into one feature `txt_minilm_p` ("probability the TEXT is
deceptive"). Chosen in compare_text_models.py: 99% of the text ceiling, fewest
false alarms, no fine-tuning, 90 MB, fast enough for the browser extension.

The 8 hand-made text features stay; this is added next to them.

NO LEAKAGE -- same rules as cnn_forensics.py
  text_model_scores.csv        5-fold cross-fitting grouped by root photo
                               (used by crossval.py)
  text_model_scores_split.csv  trained on TRAIN rows only: train rows get
                               inner out-of-fold scores, test rows are scored
                               by a model that never saw them
                               (used by train_model.py, modality_ablation.py,
                               and the extension backend)
Also saves model/text_minilm_lr.joblib for the backend.

    python text_model.py
"""
import hashlib
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

OUT = '../out'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
EMB_CACHE = f'{OUT}/model_comparison/text_emb_all-MiniLM-L6-v2.npy'
C = 4.0
_tok = _mdl = None


def embed(texts, bs=64):
    """Mean-pooled, L2-normalised MiniLM sentence vectors (384-d)."""
    global _tok, _mdl
    from transformers import AutoTokenizer, AutoModel
    if _mdl is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _mdl = AutoModel.from_pretrained(MODEL_NAME).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = _tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=96,
                     return_tensors='pt')
            h = _mdl(**b).last_hidden_state
            m = b['attention_mask'].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy())
    return np.vstack(out)


def fit(E, y):
    return LogisticRegression(max_iter=3000, C=C, class_weight='balanced').fit(E, y)


def oof(E, y, groups):
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(E, y, groups):
        s[te] = fit(E[tr], y[tr]).predict_proba(E[te])[:, 1]
    return s


def load_embeddings(texts):
    """Cached MiniLM vectors, keyed by a hash of the TEXT they were built from.

    The cache used to be keyed by filename alone. When regen_text.py rewrote every review
    for v6.2, this silently kept returning vectors of the OLD text, so `txt_minilm_p` --
    the model's single most important feature -- described reviews that no longer existed.
    The stamp file makes a stale cache impossible: change the text and it is rebuilt."""
    stamp = EMB_CACHE + '.sha256'
    key = hashlib.sha256('\n'.join(texts).encode('utf-8')).hexdigest()
    if os.path.exists(EMB_CACHE) and os.path.exists(stamp) and open(stamp).read().strip() == key:
        return np.load(EMB_CACHE)
    if os.path.exists(EMB_CACHE):
        print('  text has changed since the embedding cache was built -- re-embedding')
    E = embed(texts)
    os.makedirs(os.path.dirname(EMB_CACHE), exist_ok=True)
    np.save(EMB_CACHE, E)
    open(stamp, 'w').write(key)
    return E


def main():
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    E = load_embeddings(df.review_text.fillna('').tolist())
    assert len(E) == len(df)
    y = (df.label == 'Fake').astype(int).values          # NV rows count as not-fake here
    g = df.root_image_id.values

    s_all = oof(E, y, g)
    pd.DataFrame({'review_id': df.review_id, 'txt_minilm_p': s_all}).to_csv(
        f'{OUT}/text_model_scores.csv', index=False)

    tr = (df.split == 'train').values
    s = np.zeros(len(y))
    s[tr] = oof(E[tr], y[tr], g[tr])
    final = fit(E[tr], y[tr])
    s[~tr] = final.predict_proba(E[~tr])[:, 1]
    pd.DataFrame({'review_id': df.review_id, 'txt_minilm_p': s}).to_csv(
        f'{OUT}/text_model_scores_split.csv', index=False)
    joblib.dump(final, f'{OUT}/model/text_minilm_lr.joblib')
    print("saved text_model_scores.csv, text_model_scores_split.csv, model/text_minilm_lr.joblib")


if __name__ == '__main__':
    main()
