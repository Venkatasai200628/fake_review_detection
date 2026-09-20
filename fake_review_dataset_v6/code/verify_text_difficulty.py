"""
verify_text_difficulty.py -- is "fake text caught 100%" a real skill, or template memorising?

Our review text is generated from templates. If a test fake sentence is a near-copy of a
training fake sentence, then any transformer will score 100% on it and that number says
nothing about the model. This measures it instead of assuming it:

  1. exact duplicate rate between test texts and train texts
  2. nearest-neighbour TF-IDF cosine similarity of each test text to the training set
  3. the accuracy a trivial nearest-neighbour "copy the label of the most similar
     training review" rule gets -- the memorising floor
  4. vocabulary overlap fake vs genuine

Output: ../out/model_comparison/TEXT_DIFFICULTY_AUDIT.md
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

OUT = '../out'
DST = f'{OUT}/model_comparison'
tr = pd.read_csv(f'{OUT}/train.csv')
te = pd.read_csv(f'{OUT}/test_balanced.csv')
TR, TE = tr.review_text.fillna(''), te.review_text.fillna('')

lines = []
P = lines.append
P('# Is "fake text caught = 100%" real skill or template memorising?\n')
P(f'Train {len(tr)} reviews, test {len(te)} reviews, text generated from templates.\n')

# 1 exact duplicates
trset = set(TR.str.strip().str.lower())
dup = TE.str.strip().str.lower().isin(trset)
P('## 1. Exact duplicate texts between test and train\n')
P(f'- test texts that appear verbatim in train: **{int(dup.sum())} / {len(te)} ({dup.mean():.1%})**')
for ft in ['genuine', 'text_deception', 'coordinated_reuse', 'visual_manipulation']:
    m = te.fraud_type == ft
    P(f'  - {ft}: {int(dup[m].sum())}/{int(m.sum())} ({dup[m].mean():.1%})')
P('')

# 2 nearest neighbour similarity
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(pd.concat([TR, TE]))
A, B = vec.transform(TR), vec.transform(TE)
S = linear_kernel(B, A)                      # cosine, tf-idf is l2 normalised
nn = S.argmax(1)
sim = S.max(1)
P('## 2. How close is each test text to its nearest training text? (TF-IDF cosine, 1.0 = identical)\n')
P('| test group | mean nearest-neighbour similarity | median | share with sim > 0.9 |')
P('|---|---|---|---|')
for ft in ['genuine', 'text_deception', 'coordinated_reuse', 'visual_manipulation']:
    m = (te.fraud_type == ft).values
    P(f'| {ft} | {sim[m].mean():.3f} | {np.median(sim[m]):.3f} | {(sim[m] > 0.9).mean():.1%} |')
P('')

# 3 memorising floor: copy the nearest training review's label
nn_lab = tr.is_fake.values[nn]
acc = (nn_lab == te.is_fake.values).mean()
P('## 3. The memorising floor: "copy the label of the most similar training review"\n')
P('This rule has no learning in it at all. Whatever it scores is the part of the text task that is pure look-up.\n')
P(f'- accuracy of the 1-nearest-neighbour copy rule: **{acc:.3f}**')
for ft in ['genuine', 'text_deception', 'coordinated_reuse', 'visual_manipulation']:
    m = (te.fraud_type == ft).values
    lab = 0 if ft == 'genuine' else 1
    P(f'  - {ft}: {(nn_lab[m] == lab).mean():.1%} correct')
P('')

# 4 vocabulary overlap
def vocab(s):
    return set(w for t in s for w in str(t).lower().split())
vf = vocab(tr[tr.is_fake == 1].review_text)
vg = vocab(tr[tr.is_fake == 0].review_text)
P('## 4. Vocabulary\n')
P(f'- words used only in fake training reviews: {len(vf - vg)}')
P(f'- words used only in genuine training reviews: {len(vg - vf)}')
P(f'- shared: {len(vf & vg)}  (Jaccard {len(vf & vg)/len(vf | vg):.2f})')
P('')

open(f'{DST}/TEXT_DIFFICULTY_AUDIT.md', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
