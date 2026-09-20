"""
record_text_range.py -- write the TEXT distribution the model was trained on into model_meta.json.

The backend needs to know when a review is outside what the model has ever seen. Training
reviews are template-generated and short (genuine: 4-34 words, at most 3 sentences); a real
Amazon review can run to 200+ words. Judging one with a model fitted to the other is
extrapolation, and it showed up as a 78% "Likely fake" on an obviously real review.

Storing the range here keeps the guard in sync with whatever the model was actually trained on,
instead of a constant in the backend that silently goes stale.

    python record_text_range.py
"""
import json

import numpy as np
import pandas as pd

import features as F

OUT = '../out'
tr = pd.read_csv(f'{OUT}/train.csv')
X = F.build_all(tr)['TEXT']
stats = {
    'txt_len_p99': float(np.percentile(X.txt_len, 99)),
    'txt_len_max': float(X.txt_len.max()),
    'txt_sent_count_p99': float(np.percentile(X.txt_sent_count, 99)),
    'txt_sent_count_max': float(X.txt_sent_count.max()),
    'genuine_txt_len_max': float(X.txt_len[tr.is_fake.values == 0].max()),
    'note': ('Range of the template-generated training text. A review well outside it cannot be '
             'judged by this text model; the backend neutralises the TEXT features instead of '
             'extrapolating. See guide section 29.'),
}
p = f'{OUT}/model/model_meta.json'
meta = json.load(open(p))
meta['text_training_range'] = stats
json.dump(meta, open(p, 'w'), indent=1)
print(json.dumps(stats, indent=1))
