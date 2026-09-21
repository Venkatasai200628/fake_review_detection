"""
pipeline.py -- turns RAW reviews scraped from a product page into the exact same
features the model was trained on, and scores them.

The golden rule (guide section 17.2): never re-implement a feature here. Every
feature comes from the training code itself (features.py, forensics.py,
forensic_maps.py, clip_embedding.py, cnn_forensics.py), imported from
dataset/fake_review_dataset_v6/code. If training and serving computed a feature
differently, the model would silently break (training/serving skew).

How a new review is scored
  1. download its photo, resize to <=1024px and save once as JPEG q88
     (exactly what gen_images.save_derived() did to every training photo)
  2. photo -> pHash, CLIP vector, ELA/noise numbers, forensic maps -> CNN score
  3. text -> CLIP text vector -> photo-vs-text similarity
  4. add the new rows to the IMAGE MEMORY (all 3,348 dataset photos + whatever
     this server has seen) and run features.build_all() over the lot, so reuse,
     burst timing and the account graph are computed against everything known
  5. keep only the new rows, feed them to the fusion / text / image models
  6. turn the scores into Genuine / Needs Verification / Fake + plain reasons

Plugging in a better model later = save it in MODEL_DIR (fusion_rf.joblib +
model_meta.json) and restart. check_model() refuses to start if the model
expects a feature this pipeline cannot produce.
"""
import io
import json
import os
import re
import sys
import hashlib
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import requests
from dateutil import parser as dparser
from PIL import Image

import config

sys.path.insert(0, config.CODE_DIR)
import features as F            # noqa: E402  (training feature code)
import forensics                # noqa: E402
import gen_images as GI         # noqa: E402
import forensic_maps as FM      # noqa: E402

# training code uses paths relative to code/; the backend runs elsewhere
F.REUSE_EMB_NPY = os.path.join(config.OUT_DIR, 'reuse_embeddings_resnet50.npy')
F.REUSE_IDS_CSV = os.path.join(config.OUT_DIR, 'reuse_embeddings_ids.csv')

Image.MAX_IMAGE_PIXELS = None
UA = {'User-Agent': 'Mozilla/5.0 (fake-review-research-demo; local, read-only)'}
MEM_COLS = ['review_id', 'product_id', 'reviewer_id', 'review_text', 'rating',
            'timestamp', 'image_id', 'phash', 'embedding']

# Reasons that explain the CHECK rather than accuse the review. If only these fired, the card
# has not actually said why it flagged anything, and drivers() is asked for the real answer.
NEUTRAL_REASON = re.compile(
    r'no photo|can be normal|less reliable|too small|evidence was ignored|'
    r'not enough evidence|no specific|too long for')

# Feature names as something a person can read.
PLAIN_FEATURE = {
    'txt_minilm_p': 'the wording reads like promotional copy to the language model',
    'txt_generic_ratio': 'high share of generic praise words',
    'txt_specific_ratio': 'little concrete, checkable detail',
    'txt_len': 'unusual review length',
    'txt_sent_count': 'unusual number of sentences',
    'txt_exclam': 'exclamation marks',
    'txt_upper_ratio': 'unusual amount of CAPITALS',
    'txt_unique_ratio': 'repetitive wording',
    'txt_exact_dup_count': 'the same text appears on other reviews',
    'rating': 'the star rating',
    'img_cnn_manip': 'the photo looks edited',
    'img_max_cosine': 'the photo closely matches another photo',
    'img_min_phash_dist': 'the photo is near-identical to another one',
    'img_reuse_count': 'the photo appears on several reviews',
    'img_reuse_distinct_reviewers': 'the photo appears under several accounts',
    'img_reuse_distinct_products': 'the photo appears on several products',
    'img_reuse_in_burst': 'the photo was reused inside a short burst',
    'img_burst_ratio': 'most of the photo reuse happened in one burst',
    'img_reuse_span_hours': 'how tightly the reuse is packed in time',
    'clip_cross_modal': 'the photo does not match what the words describe',
    'beh_reviewer_review_count': 'how many reviews this account has here',
    'beh_reviewer_span_hours': 'how tightly this account posts in time',
    'g_degree': 'how many accounts this one shares photos with',
    'g_community_size': 'the size of the account cluster it sits in',
    'g_clustering': 'how tightly that cluster is connected',
}


def _key(reviewer, text, photo_id):
    """Stable identity of a scraped review, so re-analysing a page does not make a review
    match ITSELF in the image memory.

    `photo_id` is the photo's pHash -- its CONTENT -- not its URL. Keying on the URL was a
    bug: Flipkart serves the same image from rukminim1 and rukminim2, and Amazon varies the
    size code (._SY88_ vs ._SY500_), so on every reload the same review looked new. It was
    stored again and then matched its own previous copy, which showed up as "identical text
    in 1 other review(s)", creeping reuse counts, and verdicts that changed between reloads
    (measured: risk 0.305 "Genuine" -> 0.434 "Needs verification" with only the host changed).

    The text is normalised too, because Amazon appends "more" to a collapsed review and drops
    it once expanded.
    """
    t = re.sub(r'\s+', ' ', text or '').strip().lower()
    t = re.sub(r'\s*\b(read more|more|less)\s*$', '', t)
    s = f"{(reviewer or '').strip().lower()}|{t}|{photo_id}"
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:16]


class Pipeline:
    def __init__(self, exclude_review_ids=()):
        out = config.OUT_DIR
        full = pd.read_csv(os.path.join(out, 'reviews_full.csv'))
        full = full[~full.review_id.isin(set(exclude_review_ids))]
        self.mem = full[MEM_COLS].copy()
        self.mem['ext_key'] = ''
        self.mem_forensic = pd.read_csv(os.path.join(out, 'forensic_features.csv')).set_index('review_id')
        self.mem_xmodal = pd.read_csv(os.path.join(out, 'clip_cross_modal.csv')).set_index('review_id')
        cnn_csv = os.path.join(out, 'cnn_manip_scores_split.csv')
        self.mem_cnn = pd.read_csv(cnn_csv).set_index('review_id') if os.path.exists(cnn_csv) else None
        txt_csv = os.path.join(out, 'text_model_scores_split.csv')
        self.mem_text = pd.read_csv(txt_csv).set_index('review_id') if os.path.exists(txt_csv) else None
        self.mem_reuse = F._load_reuse_embeddings()          # review_id -> ResNet-50 vector
        self.n_seen = 0
        self._load_models()
        self._clip = None
        self._cnn = None

    # ------------------------------------------------------------ models
    def _load_models(self):
        md = config.MODEL_DIR
        with open(os.path.join(md, 'model_meta.json')) as f:
            self.meta = json.load(f)
        self.fusion = joblib.load(os.path.join(md, 'fusion_rf.joblib'))
        blocks = self.meta.get('feature_blocks', {})
        self.side = {}
        for name, blk in [('text', 'TEXT'), ('image', 'IMAGE')]:
            p = os.path.join(md, f'{name}_rf.joblib')
            if os.path.exists(p) and blk in blocks:
                self.side[name] = (joblib.load(p), blocks[blk])
        # How far outside the training text range a review may go before the text evidence is
        # treated as unusable. Recorded by code/record_text_range.py so it tracks the model.
        rng_ = self.meta.get('text_training_range', {})
        self.text_len_limit = float(rng_.get('txt_len_max', 34)) * 1.5
        self.text_sent_limit = float(rng_.get('txt_sent_count_max', 4)) * 1.5

        bands = self.meta.get('decision_bands', {})
        self.t_low = bands.get('genuine_below', 0.35)
        self.t_high = bands.get('fake_above', 0.65)
        self.uses_cnn = 'img_cnn_manip' in self.meta['features']
        self.uses_text_model = 'txt_minilm_p' in self.meta['features']
        p = os.path.join(md, 'text_minilm_lr.joblib')
        self.text_lr = joblib.load(p) if os.path.exists(p) else None

    def _base_overrides(self):
        ov = {'forensic': self.mem_forensic, 'xmodal': self.mem_xmodal}
        if self.mem_cnn is not None:
            ov['cnn'] = self.mem_cnn
        if self.mem_text is not None:
            ov['text_model'] = self.mem_text
        if self.mem_reuse is not None:
            ov['reuse_emb'] = self.mem_reuse
        return ov

    def check_model(self):
        """Build features for 3 memory rows and confirm every model feature exists."""
        sample = self.mem.head(3)
        blocks = F.build_all(sample.reset_index(drop=True), overrides=self._base_overrides())
        have = set(pd.concat(blocks.values(), axis=1).columns)
        missing = [c for c in self.meta['features'] if c not in have]
        if missing:
            raise RuntimeError(f"model expects features this pipeline cannot make: {missing}")
        if self.uses_cnn and not os.path.exists(os.path.join(config.MODEL_DIR, 'forensic_cnn.pt')):
            raise RuntimeError("model uses img_cnn_manip but model/forensic_cnn.pt is missing")
        if self.uses_text_model and self.text_lr is None:
            raise RuntimeError("model uses txt_minilm_p but model/text_minilm_lr.joblib is missing")
        return len(self.meta['features'])

    def clip(self):
        if self._clip is None:
            import clip_embedding as CE
            CE._init()
            self._clip = CE
        return self._clip

    def cnn(self):
        if self._cnn is None:
            import torch
            import cnn_forensics as CF
            m = CF.ForensicCNN()
            m.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'forensic_cnn.pt'),
                                         map_location='cpu'))
            m.eval()
            self._cnn = (CF, m)
        return self._cnn

    # ------------------------------------------------------------ inputs
    @staticmethod
    def load_image(url):
        if url.startswith('/demo-images/') or '/demo-images/' in url:
            name = url.rsplit('/', 1)[-1]
            return Image.open(os.path.join(config.OUT_DIR, 'images', name)).convert('RGB')
        r = requests.get(url, headers=UA, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')

    @staticmethod
    def prepare(im):
        """Exactly what save_derived() did to every training photo."""
        long_edge = max(im.size)
        im = im.copy()
        im.thumbnail((config.OUT_SIZE, config.OUT_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=config.OUT_QUALITY)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out, long_edge

    @staticmethod
    def parse_date(s):
        try:
            # Amazon shows only the date; a missing time defaults to 12:00
            return dparser.parse(s, fuzzy=True, default=dt.datetime(2026, 1, 1, 12, 0))
        except Exception:
            return dt.datetime.now().replace(microsecond=0)

    @staticmethod
    def parse_rating(v):
        if isinstance(v, (int, float)):
            return int(round(v))
        m = re.search(r'(\d+(?:\.\d+)?)', str(v or ''))
        return int(round(float(m.group(1)))) if m else 3

    # ------------------------------------------------------------ main
    def analyze(self, payload, remember=True, raw_ids=False):
        reviews = payload.get('reviews', [])
        if not reviews:
            return {'reviews': [], 'summary': {'checked': 0}}
        product = payload.get('product_id') or hashlib.sha1(
            (payload.get('page_url') or 'page').encode()).hexdigest()[:10]
        product_id = f"P_EXT_{product}"
        title = payload.get('product_title') or ''

        listing_hashes = []
        for u in (payload.get('listing_images') or [])[:6]:
            try:
                im, _ = self.prepare(self.load_image(u))
                listing_hashes.append(GI.compute_phash(im))
            except Exception:
                pass

        rows, forensic, xmodal, cnn, meta, reuse = [], {}, {}, {}, [], {}
        keys = [None] * len(reviews)      # filled in below, once each photo has been hashed
        for k, r in enumerate(reviews):
            self.n_seen += 1
            rid = f"EXT{self.n_seen:06d}"
            info = {'id': r.get('id', str(k)), 'has_photo': False, 'photo_too_small': False,
                    'tamper_unusable': False, 'photo_edge': None,
                    'photo_error': None, 'listing_match': False}
            text = (r.get('text') or '').strip()
            im = None
            for u in (r.get('images') or [])[:1]:
                try:
                    raw = self.load_image(u)
                    if '/demo-images/' in u:
                        # already a saved dataset photo (1024px, one JPEG pass):
                        # re-saving would add a second pass training never saw
                        im, edge = raw, config.OUT_SIZE
                    else:
                        im, edge = self.prepare(raw)
                    info['has_photo'] = True
                    info['photo_edge'] = int(edge)
                    info['photo_too_small'] = edge < config.MIN_FORENSIC_EDGE
                    # too small for the tamper check to mean anything: the evidence is
                    # dropped below, not merely labelled (see config.MIN_CNN_EDGE)
                    info['tamper_unusable'] = edge < config.MIN_CNN_EDGE
                except Exception as e:
                    info['photo_error'] = str(e)[:120]
            if im is not None:
                ph = GI.compute_phash(im)
                CE = self.clip()
                e_img = CE.encode_images([im])[0]
                e_txt = CE.encode_text([text or title or 'product photo'])[0]
                forensic[rid] = forensics.analyse_image(im) if hasattr(forensics, 'analyse_image') \
                    else {**forensics.ela_features(im), **forensics.noise_features(im),
                          **forensics.copy_move_score(im)}
                xmodal[rid] = float(np.dot(e_img, e_txt))
                if self.uses_cnn:
                    CF, m = self.cnn()
                    cnn[rid] = float(CF.predict(m, FM.maps_from_image(im)[None])[0])
                if self.mem_reuse is not None:
                    import reuse_embedding as RE
                    reuse[rid] = RE.encode([im])[0]
                if listing_hashes:
                    info['listing_match'] = min(GI.phash_distance(ph, h) for h in listing_hashes) \
                        <= config.LISTING_MATCH_HAMMING
                emb = e_img.tolist()
            else:
                # no photo: a hash/vector that matches nothing; image features
                # are replaced by memory medians after build_all (see below)
                ph = hashlib.md5(rid.encode()).hexdigest()[:16]
                v = np.random.default_rng(self.n_seen).normal(size=512)
                emb = (v / np.linalg.norm(v)).tolist()
                if self.mem_reuse is not None:
                    w = np.random.default_rng(self.n_seen).normal(size=2048)
                    reuse[rid] = (w / np.linalg.norm(w)).astype(np.float32)
            # identity is based on the photo's CONTENT (its pHash), so the same review on a
            # reload is recognised even when the site hands out a different image URL
            keys[k] = _key(r.get('reviewer'), text, ph if im is not None else 'nophoto')
            rows.append({'review_id': rid,
                         'product_id': r.get('product_id', product_id) if raw_ids else product_id,
                         'reviewer_id': r.get('reviewer') if raw_ids
                         else f"EXT_{r.get('reviewer') or 'anonymous'}",
                         'review_text': text, 'rating': self.parse_rating(r.get('rating')),
                         'timestamp': self.parse_date(r.get('date') or '').isoformat(),
                         'image_id': f"IMG{rid}", 'phash': ph, 'embedding': json.dumps(emb),
                         'ext_key': keys[k]})
            meta.append(info)

        new = pd.DataFrame(rows)
        mem = self.mem[~self.mem.ext_key.isin(set(keys))]      # never match yourself
        corpus = pd.concat([mem, new], ignore_index=True)
        ov = {
            'forensic': pd.concat([self.mem_forensic, pd.DataFrame.from_dict(forensic, orient='index')]),
            'xmodal': pd.concat([self.mem_xmodal, pd.DataFrame(
                {'clip_cross_modal': pd.Series(xmodal, dtype=float)})]),
        }
        if self.mem_cnn is not None or cnn:
            parts = [self.mem_cnn] if self.mem_cnn is not None else []
            parts.append(pd.DataFrame({'img_cnn_manip': pd.Series(cnn, dtype=float)}))
            ov['cnn'] = pd.concat(parts)
        if self.uses_text_model:
            import text_model as TM
            tp = self.text_lr.predict_proba(TM.embed(new.review_text.fillna('').tolist()))[:, 1]
            ov['text_model'] = pd.concat([self.mem_text, pd.DataFrame(
                {'txt_minilm_p': tp}, index=new.review_id.values)])
        if self.mem_reuse is not None:
            ov['reuse_emb'] = {**self.mem_reuse, **reuse}
        blocks = F.build_all(corpus, overrides=ov)
        X = pd.concat(blocks.values(), axis=1).replace([np.inf, -np.inf], np.nan)
        X.index = corpus.review_id.values
        med = X.loc[mem.review_id].median()
        Xn = X.loc[new.review_id].copy()
        img_cols = list(blocks['IMAGE'].columns)
        # Tamper evidence only. The reuse features (pHash / embedding / burst) are NOT in this
        # list: ResNet-50 resizes to 224px anyway, so "is this the same photo" survives a small
        # thumbnail, while "was this photo edited" does not.
        tamper_cols = [c for c in img_cols
                       if c.startswith(('ela_', 'noise_', 'cm_')) or c == 'img_cnn_manip']
        text_cols = [c for c in blocks['TEXT'].columns if c in Xn.columns]
        for i, info in enumerate(meta):
            if not info['has_photo']:
                Xn.iloc[i, [Xn.columns.get_loc(c) for c in img_cols]] = med[img_cols].values
            elif info['tamper_unusable']:
                # Replace with the memory median so the model treats the tamper evidence as
                # "ordinary" rather than acting on a score that is close to a coin flip at this
                # size (ROC 0.63 at 360px vs 0.95 at 1024px). Same mechanism as a missing photo.
                Xn.iloc[i, [Xn.columns.get_loc(c) for c in tamper_cols]] = med[tamper_cols].values

            # Same idea for TEXT. The training reviews are template-generated: genuine ones run
            # 4-34 words and at most 4 sentences. A real marketplace review can run to 200+ words,
            # which is extrapolation, and the model resolves "unusual" as "suspicious" -- a
            # 223-word iPhone review scored 0.78 "Likely fake", driven by txt_minilm_p and
            # txt_exclam. Exclamation marks are a generator artefact: 88.7% of training reviews
            # containing "!" are fake, because the deceptive templates use them and the genuine
            # ones do not. Real people use them constantly.
            n_words = float(Xn.iloc[i].get('txt_len', 0) or 0)
            n_sent = float(Xn.iloc[i].get('txt_sent_count', 0) or 0)
            if n_words > self.text_len_limit or n_sent > self.text_sent_limit:
                info['text_unreliable'] = True
                info['text_words'] = int(n_words)
                Xn.iloc[i, [Xn.columns.get_loc(c) for c in text_cols]] = med[text_cols].values
        Xn = Xn.fillna(med).fillna(0.0)

        # kept so explain_review.py can ask "which feature caused this verdict?" without
        # rebuilding the whole corpus
        self.last_X, self.last_med = Xn, med
        p = self.fusion.predict_proba(Xn[self.meta['features']].values)[:, 1]
        side = {n: m.predict_proba(Xn[c].values)[:, 1] for n, (m, c) in self.side.items()}

        results = []
        for i, info in enumerate(meta):
            f = Xn.iloc[i]
            dec = 'Genuine' if p[i] < self.t_low else ('Fake' if p[i] > self.t_high else 'Needs_Verification')
            ev = self.photo_matches(new.iloc[i], mem, reuse) if info['has_photo'] else []
            reasons = self.reasons(f, info, ev, new.iloc[i])

            # If neither the text nor the photo could be judged, there is nothing left to
            # convict on. Saying "Fake" on no usable evidence is worse than saying "I cannot
            # tell", so the verdict is capped.
            no_text = info.get('text_unreliable')
            no_photo_ev = (not info['has_photo']) or info.get('tamper_unusable')
            if no_text and no_photo_ev and not ev and dec == 'Fake':
                dec = 'Needs_Verification'
                reasons.append('not enough evidence this tool can judge: treat as unverified, '
                               'not as fake')

            # A verdict without a reason is useless. If nothing above fired, say which features
            # actually drove the score.
            if dec != 'Genuine' and not any(r for r in reasons if not NEUTRAL_REASON.search(r)):
                reasons += self.drivers(Xn, med, i)
            results.append({
                'id': info['id'], 'risk': round(float(p[i]), 3), 'decision': dec,
                'text_score': round(float(side['text'][i]), 3) if 'text' in side else None,
                'image_score': round(float(side['image'][i]), 3) if ('image' in side and info['has_photo']) else None,
                'reasons': reasons,
                'matches': ev,
            })

        if remember:   # the image memory grows with every page analysed
            self.mem = pd.concat([mem, new[MEM_COLS + ['ext_key']]], ignore_index=True)
            self.mem_forensic = ov['forensic']
            self.mem_xmodal = ov['xmodal']
            if 'cnn' in ov:
                self.mem_cnn = ov['cnn']
            if 'text_model' in ov:
                self.mem_text = ov['text_model']
            if 'reuse_emb' in ov:
                self.mem_reuse = ov['reuse_emb']

        summ = {'checked': len(results),
                'fake': sum(r['decision'] == 'Fake' for r in results),
                'needs_verification': sum(r['decision'] == 'Needs_Verification' for r in results),
                'genuine': sum(r['decision'] == 'Genuine' for r in results),
                'memory_size': len(self.mem)}
        return {'reviews': results, 'summary': summ}

    def drivers(self, Xn, med, i, top=3):
        """Which features actually produced this score, in plain words.

        Measured by ablation: put one feature back to what an ordinary review has (the memory
        median), re-score, and see how far the risk falls. All 41 variants go through the model
        in a single batched call, so this costs one extra prediction per review.

        This exists because the extension flagged a real review at 78% and the only line under
        it was "no photo: checked on text and behaviour only", which explains nothing.
        """
        cols = self.meta['features']
        row = Xn.iloc[[i]][cols]
        base = float(self.fusion.predict_proba(row.values)[:, 1][0])
        variants = np.repeat(row.values, len(cols), axis=0)
        for j, c in enumerate(cols):
            variants[j, j] = med[c]
        q = self.fusion.predict_proba(variants)[:, 1]
        drop = base - q
        out = []
        for j in np.argsort(-drop)[:top]:
            if drop[j] < 0.02:
                break
            c = cols[j]
            out.append(f'{PLAIN_FEATURE.get(c, c)} (+{drop[j]:.2f} of the risk)')
        if not out:
            return ['no single strong signal: the score is the sum of many small ones']
        return ['main reason: ' + '; '.join(out)]

    def photo_matches(self, row, mem, reuse, limit=5):
        """Which photos already in memory this one matches, and who posted them.

        The extension used to say only "4 matching photos from other accounts", which is not
        checkable by the person reading it. This returns the evidence itself: the account, the
        date, the product and how close the match is, so a claim can be believed or dismissed.
        """
        if not len(mem):
            return []
        d = np.array([GI.phash_distance(row.phash, q) for q in mem.phash.tolist()])
        close = d <= F.PHASH_REUSE_THRESHOLD
        cos = np.full(len(mem), np.nan)
        v = reuse.get(row.review_id)
        if v is not None and self.mem_reuse is not None:
            ids = mem.review_id.tolist()
            have = [j for j, rid in enumerate(ids) if rid in self.mem_reuse]
            if have:
                M = np.stack([self.mem_reuse[ids[j]] for j in have]).astype(np.float32)
                M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
                w = np.asarray(v, dtype=np.float32)
                w /= (np.linalg.norm(w) + 1e-9)
                cos[have] = M @ w
                close = close | (np.nan_to_num(cos, nan=-1) >= F.RESNET_REUSE_THRESHOLD)
        idx = np.where(close)[0]
        idx = idx[np.argsort(d[idx])][:limit]
        out = []
        for j in idx:
            m = mem.iloc[j]
            out.append({
                'reviewer': str(m.reviewer_id),
                'date': str(m.timestamp)[:10],
                'product': str(m.product_id),
                'phash_distance': int(d[j]),
                'visual_similarity': None if np.isnan(cos[j]) else round(float(cos[j]), 3),
                'source': 'our dataset' if not str(m.review_id).startswith('EXT') else 'seen earlier by this extension',
            })
        return out

    def reasons(self, f, info, ev=(), row=None):
        out = []
        if not info['has_photo']:
            out.append('no photo: checked on text and behaviour only')
        else:
            if info['listing_match']:
                out.append("photo matches the seller's own listing photo")
            if ev:
                first = min(ev, key=lambda e: e['date'])
                mine = str(row.timestamp)[:10] if row is not None else ''
                who = first['reviewer']
                if mine and first['date'] and mine > first['date']:
                    out.append(f"same photo already posted by {who} on {first['date']} "
                               f"({first['source']}); this review is dated {mine}, so this one is the later copy")
                elif mine and first['date'] and mine < first['date']:
                    out.append(f"the same photo also appears under {who} on {first['date']} "
                               f"({first['source']}), which is AFTER this review, so this one is the earlier posting")
                else:
                    out.append(f"same photo also posted by {who} on {first['date']} ({first['source']}); "
                               f"dates are too close to say which came first")
            # img_reuse_* counts photos that are near-identical (pHash) OR look
            # very alike (CLIP), so other photos of the same kind of product
            # are included. Only the BURST (many accounts within hours) is a
            # fraud signal; similar photos spread over weeks are normal.
            burst = int(f.get('img_reuse_in_burst', 0))
            near_dup = f.get('img_min_phash_dist', 64) <= F.PHASH_REUSE_THRESHOLD
            if burst >= 2:
                who = ', '.join(dict.fromkeys(e['reviewer'] for e in ev))[:120]
                out.append(f'{burst} matching photos from other accounts posted within 6 hours'
                           + (f' ({who})' if who else ''))
            elif near_dup and not ev:
                out.append('a near-identical photo exists from another account '
                           '(spread over time, can be normal)')
            # How much the tamper claim is worth depends on how big the photo is. Measured by
            # shrinking test photos and re-scoring them (backend/diagnose_small_photos.py):
            # <=1024px ROC 0.947, 800px 0.898, 600px 0.835, 480px 0.734, 360px 0.629.
            edge = info.get('photo_edge')
            if info.get('tamper_unusable'):
                # nothing is claimed either way, and the score was neutralised in analyze()
                out.append(f'photo is only {edge}px: too small to check for editing, '
                           f'so that evidence was ignored')
            elif info['photo_too_small']:
                if f.get('img_cnn_manip', 0) >= config.WEAK_CNN_MIN_SCORE:
                    out.append(f"possible signs of editing (CNN {f['img_cnn_manip']:.2f}), "
                               f"but the photo is {edge}px so this is weaker evidence")
                else:
                    out.append(f'photo is {edge}px: edit check less reliable below 1000px')
            elif f.get('img_cnn_manip', 0) >= 0.5:
                out.append(f"photo shows signs of editing (CNN {f['img_cnn_manip']:.2f})")
        if info.get('text_unreliable'):
            # said plainly: this is a limit of the training data, not a finding about the review
            out.append(f"this review is {info.get('text_words', '?')} words; the text model was "
                       f"trained on reviews of at most {int(self.text_len_limit / 1.5)} words, "
                       f"so it is too long for the text check and that evidence was ignored")
        if f.get('txt_exact_dup_count', 1) > 1:
            out.append(f"identical text in {int(f['txt_exact_dup_count']) - 1} other review(s)")
        if f.get('txt_generic_ratio', 0) >= 0.25:
            out.append('generic, superlative-heavy wording')
        if f.get('txt_len', 99) <= 6:
            out.append('very short text')
        return out
