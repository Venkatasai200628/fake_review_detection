"""
run_forensics.py -- compute the pixel-level manipulation evidence (ELA + noise
residual, see forensics.py) for every review image and write
../out/forensic_features.csv.

Same result as the one-line command in the implementation guide (section 13.3),
just run in parallel so ~3300 images take a few minutes instead of many.
Nothing here reads a label.
"""
import os
import multiprocessing as mp
import pandas as pd
import forensics

OUT = '../out'


def _one(path):
    return forensics.analyse(os.path.join(OUT, path))


def main():
    df = pd.read_csv(f'{OUT}/reviews_full.csv')
    with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
        feats = p.map(_one, df.image_file.tolist(), chunksize=16)
    f = pd.DataFrame(feats)
    f.insert(0, 'review_id', df.review_id.values)
    f.to_csv(f'{OUT}/forensic_features.csv', index=False)
    print(f"forensic_features.csv: {len(f)} rows, {f.shape[1] - 1} features")


if __name__ == '__main__':
    main()
