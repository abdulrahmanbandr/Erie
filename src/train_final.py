"""Fit the final Eiry models on the V4 features and save them to models/.

Run once after extract_layers.py. The output file is small (a ridge and
a logistic regression on 768-d features) and is committed to the repo so
that predict.py works from a clone without any training data.
"""

import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_MIDPOINTS, load_splits  # noqa: E402

MODEL_NAME = "microsoft/wavlm-base-plus"
LAYER = 5
ALPHAS = np.logspace(-1, 4, 12)


def train(features: str, splits: str, out: str) -> None:
    df = load_splits(splits)
    X = np.load(features)[LAYER]
    assert X.shape[0] == len(df), f"{X.shape[0]} feature rows vs {len(df)} metadata rows"
    keep = df.age_group.isin(AGE_MIDPOINTS).values
    y_age = df.age_group.map(AGE_MIDPOINTS).values[keep]
    y_fem = df.gender.str.startswith("female").values

    age = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X[keep], y_age)
    gender = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=3000, class_weight="balanced")).fit(X, y_fem)

    models = {"age": age, "gender": gender, "layer": LAYER, "encoder": MODEL_NAME,
              "n_speakers": int(df.client_id.nunique()), "n_clips": int(len(df)),
              "languages": sorted(df.lang.unique()) if "lang" in df else None,
              "age_bins": sorted(set(df.age_group[keep]))}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    joblib.dump(models, out, compress=3)
    print(f"fitted on {models['n_speakers']} speakers, {models['n_clips']} clips | "
          f"ridge alpha {age[-1].alpha_:.0f} | saved {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="layers_wavlm-base-plus.npy")
    parser.add_argument("--splits", default="splits_v4.csv")
    parser.add_argument("--out", default="models/eiry_v4.joblib")
    args = parser.parse_args()
    train(args.features, args.splits, args.out)
