"""Step 5: fit the final models on every speaker and save them.

Age: ridge regression on the age-bin midpoint. Gender: logistic
regression. Both read the same features. Writes models/eiry.joblib,
the file predict.py loads.
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
from common import AGE_MIDPOINTS, FEATURES, MODEL_FILE, SPLITS, load_dataset  # noqa: E402

ALPHAS = np.logspace(-1, 4, 12)


def age_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def gender_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))


def train(splits: str, features: str, out: str) -> None:
    df, X, info = load_dataset(splits, features)
    age = age_model().fit(X, df.age.values)
    gender = gender_model().fit(X, df.is_female.values)

    models = {"age": age, "gender": gender, **info,
              "n_speakers": int(df.client_id.nunique()), "n_clips": int(len(df)),
              "languages": sorted(df.lang.unique()),
              "age_bins": [b for b in AGE_MIDPOINTS if b in set(df.age_group)]}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(models, out, compress=3)
    print(f"fitted on {models['n_speakers']} speakers, {models['n_clips']} clips | "
          f"saved {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", default=SPLITS)
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--out", default=MODEL_FILE)
    args = parser.parse_args()

    train(args.splits, args.features, args.out)
