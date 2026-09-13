"""Step 4: measure the model on speakers it was not trained on.

1. Test split: fit on the train + val speakers, score the test speakers once.
2. Unseen language: for each language, fit on the other languages and
   score every speaker of the held-out one.

Age is scored as mean absolute error (MAE) in years and rank correlation,
per clip and per speaker (the mean over a speaker's clips), next to the
error of always predicting the training mean. Gender is scored as
macro-F1. Writes results/metrics.json.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_MIDPOINTS, FEATURES, METRICS, SPLITS, load_dataset  # noqa: E402
from train import age_model, gender_model  # noqa: E402


def age_scores(y, pred):
    err = np.abs(pred - y)
    return {"mae": round(float(err.mean()), 2),
            "median_ae": round(float(np.median(err)), 2),
            "pearson_r": round(float(pearsonr(y, pred)[0]), 3),
            "spearman_rho": round(float(spearmanr(y, pred)[0]), 3)}


def by_bin(series, digits):
    """A per-bin series as a dict in age order."""
    return {b: round(float(series[b]), digits) for b in AGE_MIDPOINTS if b in series}


def score(X, df, train, test):
    """Fit both models on the `train` rows and score the `test` rows."""
    age = age_model().fit(X[train], df.age.values[train])
    gender = gender_model().fit(X[train], df.is_female.values[train])
    d = df[test].assign(age_pred=age.predict(X[test]),
                        p_female=gender.predict_proba(X[test])[:, 1])
    spk = d.groupby("client_id").agg(age=("age", "first"), age_pred=("age_pred", "mean"),
                                     is_female=("is_female", "first"),
                                     p_female=("p_female", "mean"))
    err = (d.age_pred - d.age).abs()
    sex = d.is_female.map({True: "female", False: "male"})
    return {
        "n_train_speakers": int(df[train].client_id.nunique()),
        "n_test_speakers": int(len(spk)),
        "n_test_clips": int(len(d)),
        "age": {
            "clip": age_scores(d.age.values, d.age_pred.values),
            "speaker": age_scores(spk.age.values, spk.age_pred.values),
            "baseline_mae_predict_mean": round(float(
                np.mean(np.abs(df.age.values[train].mean() - d.age.values))), 2),
            "clip_mae_by_gender": err.groupby(sex).mean().round(2).to_dict(),
            "clip_mae_by_lang": err.groupby(d.lang).mean().round(2).to_dict(),
            "clip_mae_by_true_bin": by_bin(err.groupby(d.age_group).mean(), 2),
            "clip_mean_pred_by_true_bin": by_bin(d.groupby("age_group").age_pred.mean(), 1),
        },
        "gender": {
            "clip_macro_f1": round(float(f1_score(
                d.is_female, d.p_female >= 0.5, average="macro")), 3),
            "speaker_macro_f1": round(float(f1_score(
                spk.is_female, spk.p_female >= 0.5, average="macro")), 3),
        },
    }


def report(name, res):
    a = res["age"]
    print(f"{name:22s} {res['n_test_speakers']:5d} speakers | "
          f"age MAE {a['speaker']['mae']:5.2f} y (predict-the-mean {a['baseline_mae_predict_mean']:5.2f}) | "
          f"Spearman {a['speaker']['spearman_rho']:.2f} | "
          f"gender F1 {res['gender']['speaker_macro_f1']:.3f}")


def evaluate(splits: str, features: str, out: str) -> None:
    df, X, info = load_dataset(splits, features)
    print(f"{len(df)} clips | {df.client_id.nunique()} speakers | "
          f"{info['encoder']} layer {info['layer']} | scores per speaker\n")

    test = (df.split == "test").values
    results = {**info, "test_split": score(X, df, ~test, test), "unseen_language": {}}
    report("test split", results["test_split"])

    if df.lang.nunique() > 1:
        for lang in sorted(df.lang.unique()):
            held = (df.lang == lang).values
            results["unseen_language"][lang] = score(X, df, ~held, held)
            report(f"unseen language: {lang}", results["unseen_language"][lang])

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", default=SPLITS)
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--out", default=METRICS)
    args = parser.parse_args()

    evaluate(args.splits, args.features, args.out)
