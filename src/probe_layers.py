"""Linear probe of every layer for age and gender, then a test-split
evaluation of the best age layer.

Probing uses StratifiedGroupKFold on train+val speakers only. The test
split is touched once, at the end, with clip-level and speaker-level
scores plus two ordinal metrics. --scheme selects the age classes.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import (StratifiedGroupKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import DEFAULT_SCHEME, SEED, classes_for, load_splits  # noqa: E402


def make_clf():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced"))


def majority_f1(y):
    return float(f1_score(y, np.full_like(y, np.bincount(y).argmax()), average="macro"))


def probe(X, y, groups, name):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = []
    for l in range(X.shape[0]):
        s = cross_val_score(make_clf(), X[l], y, groups=groups, cv=cv,
                            scoring="f1_macro")
        res.append([float(s.mean()), float(s.std())])
        print(f"{name:6s} layer {l:2d}: {s.mean():.3f} ± {s.std():.3f}")
    return res


def stratified_report(X, y, groups, is_female, classes):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred = cross_val_predict(make_clf(), X, y, groups=groups, cv=cv)
    out = {}
    for name, m in [("male", ~is_female), ("female", is_female)]:
        f1 = float(f1_score(y[m], pred[m], average="macro"))
        out[name] = {"n_clips": int(m.sum()), "macro_f1": round(f1, 4)}
        print(f"\n== {name}: {m.sum()} clips | macro-F1 = {f1:.3f}")
        print(classification_report(y[m], pred[m], labels=range(len(classes)),
                                    target_names=classes, digits=3, zero_division=0))
    return out


def eval_on_test(X, df, classes):
    tr = (df.split != "test").values
    te = ~tr
    y_te = df.label.values[te]
    clf = make_clf().fit(X[tr], df.label.values[tr])
    P = clf.predict_log_proba(X[te])
    pred = P.argmax(1)

    spk = pd.DataFrame(P).groupby(df.client_id.values[te]).mean()
    y_spk = df[te].groupby("client_id").label.first().loc[spk.index].values
    pred_spk = spk.values.argmax(1)

    per_class = f1_score(y_te, pred, average=None, labels=range(len(classes)))
    res = {
        "clip_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "speaker_macro_f1": round(float(f1_score(y_spk, pred_spk, average="macro")), 4),
        "majority_macro_f1": round(majority_f1(y_te), 4),
        "per_class_f1": {c: round(float(v), 4) for c, v in zip(classes, per_class)},
        "within_one_bin_acc": round(float(np.mean(np.abs(pred - y_te) <= 1)), 4),
        "bin_mae": round(float(np.mean(np.abs(pred - y_te))), 4),
        "n_test_clips": int(te.sum()),
        "n_test_speakers": int(len(spk)),
        "confusion_matrix": confusion_matrix(y_te, pred, labels=range(len(classes))).tolist(),
    }
    print("\n== TEST (best layer) ==")
    print(classification_report(y_te, pred, labels=range(len(classes)),
                                target_names=classes, digits=3, zero_division=0))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("confusion_matrix", "per_class_f1")}, indent=2))
    print(pd.DataFrame(res["confusion_matrix"], index=classes, columns=classes))
    return res


def run(features: str, splits: str, scheme: str, out: str, fig: str | None) -> None:
    X = np.load(features)
    full = load_splits(splits, scheme)
    assert X.shape[1] == len(full), f"{X.shape[1]} feature rows vs {len(full)} metadata rows"
    classes = classes_for(scheme)

    keep = full.keep.values
    df = full[keep].reset_index(drop=True)
    df["label"] = df.label.astype(int)          # NaN rows dropped -> back to int
    X = X[:, keep]
    print(f"scheme {scheme}: classes {classes} | {len(df)} clips | "
          f"{df.client_id.nunique()} speakers")
    print(df.drop_duplicates("client_id").age_class.value_counts().to_string())

    dev = (df.split != "test").values
    y_age, y_gen, g = df.label.values[dev], df.is_female.values[dev], df.client_id.values[dev]
    print(f"\nmajority-class macro-F1 (dev): {majority_f1(y_age):.3f}\n")

    age = probe(X[:, dev], y_age, g, "age")
    gen = probe(X[:, dev], y_gen, g, "gender")
    best = int(np.argmax([a[0] for a in age]))
    print("\nbest age layer:", best, age[best])

    strat = stratified_report(X[best][dev], y_age, g, y_gen, classes)
    test = eval_on_test(X[best], df, classes)

    results = {
        "features": os.path.basename(features),
        "scheme": scheme,
        "classes": classes,
        "n_clips": int(len(df)),
        "n_speakers": int(df.client_id.nunique()),
        "speakers_per_class": df.drop_duplicates("client_id").age_class.value_counts().to_dict(),
        "cv": "StratifiedGroupKFold(5) on train+val speakers",
        "majority_macro_f1_dev": round(majority_f1(y_age), 4),
        "age_f1_by_layer": age,
        "gender_f1_by_layer": gen,
        "best_age_layer": best,
        "best_age_f1": age[best][0],
        "by_gender_at_best_layer": strat,
        "test_split_at_best_layer": test,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved:", out)

    if fig:
        import matplotlib.pyplot as plt
        a, gsd = np.array(age), np.array(gen)
        plt.errorbar(range(len(a)), a[:, 0], a[:, 1], marker="o", label="age")
        plt.errorbar(range(len(gsd)), gsd[:, 0], gsd[:, 1], marker="s", label="gender")
        plt.axhline(majority_f1(y_age), ls=":", c="gray", label="majority class")
        plt.xlabel("layer"); plt.ylabel("macro-F1")
        plt.title(f"{results['features']} | {scheme}")
        plt.legend(); plt.savefig(fig, dpi=120, bbox_inches="tight")
        print("saved:", fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="layers_wavlm-base-plus.npy")
    parser.add_argument("--splits", default="ru_splits_all.csv")
    parser.add_argument("--scheme", default=DEFAULT_SCHEME, choices=["fine4", "coarse3", "coarse3_gap"])
    parser.add_argument("--out", default=None,
                        help="default: results/v3/probe_<features>_<scheme>.json")
    parser.add_argument("--fig", default=None)
    args = parser.parse_args()

    short = os.path.basename(args.features).replace("layers_", "").replace(".npy", "")
    out = args.out or f"results/v3/probe_{short}_{args.scheme}.json"
    run(args.features, args.splits, args.scheme, out, args.fig)
