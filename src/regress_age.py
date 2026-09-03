"""Age as a number: ridge regression on per-layer features, error in years.

Common Voice gives age bins, so the target is the bin midpoint
(teens 16, twenties 25, thirties 35, fourties 45; 50+ excluded by
default). Per-layer StratifiedGroupKFold on train+val speakers, then the
best layer is scored once on the test split: MAE in years at clip and
speaker level, by gender and by true bin, and the predictions re-binned
into the 4 classes so the result is comparable with the classifiers.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_MIDPOINTS, OLDER_BINS, SEED, load_splits  # noqa: E402

MIDPOINTS = {b: AGE_MIDPOINTS[b] for b in ["teens", "twenties", "thirties", "fourties"]}
BIN_EDGES = [20, 30, 40, 50]          # predicted age -> teens/twenties/thirties/fourties/50+
BIN_NAMES = ["teens", "twenties", "thirties", "fourties", "fifties", "sixties",
             "seventies", "eighties", "nineties", "50+"]
ALPHAS = np.logspace(-1, 4, 12)


def make_reg():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def to_bin(age):
    return np.digitize(age, BIN_EDGES)


def scores(y, pred):
    return {"mae": float(np.mean(np.abs(pred - y))),
            "median_ae": float(np.median(np.abs(pred - y))),
            "pearson_r": float(pearsonr(y, pred)[0]),
            "spearman_rho": float(spearmanr(y, pred)[0])}


def sample_weights(bins):
    counts = pd.Series(bins).value_counts()
    return pd.Series(bins).map(1.0 / counts).values * len(bins) / len(counts)


def cv_layer(X, y, bins, groups, balanced):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    maes, base = [], []
    for tr, va in cv.split(X, bins, groups):
        reg = make_reg()
        fit_kw = {"ridgecv__sample_weight": sample_weights(bins[tr])} if balanced else {}
        reg.fit(X[tr], y[tr], **fit_kw)
        maes.append(np.mean(np.abs(reg.predict(X[va]) - y[va])))
        base.append(np.mean(np.abs(np.mean(y[tr]) - y[va])))
    return float(np.mean(maes)), float(np.std(maes)), float(np.mean(base))


def eval_on_test(X, df, balanced):
    tr = (df.split != "test").values
    te = ~tr
    reg = make_reg()
    fit_kw = {"ridgecv__sample_weight": sample_weights(df.bin.values[tr])} if balanced else {}
    reg.fit(X[tr], df.age_mid.values[tr], **fit_kw)
    pred = reg.predict(X[te])
    y = df.age_mid.values[te]
    d = df[te].assign(pred=pred)

    spk = d.groupby("client_id").agg(pred=("pred", "mean"), y=("age_mid", "first"))
    fine_mask = d.bin.isin(MIDPOINTS).values
    res = {
        "clip": scores(y, pred),
        "speaker": scores(spk.y.values, spk.pred.values),
        "baseline_mae_predict_mean": float(np.mean(np.abs(np.mean(df.age_mid.values[tr]) - y))),
        "mae_by_gender": {g: round(float(np.mean(np.abs(sub.pred - sub.age_mid))), 2)
                          for g, sub in d.groupby(d.is_female.map({True: "female", False: "male"}))},
        "mae_by_true_bin": {b: round(float(np.mean(np.abs(sub.pred - sub.age_mid))), 2)
                            for b, sub in d.groupby("bin")},
        "mae_by_lang": ({l: round(float(np.mean(np.abs(sub.pred - sub.age_mid))), 2)
                         for l, sub in d.groupby("lang")} if "lang" in d else None),
        "mean_pred_by_true_bin": {b: round(float(sub.pred.mean()), 1) for b, sub in d.groupby("bin")},
        "rebinned_4class_macro_f1": float(f1_score(
            to_bin(y[fine_mask]), np.clip(to_bin(pred[fine_mask]), 0, 3), average="macro")),
        "rebinned_within_one_bin_acc": float(np.mean(
            np.abs(to_bin(y[fine_mask]) - np.clip(to_bin(pred[fine_mask]), 0, 3)) <= 1)),
        "n_test_clips": int(te.sum()),
        "n_test_speakers": int(len(spk)),
        "ridge_alpha": float(reg[-1].alpha_),
    }
    print("\n== TEST (best layer) ==")
    print(json.dumps(res, indent=2))
    return res, d


def eval_holdout(X, df, lang, balanced):
    """Train on every other language (all splits), test on all of `lang`."""
    tr = (df.lang != lang).values
    te = ~tr
    reg = make_reg()
    fit_kw = {"ridgecv__sample_weight": sample_weights(df.bin.values[tr])} if balanced else {}
    reg.fit(X[tr], df.age_mid.values[tr], **fit_kw)
    pred = reg.predict(X[te])
    d = df[te].assign(pred=pred)
    spk = d.groupby("client_id").agg(pred=("pred", "mean"), y=("age_mid", "first"))
    res = {"held_out": lang, "n_train_speakers": int(df[tr].client_id.nunique()),
           "n_test_speakers": int(len(spk)), "n_test_clips": int(te.sum()),
           "clip": scores(d.age_mid.values, pred),
           "speaker": scores(spk.y.values, spk.pred.values),
           "baseline_mae_predict_mean": float(np.mean(np.abs(np.mean(df.age_mid.values[tr]) - d.age_mid.values))),
           "mae_by_gender": {g: round(float(np.mean(np.abs(sub.pred - sub.age_mid))), 2)
                             for g, sub in d.groupby(d.is_female.map({True: "female", False: "male"}))},
           "mean_pred_by_true_bin": {b: round(float(sub.pred.mean()), 1) for b, sub in d.groupby("bin")}}
    print(f"\n== HELD-OUT LANGUAGE {lang}: trained on {res['n_train_speakers']} speakers, "
          f"tested on {res['n_test_speakers']} ==")
    print(json.dumps(res, indent=2))
    return res


def run(features, splits, out, fig, exclude_older, balanced, holdout=None):
    X = np.load(features)
    full = load_splits(splits)
    assert X.shape[1] == len(full), f"{X.shape[1]} feature rows vs {len(full)} metadata rows"

    mid = dict(MIDPOINTS)
    if not exclude_older:
        mid.update({b: AGE_MIDPOINTS[b] for b in OLDER_BINS})
    keep = full.age_group.isin(mid).values
    df = full[keep].reset_index(drop=True)
    df["bin"] = df.age_group
    df["age_mid"] = df.age_group.map(mid)
    df["is_female"] = df.gender.str.startswith("female")
    X = X[:, keep]
    print(f"target: bin midpoints {mid} | balanced={balanced}")
    print(f"{len(df)} clips | {df.client_id.nunique()} speakers")
    print(df.drop_duplicates("client_id").bin.value_counts().to_string(), "\n")

    if holdout:
        assert "lang" in df, "split CSV has no lang column"
        layer = holdout["layer"]
        langs = sorted(df.lang.unique()) if holdout["lang"] == "all" else [holdout["lang"]]
        results = {"features": os.path.basename(features), "layer": layer, "balanced": balanced,
                   "mode": "leave-one-language-out",
                   "held_out": [eval_holdout(X[layer], df, l, balanced) for l in langs]}
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print("saved:", out)
        return

    dev = (df.split != "test").values
    y, bins, g = df.age_mid.values[dev], df.bin.values[dev], df.client_id.values[dev]
    per_layer = []
    for l in range(X.shape[0]):
        m, s, b = cv_layer(X[l][dev], y, bins, g, balanced)
        per_layer.append([m, s])
        print(f"layer {l:2d}: MAE {m:5.2f} ± {s:4.2f} years   (predict-the-mean baseline {b:5.2f})")
    best = int(np.argmin([p[0] for p in per_layer]))
    print("\nbest layer:", best, per_layer[best])

    if "lang" in df:
        print("speakers per language:", df.drop_duplicates("client_id").lang.value_counts().to_dict())
    test, d = eval_on_test(X[best], df, balanced)
    results = {"features": os.path.basename(features), "target": mid, "balanced": balanced,
               "n_clips": int(len(df)), "n_speakers": int(df.client_id.nunique()),
               "cv": "StratifiedGroupKFold(5) on train+val speakers, stratified by bin",
               "mae_by_layer": per_layer, "best_layer": best, "best_cv_mae": per_layer[best][0],
               "test_split_at_best_layer": test}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved:", out)

    if fig:
        import matplotlib.pyplot as plt
        fig_, ax = plt.subplots(1, 2, figsize=(11, 4))
        p = np.array(per_layer)
        ax[0].errorbar(range(len(p)), p[:, 0], p[:, 1], marker="o")
        ax[0].set_xlabel("layer"); ax[0].set_ylabel("CV MAE (years)"); ax[0].set_title(results["features"])
        order = [b for b in BIN_NAMES if b in set(d.bin)]
        ax[1].boxplot([d.pred[d.bin == b] for b in order], tick_labels=order)
        ax[1].set_ylabel("predicted age"); ax[1].set_title(f"test split, layer {best}")
        plt.tight_layout(); plt.savefig(fig, dpi=120); print("saved:", fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="layers_wavlm-base-plus.npy")
    parser.add_argument("--splits", default="ru_splits_all.csv")
    parser.add_argument("--out", default=None, help="default: results/v3/regress_<features>.json")
    parser.add_argument("--fig", default=None)
    parser.add_argument("--exclude-older", action="store_true",
                        help="ages 13-49 only (reproduces Exp 7); default keeps every bin")
    parser.add_argument("--balanced", action="store_true",
                        help="weight samples by inverse bin frequency")
    parser.add_argument("--holdout-lang", default=None,
                        help="leave-one-language-out: a language code, or 'all' for each in turn")
    parser.add_argument("--layer", type=int, default=5, help="layer used with --holdout-lang")
    args = parser.parse_args()

    short = os.path.basename(args.features).replace("layers_", "").replace(".npy", "")
    suffix = ("_to49" if args.exclude_older else "") + ("_balanced" if args.balanced else "")
    if args.holdout_lang:
        suffix += f"_holdout-{args.holdout_lang}"
    out = args.out or f"results/v4/regress_{short}{suffix}.json"
    holdout = {"lang": args.holdout_lang, "layer": args.layer} if args.holdout_lang else None
    run(args.features, args.splits, out, args.fig, args.exclude_older, args.balanced, holdout)
