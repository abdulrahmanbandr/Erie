"""Train and evaluate the ECAPA + LogisticRegression baseline."""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"))


def run(emb_path: str, target: str, out_path: str) -> None:
    data = np.load(emb_path, allow_pickle=True)
    X, groups = data["X"], data["groups"]
    y = data["y"] if target == "age" else data["gender"]

    print(f"{X.shape[0]} clips | {len(np.unique(groups))} speakers")

    clf = build_classifier()

    # GroupKFold keeps each speaker entirely within one fold —
    # a single split can swing results by ±0.02.
    scores = cross_val_score(clf, X, y, groups=groups,
                             cv=GroupKFold(n_splits=5),
                             scoring="f1_macro")

    print("folds:", scores.round(3))
    print(f"macro-F1: {scores.mean():.3f} ± {scores.std():.3f}")

    results = {
        "task": f"{target}_ru",
        "features": "ECAPA frozen embeddings (192d)",
        "model": "StandardScaler + LogisticRegression(balanced)",
        "cv": "GroupKFold(5) on client_id",
        "macro_f1_mean": round(float(scores.mean()), 4),
        "macro_f1_std": round(float(scores.std()), 4),
        "fold_scores": [round(float(s), 4) for s in scores],
        "n_clips": int(X.shape[0]),
        "n_speakers": int(len(np.unique(groups))),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", default="ru_age_emb.npz")
    parser.add_argument("--target", choices=["age", "gender"], default="age")
    parser.add_argument("--out", default="results/baseline_results.json")
    args = parser.parse_args()

    run(args.emb, args.target, args.out)