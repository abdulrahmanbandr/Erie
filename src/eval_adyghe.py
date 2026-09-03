"""Adyghe evaluation of the V4 age regressor and gender classifier.

Protocol (docs/EXPERIMENT_LOG.md, "Adyghe protocol"), all at speaker
level unless stated:
  1. zero-shot MAE / Spearman with bootstrap CI and permutation test
  2. calibrated MAE: leave-one-speaker-out linear fit on the other speakers
  3. within-Adyghe reference: ridge trained on Adyghe only, leave-one-out
  4. gender check: fraction of Adyghe women predicted female
  5. mean predicted age per true bin

Inputs: the Adyghe TSV(s) and audio directory, plus the V4 training
features and split CSV. Features for Adyghe are extracted with the same
encoder and layer unless --ady-features points at a precomputed array.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_MIDPOINTS, SEED, SR, get_device, load_splits  # noqa: E402
from regress_age import ALPHAS  # noqa: E402

GENDERS = ["male_masculine", "female_feminine"]


def load_adyghe(tsvs, audio_dir, clips_per_speaker, female_only):
    df = pd.concat([pd.read_csv(t, sep="\t", usecols=["client_id", "path", "age", "gender"],
                                low_memory=False) for t in tsvs], ignore_index=True)
    n_all = df.client_id.nunique()
    df = df[df.gender.isin(GENDERS) & df.age.isin(AGE_MIDPOINTS)].copy()
    if female_only:
        df = df[df.gender == "female_feminine"]
    n_labels = df.groupby("client_id")[["age", "gender"]].nunique()
    df = df[~df.client_id.isin(n_labels[(n_labels > 1).any(axis=1)].index)]
    found = {os.path.basename(p): p for p in glob.glob(f"{audio_dir}/**/*.mp3", recursive=True)}
    df["fullpath"] = df.path.map(found)
    df = df.dropna(subset=["fullpath"])
    df = (df.sample(frac=1, random_state=SEED).groupby("client_id").head(clips_per_speaker)
            .sort_values(["client_id", "path"]).reset_index(drop=True))
    df["age_mid"] = df.age.map(AGE_MIDPOINTS)
    df["is_female"] = df.gender == "female_feminine"
    print(f"Adyghe: {n_all} speakers in TSVs | {df.client_id.nunique()} evaluated | {len(df)} clips")
    print("speakers per bin:\n" + df.drop_duplicates("client_id").age.value_counts().to_string())
    return df


def extract_layer(model_name, layer, paths, max_sec):
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    device = get_device()
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            a, _ = librosa.load(p, sr=SR, mono=True, duration=max_sec)
            x = fe(a, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            h = enc(x, output_hidden_states=True).hidden_states[layer]
            out.append(h[0].mean(0).float().cpu().numpy())
            if i % 200 == 0:
                print(f"  {i} / {len(paths)}")
    return np.stack(out)


def speaker_level(df, pred):
    d = df.assign(pred=pred)
    s = d.groupby("client_id").agg(pred=("pred", "mean"), y=("age_mid", "first"), bin=("age", "first"))
    return s


def mae_ci(y, pred, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    err = np.abs(pred - y)
    boots = [err[rng.integers(0, len(err), len(err))].mean() for _ in range(n_boot)]
    return float(err.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def permutation_p(y, pred, n_perm=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    real = np.mean(np.abs(pred - y))
    perms = [np.mean(np.abs(pred - rng.permutation(y))) for _ in range(n_perm)]
    return float((np.sum(np.array(perms) <= real) + 1) / (n_perm + 1))


def loso_calibration(s):
    """Leave-one-speaker-out linear recalibration a*pred + b."""
    p, y = s.pred.values, s.y.values
    cal = np.empty_like(p)
    for i in range(len(p)):
        m = np.arange(len(p)) != i
        a, b = np.polyfit(p[m], y[m], 1)
        cal[i] = a * p[i] + b
    return cal


def loso_within(X_spk, y):
    """Ridge trained on Adyghe speakers only, leave-one-out (speaker-level features)."""
    out = np.empty_like(y)
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        reg = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(X_spk[m], y[m])
        out[i] = reg.predict(X_spk[i:i + 1])[0]
    return out


def run(args):
    train = load_splits(args.splits)
    Xtr = np.load(args.features)[args.layer]
    assert Xtr.shape[0] == len(train)
    train["age_mid"] = train.age_group.map(AGE_MIDPOINTS)
    train["is_female"] = train.gender.str.startswith("female")

    ady = load_adyghe(args.tsv.split(","), args.audio_dir, args.clips_per_speaker, not args.all_genders)
    if args.ady_features and os.path.exists(args.ady_features):
        Xa = np.load(args.ady_features)
        assert Xa.shape[0] == len(ady), f"{Xa.shape[0]} feature rows vs {len(ady)} clips"
    else:
        print(f"extracting {args.model} layer {args.layer} for {len(ady)} clips")
        Xa = extract_layer(args.model, args.layer, ady.fullpath.tolist(), args.max_sec)
        if args.ady_features:
            np.save(args.ady_features, Xa)

    # 1. zero-shot age
    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(Xtr, train.age_mid.values)
    s = speaker_level(ady, reg.predict(Xa))
    mae, lo, hi = mae_ci(s.y.values, s.pred.values)
    zero = {"speaker_mae": mae, "ci95": [lo, hi],
            "spearman": float(spearmanr(s.y, s.pred)[0]) if s.y.nunique() > 1 else None,
            "permutation_p": permutation_p(s.y.values, s.pred.values),
            "baseline_mae_train_mean": float(np.mean(np.abs(train.age_mid.mean() - s.y.values))),
            "baseline_mae_adyghe_mean": float(np.mean(np.abs(s.y.mean() - s.y.values))),
            "clip_mae": float(np.mean(np.abs(reg.predict(Xa) - ady.age_mid.values))),
            "mean_pred_by_true_bin": s.groupby("bin").pred.mean().round(1).to_dict(),
            "n_speakers": int(len(s)), "n_clips": int(len(ady))}

    # 2. calibrated
    cal = loso_calibration(s)
    cmae, clo, chi = mae_ci(s.y.values, cal)
    calib = {"speaker_mae": cmae, "ci95": [clo, chi],
             "mean_pred_by_true_bin": s.assign(cal=cal).groupby("bin").cal.mean().round(1).to_dict()}

    # 3. within-Adyghe reference
    X_spk = pd.DataFrame(Xa).groupby(ady.client_id.values).mean().loc[s.index].values
    within = loso_within(X_spk, s.y.values)
    wmae, wlo, whi = mae_ci(s.y.values, within)
    within_res = {"speaker_mae": wmae, "ci95": [wlo, whi],
                  "spearman": float(spearmanr(s.y, within)[0]) if s.y.nunique() > 1 else None}

    # 4. gender
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(Xtr, train.is_female.values)
    pf = pd.Series(clf.predict_proba(Xa)[:, 1]).groupby(ady.client_id.values).mean()
    truth = ady.groupby("client_id").is_female.first().loc[pf.index]
    gender = {"n_speakers": int(len(pf)),
              "female_predicted_female": float(np.mean(pf[truth] >= 0.5)) if truth.any() else None,
              "male_predicted_male": float(np.mean(pf[~truth] < 0.5)) if (~truth).any() else None,
              "mean_p_female_for_women": float(pf[truth].mean()) if truth.any() else None}

    results = {"model": args.model, "layer": args.layer, "train_speakers": int(train.client_id.nunique()),
               "adyghe_speakers_per_bin": s.bin.value_counts().to_dict(),
               "zero_shot": zero, "calibrated_loso": calib, "within_adyghe_loso": within_res,
               "gender": gender}
    print(json.dumps(results, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    s.assign(calibrated=cal, within=within).to_csv(args.out.replace(".json", "_speakers.csv"))
    print("saved:", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="Adyghe TSV path(s), comma-separated")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--features", default="layers_wavlm-base-plus.npy", help="V4 training features")
    parser.add_argument("--splits", default="splits_v4.csv")
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--ady-features", default="layers_ady_wavlm_l5.npy",
                        help="cache for the Adyghe features (reused if it exists)")
    parser.add_argument("--clips-per-speaker", type=int, default=10)
    parser.add_argument("--max-sec", type=float, default=6.0)
    parser.add_argument("--all-genders", action="store_true", help="include the male speaker(s)")
    parser.add_argument("--out", default="results/v4/adyghe_eval.json")
    run(parser.parse_args())
