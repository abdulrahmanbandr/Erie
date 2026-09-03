"""Speaker-level train / val / test split of the age metadata.

Every speaker lands in exactly one split. The test split is never used
for model selection; early stopping and layer selection use val.

Rows are kept for every raw age bin that any scheme uses (teens through
50+), so one split CSV serves both the 4-class and 3-class tasks. The
raw bins are stored; class mapping happens at load time (common.py).
Note: adding the 50+ speakers changes the split relative to V2's
4-class-only CSV. Write it to a new file name.
"""

import argparse
import glob
import os
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_SCHEMES, SEED  # noqa: E402

ALL_BINS = sorted({b for s in AGE_SCHEMES.values() for b in s["map"]})


def make_splits(csv: str, audio_dir: str, out: str,
                test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = df[df.age_group.isin(ALL_BINS)].copy()

    found = {os.path.basename(p): p
             for p in glob.glob(f"{audio_dir}/**/*.mp3", recursive=True)}
    df["fullpath"] = df.path.map(found)
    df = df.dropna(subset=["fullpath"]).reset_index(drop=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(gss.split(df, groups=df.client_id))
    df["split"] = "dev"
    df.loc[test_idx, "split"] = "test"

    dev = df[df.split == "dev"]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size),
                             random_state=seed)
    tr_idx, va_idx = next(gss2.split(dev, groups=dev.client_id))
    df.loc[dev.index[tr_idx], "split"] = "train"
    df.loc[dev.index[va_idx], "split"] = "val"

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = set(df[df.split == a].client_id) & set(df[df.split == b].client_id)
        assert not overlap, f"speaker overlap between {a} and {b}"

    df.to_csv(out, index=False)
    print(df.groupby("split").agg(clips=("path", "size"),
                                  speakers=("client_id", "nunique")))
    print("\nclips per split and raw bin:")
    print(pd.crosstab(df.split, df.age_group))
    spk = df.drop_duplicates("client_id")
    print("\nspeakers per raw bin:")
    print(spk.age_group.value_counts().to_string())
    if "lang" in df:
        print("\nspeakers per language and split:")
        print(pd.crosstab(spk.lang, spk.split).to_string())
    print("saved:", out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="all_age.csv")
    parser.add_argument("--audio-dir", default="data/audio")
    parser.add_argument("--out", default="ru_splits_all.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    make_splits(args.csv, args.audio_dir, args.out,
                args.test_size, args.val_size, args.seed)
