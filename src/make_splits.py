"""Step 2: split speakers into train (65%), val (15%) and test (20%).

All clips of a speaker land in the same split, so the model is always
scored on voices it has never heard. Clips whose audio is missing are
dropped. Writes data/splits.csv.
"""

import argparse
import glob
import os
import sys

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_MIDPOINTS, DATA_DIR, METADATA, SEED, SPLITS  # noqa: E402


def make_splits(metadata: str, audio_dir: str, out: str,
                test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    df = pd.read_csv(metadata)
    df = df[df.age_group.isin(AGE_MIDPOINTS)].copy()

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
    print(df.groupby("split").agg(clips=("path", "size"), speakers=("client_id", "nunique")))
    print("\nspeakers per language and split:")
    print(pd.crosstab(df.drop_duplicates("client_id").lang,
                      df.drop_duplicates("client_id").split).to_string())
    print("saved:", out)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metadata", default=METADATA)
    parser.add_argument("--audio-dir", default=os.path.join(DATA_DIR, "audio"))
    parser.add_argument("--out", default=SPLITS)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    make_splits(args.metadata, args.audio_dir, args.out,
                args.test_size, args.val_size, args.seed)
