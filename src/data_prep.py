"""Build balanced metadata for one or more Common Voice languages and
download only the audio that is needed, one shard at a time.

Per language:
  1. read train/dev/test TSVs remotely, keep speakers with age + gender
  2. cap speakers per (age bin, gender) cell so no cell or language
     dominates; keep the first N clips of each selected speaker
  3. write <lang>_age.csv; a combined all_age.csv covers every language
  4. download each audio shard, extract only the selected clips, delete
     the shard, stop once every clip is found

Raw age bins are kept (teens ... nineties); class mapping happens later
in common.py.
"""

import argparse
import os
import tarfile

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "fsicoli/common_voice_21_0"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/transcript"
SPLITS = ["train", "dev", "test"]
GENDERS = ["male_masculine", "female_feminine"]
AGE_BINS = ["teens", "twenties", "thirties", "fourties", "fifties",
            "sixties", "seventies", "eighties", "nineties"]


def build_metadata(lang: str, cap_per_cell: int, clips_per_speaker: int,
                   seed: int, out_dir: str = ".") -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f"{BASE_URL}/{lang}/{s}.tsv", sep="\t",
                                usecols=["client_id", "path", "age", "gender"],
                                low_memory=False).assign(split_cv=s)
                    for s in SPLITS], ignore_index=True)
    df = df[df.gender.isin(GENDERS) & df.age.isin(AGE_BINS)].copy()
    df["age_group"] = df.age
    df["lang"] = lang

    # A few accounts changed age or gender between recordings: drop them.
    n_labels = df.groupby("client_id")[["age_group", "gender"]].nunique()
    inconsistent = n_labels[(n_labels > 1).any(axis=1)].index
    df = df[~df.client_id.isin(inconsistent)]
    if len(inconsistent):
        print(f"{lang}: dropped {len(inconsistent)} speakers with inconsistent labels")

    # Labels belong to speakers: select speakers per cell, then clips.
    spk = df.drop_duplicates("client_id")[["client_id", "age_group", "gender"]]
    parts = [g.sample(n=min(len(g), cap_per_cell), random_state=seed)
             for _, g in spk.groupby(["age_group", "gender"])]
    keep = pd.concat(parts).client_id
    df = df[df.client_id.isin(keep)]
    df = (df.sample(frac=1, random_state=seed)
            .groupby("client_id").head(clips_per_speaker)
            .sort_values(["client_id", "path"]).reset_index(drop=True))

    out = os.path.join(out_dir, f"{lang}_age.csv")
    df.to_csv(out, index=False)
    table = pd.crosstab(df.drop_duplicates("client_id").age_group,
                        df.drop_duplicates("client_id").gender).reindex(AGE_BINS).fillna(0).astype(int)
    print(f"\n=== {lang}: {df.client_id.nunique()} speakers | {len(df)} clips -> {out}")
    print("speakers per cell:")
    print(table.to_string())
    return df


def download_audio(lang: str, wanted: set, out_dir: str, splits=SPLITS) -> None:
    """Download shards one by one; keep only wanted clips; delete the shard."""
    api = HfApi()
    remaining = set(wanted)
    tmp = os.path.join(out_dir, "_shards")
    for split in splits:
        files = sorted(f.path for f in api.list_repo_tree(
            REPO, path_in_repo=f"audio/{lang}/{split}", repo_type="dataset")
            if f.path.endswith(".tar"))
        dest = os.path.join(out_dir, "audio", lang, split)
        os.makedirs(dest, exist_ok=True)
        for i, fp in enumerate(files):
            if not remaining:
                break
            local = hf_hub_download(REPO, filename=fp, repo_type="dataset", local_dir=tmp)
            found = 0
            with tarfile.open(local) as tar:
                for m in tar:
                    name = os.path.basename(m.name)
                    if m.isfile() and name in remaining:
                        m.name = name
                        tar.extract(m, path=dest)
                        remaining.discard(name)
                        found += 1
            os.remove(local)
            print(f"{lang}/{split} shard {i + 1}/{len(files)}: kept {found} | "
                  f"{len(remaining)} clips still missing")
    print(f"{lang}: done, {len(wanted) - len(remaining)}/{len(wanted)} clips on disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="ca,de,ru")
    parser.add_argument("--cap-per-cell", type=int, default=100,
                        help="max speakers per (language, age bin, gender)")
    parser.add_argument("--clips-per-speaker", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--splits", default=",".join(SPLITS),
                        help="CV splits to download (for testing)")
    parser.add_argument("--skip-audio", action="store_true")
    args = parser.parse_args()

    frames = []
    for lang in args.langs.split(","):
        df = build_metadata(lang, args.cap_per_cell, args.clips_per_speaker, args.seed)
        frames.append(df)
        if not args.skip_audio:
            download_audio(lang, set(df.path), args.out_dir, args.splits.split(","))

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv("all_age.csv", index=False)
    print(f"\nall_age.csv: {all_df.client_id.nunique()} speakers | {len(all_df)} clips | "
          f"languages {sorted(all_df.lang.unique())}")
