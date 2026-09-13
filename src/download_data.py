"""Step 1: choose balanced speakers from Common Voice and download their clips.

For each language:
  1. read the train/dev/test label tables from the Hugging Face mirror
  2. keep speakers with an age bin and a gender; drop speakers whose labels change
  3. pick at most --cap-per-cell speakers per (age bin, gender) and
     --clips-per-speaker clips per speaker
  4. download the audio shard by shard, keep only the chosen clips, delete the shard

Writes data/metadata.csv and data/audio/<lang>/<split>/*.mp3.
"""

import argparse
import os
import sys
import tarfile

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))
from common import AGE_MIDPOINTS, DATA_DIR, METADATA, SEED  # noqa: E402

REPO = "fsicoli/common_voice_21_0"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/transcript"
CV_SPLITS = ["train", "dev", "test"]
GENDERS = ["male_masculine", "female_feminine"]
AGE_BINS = list(AGE_MIDPOINTS)


def select_speakers(lang: str, cap_per_cell: int, clips_per_speaker: int,
                    seed: int) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f"{BASE_URL}/{lang}/{s}.tsv", sep="\t",
                                usecols=["client_id", "path", "age", "gender"],
                                low_memory=False).assign(split_cv=s)
                    for s in CV_SPLITS], ignore_index=True)
    df = df[df.gender.isin(GENDERS) & df.age.isin(AGE_BINS)].copy()
    df["age_group"] = df.age
    df["lang"] = lang

    # A few accounts changed age or gender between recordings: drop them.
    n_labels = df.groupby("client_id")[["age_group", "gender"]].nunique()
    inconsistent = n_labels[(n_labels > 1).any(axis=1)].index
    df = df[~df.client_id.isin(inconsistent)]

    # Labels belong to speakers: select speakers per cell, then clips.
    spk = df.drop_duplicates("client_id")[["client_id", "age_group", "gender"]]
    parts = [g.sample(n=min(len(g), cap_per_cell), random_state=seed)
             for _, g in spk.groupby(["age_group", "gender"])]
    keep = pd.concat(parts).client_id
    df = df[df.client_id.isin(keep)]
    df = (df.sample(frac=1, random_state=seed)
            .groupby("client_id").head(clips_per_speaker)
            .sort_values(["client_id", "path"]).reset_index(drop=True))

    speakers = df.drop_duplicates("client_id")
    table = (pd.crosstab(speakers.age_group, speakers.gender)
               .reindex(AGE_BINS).fillna(0).astype(int))
    print(f"\n{lang}: {len(speakers)} speakers | {len(df)} clips | "
          f"{len(inconsistent)} dropped for inconsistent labels")
    print(table.to_string())
    return df


def download_audio(lang: str, wanted: set, out_dir: str, cv_splits=CV_SPLITS) -> None:
    """Download shards one by one; keep only wanted clips; delete the shard."""
    api = HfApi()
    remaining = set(wanted)
    tmp = os.path.join(out_dir, "_shards")
    for split in cv_splits:
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
    print(f"{lang}: {len(wanted) - len(remaining)}/{len(wanted)} clips on disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--langs", default="ca,de,ru", help="Common Voice language codes")
    parser.add_argument("--cap-per-cell", type=int, default=100,
                        help="max speakers per (language, age bin, gender)")
    parser.add_argument("--clips-per-speaker", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-audio", action="store_true", help="write metadata only")
    args = parser.parse_args()

    langs = args.langs.split(",")
    frames = [select_speakers(l, args.cap_per_cell, args.clips_per_speaker, args.seed)
              for l in langs]
    metadata = pd.concat(frames, ignore_index=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    metadata.to_csv(METADATA, index=False)
    print(f"\n{METADATA}: {metadata.client_id.nunique()} speakers | {len(metadata)} clips")

    if not args.skip_audio:
        for lang, df in zip(langs, frames):
            download_audio(lang, set(df.path), DATA_DIR)
