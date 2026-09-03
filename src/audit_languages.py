"""Speaker counts per age bin and gender for Common Voice languages,
read from the remote TSVs without downloading audio.

Use it to choose extra training languages: you want teens and 50+
speakers of both genders, not just a big corpus.
"""

import argparse

import pandas as pd

BASE = "https://huggingface.co/datasets/fsicoli/common_voice_21_0/resolve/main/transcript"
SPLITS = ["train", "dev", "test"]
COLS = ["client_id", "age", "gender"]
AGE_ORDER = ["teens", "twenties", "thirties", "fourties", "fifties",
             "sixties", "seventies", "eighties", "nineties"]
GENDERS = {"male_masculine": "m", "female_feminine": "f"}


def audit(lang: str) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f"{BASE}/{lang}/{s}.tsv", sep="\t", usecols=COLS,
                                low_memory=False) for s in SPLITS], ignore_index=True)
    n_all = df.client_id.nunique()
    df = df[df.gender.isin(GENDERS)].dropna(subset=["age"])
    spk = df.drop_duplicates("client_id")
    spk = spk.assign(g=spk.gender.map(GENDERS))
    tab = pd.crosstab(spk.age, spk.g).reindex(AGE_ORDER).fillna(0).astype(int)
    tab = tab.copy()
    tab["total"] = tab.sum(axis=1)
    print(f"\n=== {lang}: {n_all} speakers, {len(spk)} with age+gender ===")
    print(tab.to_string())
    return tab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="de,fr,ca,it,es,pl,uk")
    args = parser.parse_args()
    for lang in args.langs.split(","):
        try:
            audit(lang)
        except Exception as exc:
            print(f"\n=== {lang}: failed ({type(exc).__name__}: {exc})")
