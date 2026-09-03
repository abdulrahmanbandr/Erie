"""Shared constants and loaders for the V2 / V3 scripts."""

import pickle

import pandas as pd

SR = 16000
SEED = 42

# Raw Common Voice bins. "50+" is the merged bin written by the old V1 data_prep.
OLDER_BINS = ["fifties", "sixties", "seventies", "eighties", "nineties", "50+"]
AGE_MIDPOINTS = {"teens": 16.0, "twenties": 25.0, "thirties": 35.0, "fourties": 45.0,
                 "fifties": 55.0, "sixties": 65.0, "seventies": 75.0, "eighties": 85.0,
                 "nineties": 92.0, "50+": 60.0}

# Age schemes: raw Common Voice bin -> class. Class lists are in ordinal order.
AGE_SCHEMES = {
    "fine4": {
        "map": {"teens": "teens", "twenties": "twenties",
                "thirties": "thirties", "fourties": "fourties"},
        "classes": ["teens", "twenties", "thirties", "fourties"],
    },
    "coarse3": {
        "map": {"teens": "young", "twenties": "adult", "thirties": "adult",
                "fourties": "adult", **{b: "older" for b in OLDER_BINS}},
        "classes": ["young", "adult", "older"],
    },
    # proof-of-concept variant: twenties left out as a buffer between young and adult
    "coarse3_gap": {
        "map": {"teens": "young", "thirties": "adult", "fourties": "adult",
                **{b: "older" for b in OLDER_BINS}},
        "classes": ["young", "adult", "older"],
    },
}
DEFAULT_SCHEME = "fine4"
CLASSES = AGE_SCHEMES[DEFAULT_SCHEME]["classes"]   # backwards compatibility


def classes_for(scheme: str) -> list:
    return AGE_SCHEMES[scheme]["classes"]


def load_splits(path: str, scheme: str | None = None) -> pd.DataFrame:
    """Load the split CSV written by make_splits.py.

    With scheme=None every row is returned unchanged (audio cache and
    feature extraction work on all rows, in CSV order). With a scheme,
    columns age_class / label / is_female / keep are added and the frame
    is still returned unfiltered so that row order matches feature arrays.
    Callers filter with df[df.keep].
    """
    df = pd.read_csv(path)
    if scheme is None:
        return df
    spec = AGE_SCHEMES[scheme]
    df["age_class"] = df.age_group.map(spec["map"])
    df["keep"] = df.age_class.notna()
    df["label"] = df.age_class.map({c: i for i, c in enumerate(spec["classes"])})
    df["is_female"] = df.gender.str.startswith("female")
    return df


def load_cache(path: str) -> dict:
    """Load the decoded-audio cache written by cache_audio.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
