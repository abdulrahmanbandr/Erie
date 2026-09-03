"""Decode every clip once (16 kHz mono, truncated) and cache to a pickle.

Decoding mp3 is the slowest step of every experiment; do it once.
The cache is a dict {fullpath: float16 numpy array}.
"""

import argparse
import os
import pickle
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import SR, load_splits  # noqa: E402


def cache_audio(splits: str, out: str, max_sec: float) -> None:
    df = load_splits(splits)
    audio, bad = {}, []
    for i, p in enumerate(df.fullpath):
        try:
            y, _ = librosa.load(p, sr=SR, mono=True, duration=max_sec)
            audio[p] = y.astype(np.float16)
        except Exception as exc:
            bad.append(f"{p}: {type(exc).__name__}")
        if i % 500 == 0:
            print(f"{i} / {len(df)}")

    with open(out, "wb") as f:
        pickle.dump(audio, f)
    print(f"cached {len(audio)} clips | failed {len(bad)} | saved: {out}")
    if bad:
        print("sample failures:", bad[:3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="ru_splits_all.csv")
    parser.add_argument("--out", default="audio_cache.pkl")
    parser.add_argument("--max-sec", type=float, default=6.0)
    args = parser.parse_args()

    cache_audio(args.splits, args.out, args.max_sec)
