"""Step 3: turn every clip into a 768-number feature vector.

Each clip is loaded at 16 kHz, cut to its first 6 seconds and passed
through the frozen WavLM encoder; the hidden states of layer 5 are
averaged over time. Writes data/features.npz. A GPU makes this much faster.
"""

import argparse
import os
import sys

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLIP_SEC, ENCODER, FEATURES, LAYER, SPLITS, SR,  # noqa: E402
                    embed, load_encoder)


def extract(splits: str, out: str, encoder: str, layer: int) -> None:
    df = pd.read_csv(splits)
    fe, enc, device = load_encoder(encoder)
    print(f"{encoder} layer {layer} | device {device} | {len(df)} clips")

    X, names, failed = [], [], []
    for i, row in enumerate(df.itertuples()):
        try:
            audio, _ = librosa.load(row.fullpath, sr=SR, mono=True, duration=CLIP_SEC)
            X.append(embed(audio, fe, enc, device, layer))
            names.append(row.path)
        except Exception as exc:
            failed.append(f"{row.path}: {type(exc).__name__}")
        if i % 500 == 0:
            print(f"{i} / {len(df)}")

    np.savez(out, X=np.stack(X), path=np.array(names), encoder=encoder, layer=layer)
    print(f"saved: {out} | {len(X)} clips | {len(failed)} failed")
    if failed:
        print("failed clips are left out, e.g.", failed[:3])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", default=SPLITS)
    parser.add_argument("--out", default=FEATURES)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--layer", type=int, default=LAYER)
    args = parser.parse_args()

    extract(args.splits, args.out, args.encoder, args.layer)
