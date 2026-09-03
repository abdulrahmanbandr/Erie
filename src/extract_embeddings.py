"""Extract ECAPA speaker embeddings from audio clips."""

import argparse
import glob
import os

import librosa
import numpy as np
import pandas as pd
import torch
from speechbrain.inference import EncoderClassifier

ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
SAMPLE_RATE = 16000


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract(csv_path: str, audio_dir: str, out_path: str) -> None:
    device = get_device()
    print("device:", device)

    encoder = EncoderClassifier.from_hparams(
        source=ECAPA, savedir="tmp_ecapa", run_opts={"device": device})

    df = pd.read_csv(csv_path)
    df = df[df.age_group != "50+"].copy()

    found = {os.path.basename(p): p
             for p in glob.glob(f"{audio_dir}/**/*.mp3", recursive=True)}
    df["fullpath"] = df["path"].map(found)
    df = df.dropna(subset=["fullpath"])
    print(f"matched {len(df)} clips")

    embeddings, kept, errors = [], [], []
    for i, row in df.iterrows():
        try:
            # ECAPA expects 16 kHz; Common Voice ships 48 kHz.
            audio, _ = librosa.load(row.fullpath, sr=SAMPLE_RATE, mono=True)
            sig = torch.tensor(audio).unsqueeze(0).to(device)
            emb = encoder.encode_batch(sig)
            embeddings.append(emb.squeeze().detach().cpu().numpy())
            kept.append(i)
        except Exception as exc:
            errors.append(f"{row.path}: {type(exc).__name__}")

        if embeddings and len(embeddings) % 500 == 0:
            print(f"{len(embeddings)} / {len(df)}")

    print(f"processed: {len(embeddings)} | failed: {len(errors)}")
    if errors:
        print("sample errors:", errors[:3])

    df = df.loc[kept].reset_index(drop=True)
    np.savez(out_path,
             X=np.array(embeddings),
             y=df.age_group.values,
             gender=df.gender.values,
             groups=df.client_id.values)   # required for speaker-safe splits
    print("saved:", out_path, np.array(embeddings).shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="ru_age.csv")
    parser.add_argument("--audio-dir", default="data/audio/ru")
    parser.add_argument("--out", default="ru_age_emb.npz")
    args = parser.parse_args()

    extract(args.csv, args.audio_dir, args.out)