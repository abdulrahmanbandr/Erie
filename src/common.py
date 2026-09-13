"""Settings and helpers shared by every script."""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
METADATA = os.path.join(DATA_DIR, "metadata.csv")
SPLITS = os.path.join(DATA_DIR, "splits.csv")
FEATURES = os.path.join(DATA_DIR, "features.npz")
MODEL_FILE = os.path.join(ROOT, "models", "eiry.joblib")
METRICS = os.path.join(ROOT, "results", "metrics.json")

SR = 16000                          # sample rate the encoder expects
CLIP_SEC = 6.0                      # clip length in training, window length in prediction
SEED = 42
ENCODER = "microsoft/wavlm-base-plus"
LAYER = 5                           # age information peaks in the middle layers

# Common Voice labels age in decade bins; the regression target is the bin midpoint.
AGE_MIDPOINTS = {"teens": 16.0, "twenties": 25.0, "thirties": 35.0, "fourties": 45.0,
                 "fifties": 55.0, "sixties": 65.0, "seventies": 75.0, "eighties": 85.0,
                 "nineties": 92.0}


def get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(name: str = ENCODER):
    from transformers import AutoFeatureExtractor, AutoModel
    device = get_device()
    fe = AutoFeatureExtractor.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).to(device).eval()
    return fe, enc, device


def embed(audio: np.ndarray, fe, enc, device: str, layer: int = LAYER) -> np.ndarray:
    """Hidden states of one encoder layer, averaged over time: one 768-d vector."""
    import torch
    with torch.no_grad():
        x = fe(audio.astype(np.float32), sampling_rate=SR,
               return_tensors="pt").input_values.to(device)
        h = enc(x, output_hidden_states=True).hidden_states[layer]
    return h[0].mean(0).float().cpu().numpy()


def load_dataset(splits: str = SPLITS, features: str = FEATURES):
    """Split table and feature matrix with matching rows, plus encoder info."""
    df = pd.read_csv(splits)
    data = np.load(features)
    row = {name: i for i, name in enumerate(data["path"])}
    df = df[df.path.isin(row)].reset_index(drop=True)
    X = data["X"][df.path.map(row).values]
    df["age"] = df.age_group.map(AGE_MIDPOINTS)
    df["is_female"] = df.gender.str.startswith("female")
    return df, X, {"encoder": str(data["encoder"]), "layer": int(data["layer"])}
