"""Predict age and gender for one or more audio files with the final model.

Needs only the repo: the fitted models in models/eiry_v4.joblib and the
WavLM encoder, which downloads on first use. Each clip is cut into
6-second windows, every window is scored, and the predictions are
averaged, the same speaker-level averaging used in the evaluation.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import SR, get_device  # noqa: E402

WINDOW_SEC = 6.0
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "eiry_v4.joblib")


def load_encoder(name):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel
    device = get_device()
    fe = AutoFeatureExtractor.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).to(device).eval()
    return fe, enc, device


def embed_windows(path, fe, enc, device, layer):
    """Mean-pooled features of one encoder layer for each 6 s window."""
    import librosa
    import torch
    audio, _ = librosa.load(path, sr=SR, mono=True)
    n = int(WINDOW_SEC * SR)
    starts = list(range(0, max(1, len(audio) - n // 2), n)) or [0]
    feats = []
    with torch.no_grad():
        for s in starts:
            chunk = audio[s:s + n]
            if len(chunk) < SR:                    # skip windows under 1 s
                continue
            x = fe(chunk, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            h = enc(x, output_hidden_states=True).hidden_states[layer]
            feats.append(h[0].mean(0).float().cpu().numpy())
    return np.stack(feats), len(audio) / SR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", help="audio file(s): wav, mp3, m4a, opus, flac ...")
    parser.add_argument("--model-file", default=DEFAULT_MODEL)
    args = parser.parse_args()

    import joblib
    if not os.path.exists(args.model_file):
        sys.exit(f"model file not found: {args.model_file}\n"
                 "Create it with: python src/train_final.py (needs the V4 features)")
    models = joblib.load(args.model_file)
    print(f"model: {os.path.basename(args.model_file)} | {models['encoder']} layer {models['layer']} | "
          f"trained on {models['n_speakers']} speakers")
    fe, enc, device = load_encoder(models["encoder"])

    print(f"\n{'file':40s} {'length':>7s} {'windows':>7s} {'age':>6s} {'p(female)':>10s}  gender")
    for path in args.audio:
        F, seconds = embed_windows(path, fe, enc, device, models["layer"])
        age = float(np.mean(models["age"].predict(F)))
        p_f = float(np.mean(models["gender"].predict_proba(F)[:, 1]))
        label = "female" if p_f >= 0.5 else "male"
        print(f"{os.path.basename(path)[:40]:40s} {seconds:6.1f}s {len(F):7d} {age:6.1f} {p_f:10.2f}  {label}")
    print("\nAge is an estimate with a typical error of about 8 years; predictions for the very young"
          " and very old are pulled toward the middle.")


if __name__ == "__main__":
    main()
