"""Estimate age and gender from audio files with the trained model.

Each recording is cut into 6-second windows, every window is scored and
the predictions are averaged. Windows shorter than 1 second are skipped.

    python src/predict.py clip.wav [more files ...]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CLIP_SEC, MODEL_FILE, SR, embed, load_encoder  # noqa: E402


def windows(audio: np.ndarray) -> list:
    n = int(CLIP_SEC * SR)
    starts = range(0, max(1, len(audio) - n // 2), n)
    return [audio[s:s + n] for s in starts if len(audio[s:s + n]) >= SR]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", nargs="+", help="audio file(s): wav, mp3, flac, ogg, m4a ...")
    parser.add_argument("--model-file", default=MODEL_FILE)
    args = parser.parse_args()

    import joblib
    import librosa
    if not os.path.exists(args.model_file):
        sys.exit(f"model file not found: {args.model_file}")
    models = joblib.load(args.model_file)
    fe, enc, device = load_encoder(models["encoder"])

    print(f"\n{'file':32s} {'length':>7s} {'age':>6s} {'p(female)':>10s}  gender")
    for path in args.audio:
        name = os.path.basename(path)[:32]
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as exc:
            print(f"{name:32s} could not read the file ({type(exc).__name__})")
            continue
        chunks = windows(audio)
        if not chunks:
            print(f"{name:32s} too short: needs at least 1 second of audio")
            continue
        F = np.stack([embed(c, fe, enc, device, models["layer"]) for c in chunks])
        age = float(np.mean(models["age"].predict(F)))
        p_female = float(np.mean(models["gender"].predict_proba(F)[:, 1]))
        gender = "female" if p_female >= 0.5 else "male"
        print(f"{name:32s} {len(audio) / SR:6.1f}s {age:6.1f} {p_female:10.2f}  {gender}")

    print("\nAge is an estimate: typical error is about 8 years, and very young or"
          " very old voices are pulled toward the middle.")


if __name__ == "__main__":
    main()
