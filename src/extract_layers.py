"""Mean-pooled hidden states from every layer of a frozen encoder.

Output: a .npy array of shape (n_layers + 1, n_clips, dim), rows in the
order of the split CSV. Works for wav2vec2 / WavLM style models and for
Whisper encoders (detected by name).
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import SR, get_device, load_cache, load_splits  # noqa: E402


def extract_ssl(name, paths, audio, device):
    from transformers import AutoFeatureExtractor, AutoModel
    fe = AutoFeatureExtractor.from_pretrained(name)
    enc = AutoModel.from_pretrained(name).to(device).eval()
    feats = [[] for _ in range(enc.config.num_hidden_layers + 1)]
    with torch.no_grad():
        for i, p in enumerate(paths):
            x = fe(audio[p].astype(np.float32), sampling_rate=SR,
                   return_tensors="pt").input_values.to(device)
            hs = enc(x, output_hidden_states=True).hidden_states
            for l, h in enumerate(hs):
                feats[l].append(h[0].mean(0).float().cpu().numpy())
            if i % 1000 == 0:
                print(i)
    return feats


def extract_whisper(name, paths, audio, device):
    """Whisper pads to 30 s; pool only the frames that contain speech (50 / s)."""
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained(name)
    enc = WhisperModel.from_pretrained(name).encoder.to(device).eval()
    feats = [[] for _ in range(enc.config.encoder_layers + 1)]
    with torch.no_grad():
        for i, p in enumerate(paths):
            a = audio[p].astype(np.float32)
            x = fe(a, sampling_rate=SR, return_tensors="pt").input_features.to(device)
            n = max(1, int(np.ceil(len(a) / SR * 50)))
            hs = enc(x, output_hidden_states=True).hidden_states
            for l, h in enumerate(hs):
                feats[l].append(h[0, :n].mean(0).float().cpu().numpy())
            if i % 1000 == 0:
                print(i)
    return feats


def extract(model: str, splits: str, cache: str, out: str) -> None:
    device = get_device()
    df = load_splits(splits)
    audio = load_cache(cache)
    paths = [p for p in df.fullpath if p in audio]
    print(f"model: {model} | device: {device} | clips: {len(paths)}")

    fn = extract_whisper if "whisper" in model.lower() else extract_ssl
    feats = fn(model, paths, audio, device)
    X = np.stack([np.stack(f) for f in feats])
    np.save(out, X)
    print("saved:", out, X.shape)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--splits", default="ru_splits_all.csv")
    parser.add_argument("--cache", default="audio_cache.pkl")
    parser.add_argument("--out", default=None,
                        help="default: layers_<model-short-name>.npy")
    args = parser.parse_args()

    out = args.out or f"layers_{args.model.split('/')[-1]}.npy"
    extract(args.model, args.splits, args.cache, out)
