"""Fine-tune an audio classifier on the age (or any subset of) classes.

Two-stage recipe that recovered the wav2vec2 run from its collapse:
  1. --freeze-encoder, lr 1e-3: train only the head, with the weighted
     layer sum reading all encoder layers.
  2. --init-from <stage-1 dir>, lr 2e-5: unfreeze the transformer.

Always: class-weighted cross-entropy, the CNN feature encoder frozen,
early stopping on val macro-F1, the test split scored once at the end.
Note: layerdrop is forced to 0. In transformers 5 a dropped layer emits
no hidden state, which breaks the weighted layer sum (12 vs 13 states).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

sys.path.insert(0, os.path.dirname(__file__))
from common import DEFAULT_SCHEME, SEED, SR, classes_for, load_cache, load_splits  # noqa: E402

BATCH, ACCUM = 8, 2


class ClipDataset(Dataset):
    def __init__(self, frame, audio):
        self.paths, self.labels, self.audio = frame.fullpath.tolist(), frame.label.tolist(), audio

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"audio": self.audio[self.paths[i]].astype(np.float32),
                "labels": self.labels[i]}


def make_collate(fe):
    def collate(batch):
        out = fe([b["audio"] for b in batch], sampling_rate=SR,
                 padding=True, return_tensors="pt")
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out
    return collate


def metrics(ep):
    return {"f1_macro": f1_score(ep.label_ids, ep.predictions.argmax(-1), average="macro")}


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        logits = model(**inputs).logits
        loss = nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights.to(logits.device))
        # Return logits only: with use_weighted_layer_sum the model also
        # returns every hidden state, which the Trainer would accumulate.
        return (loss, {"logits": logits}) if return_outputs else loss


def run_finetune(model_name, classes, tag, freeze_encoder, lr, epochs, splits, cache,
                 scheme=DEFAULT_SCHEME, init_from=None, precision="fp32", patience=3,
                 out_dir="runs", results_dir="results/v3"):
    df = load_splits(splits, scheme)
    audio = load_cache(cache)
    df = df[df.keep & df.fullpath.isin(audio)]

    sub = df[df.age_class.isin(classes)].copy()
    c2i = {c: i for i, c in enumerate(classes)}
    sub["label"] = sub.age_class.map(c2i)
    print(f"scheme {scheme} | classes {classes} | {len(sub)} clips | {sub.client_id.nunique()} speakers")
    tr, va, te = (sub[sub.split == s] for s in ["train", "val", "test"])
    w = torch.tensor(compute_class_weight("balanced", classes=np.arange(len(classes)),
                                          y=tr.label.values), dtype=torch.float32)
    print("class weights:", dict(zip(classes, w.numpy().round(2))))
    print(f"uniform-predictor loss = ln({len(classes)}) = {np.log(len(classes)):.3f}")

    fe = AutoFeatureExtractor.from_pretrained(model_name)
    src = init_from or model_name
    model = AutoModelForAudioClassification.from_pretrained(
        src, num_labels=len(classes), label2id=c2i,
        id2label={i: c for c, i in c2i.items()},
        use_weighted_layer_sum=True, layerdrop=0.0)
    for p in model.base_model.parameters():
        p.requires_grad = not freeze_encoder
    model.freeze_feature_encoder()

    steps_per_epoch = int(np.ceil(len(tr) / (BATCH * ACCUM)))
    run_dir = os.path.join(out_dir, tag)
    args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=BATCH,
        learning_rate=lr, num_train_epochs=epochs,
        warmup_steps=int(0.1 * steps_per_epoch * epochs), max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        greater_is_better=True, logging_steps=25,
        fp16=(precision == "fp16"), bf16=(precision == "bf16"),
        remove_unused_columns=False, report_to="none", seed=SEED)

    trainer = WeightedTrainer(
        model=model, args=args, class_weights=w,
        train_dataset=ClipDataset(tr, audio), eval_dataset=ClipDataset(va, audio),
        data_collator=make_collate(fe), compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    trainer.train()
    trainer.save_model(os.path.join(run_dir, "best"))

    p = trainer.predict(ClipDataset(te, audio))
    pred = p.predictions.argmax(-1)
    print("\n== TEST ==")
    print(classification_report(p.label_ids, pred, labels=range(len(classes)),
                                target_names=classes, digits=3, zero_division=0))

    epochs_log = [{k: e[k] for k in ("epoch", "eval_loss", "eval_f1_macro") if k in e}
                  for e in trainer.state.log_history if "eval_loss" in e]
    res = {"tag": tag, "model": model_name, "scheme": scheme, "classes": classes,
           "init_from": init_from, "freeze_encoder": freeze_encoder,
           "lr": lr, "epochs": epochs, "precision": precision,
           "best_val_f1": trainer.state.best_metric,
           "test_f1_macro": float(f1_score(p.label_ids, pred, average="macro")),
           "n_train_clips": len(tr), "n_val_clips": len(va), "n_test_clips": len(te),
           "val_by_epoch": epochs_log}
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"finetune_{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved:", out)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--scheme", default=DEFAULT_SCHEME,
                        choices=["fine4", "coarse3", "coarse3_gap"])
    parser.add_argument("--classes", default=None,
                        help="comma-separated subset of the scheme's classes; default: all")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--init-from", default=None,
                        help="directory saved by a previous run (<out-dir>/<tag>/best)")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--splits", default="ru_splits_all.csv")
    parser.add_argument("--cache", default="audio_cache.pkl")
    parser.add_argument("--out-dir", default="runs")
    args = parser.parse_args()

    precision = args.precision
    if precision == "auto":
        precision = ("bf16" if torch.cuda.is_available()
                     and torch.cuda.get_device_capability()[0] >= 8 else "fp32")

    classes = args.classes.split(",") if args.classes else classes_for(args.scheme)
    run_finetune(args.model, classes, args.tag, args.freeze_encoder,
                 args.lr, args.epochs, args.splits, args.cache,
                 scheme=args.scheme, init_from=args.init_from, precision=precision,
                 patience=args.patience, out_dir=args.out_dir)
