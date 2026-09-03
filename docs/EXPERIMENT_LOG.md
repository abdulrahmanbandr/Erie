# Experiment Log

Every run, including the failed ones. Numbers are macro-F1 unless stated.

## Dataset

- Common Voice 21.0 Russian (`fsicoli/common_voice_21_0` mirror)
- 1,184 speakers, 5,333 clips (capped 5 / speaker)
- 4 classes: teens, twenties, thirties, fourties
- `50+` dropped: 145 clips, precision 0.02 in an early run. This decision
  was made after seeing test performance. It also means the model cannot
  be applied to older Adyghe speakers.

### Rejected datasets

- **Adyghe (ady, CV 26.0)**: 1 male speaker / 168 → gender untrainable and
  unevaluable; age labels female-only in practice. Kept as transfer target.
- **Sh1man/common_voice_21_ru**: no `client_id`, no age / gender (ASR repack).

### Language choice

Compared ru / tr / fa / de from the TSVs before downloading audio.
Russian chosen: 1,305 labeled speakers, 3.1:1 male skew.

---

## V1

### Exp 1 — ECAPA frozen + LogisticRegression

`speechbrain/spkrec-ecapa-voxceleb` (192-d), StandardScaler,
LogisticRegression(balanced), GroupKFold(5) on `client_id`.

| Task | macro-F1 |
|---|---|
| Age (4-class) | 0.360 ± 0.022 |
| Gender | 0.947 |

Within noise (±0.022): `class_weight` on / off, MLPClassifier,
speaker-level embedding averaging.

Open item: confirm whether the 0.947 gender number was computed on the
speaker-balanced gender CSV or on the male-skewed age set.
`train_baseline.py` reads gender labels from the age embedding file.

### Exp 2 — wav2vec2-base fine-tuning (failed)

Feature encoder frozen, `Wav2Vec2ForSequenceClassification` default head
(last layer mean-pooled), unweighted loss, single learning rate.

| Config | macro-F1 |
|---|---|
| lr 3e-5, fp16, 1 epoch | 0.1599 |
| lr 5e-5, fp16, 8 epochs | 0.1599 |
| lr 1e-4, fp32, 5 epochs | 0.1599 |
| lr 1e-4, mask / layerdrop 0 | 0.1599 |
| lr 1e-4, class-weighted loss, 5 epochs | 0.1599 |

Observations:
- Unweighted runs: val loss 1.212 ≈ class-distribution entropy 1.216.
  Constant prediction = class prior.
- Class-weighted run: val loss 1.35 ≈ ln 4 = 1.386.
  Constant prediction = uniform. Weighting alone did not fix it.
- Reported train loss is inflated ×2 by `gradient_accumulation_steps`
  (train ≈ 2.4 with val ≈ 1.2). Not a real train / val inversion.
- 50-sample overfit test passed (1.18 → 0.02). This only proves gradients
  flow; it would also pass with shuffled labels.
- Same recipe on **gender**: 0.953 in 3 epochs. The pipeline learns from
  audio; the age failure was not mechanical.

Original conclusion ("95M params insufficient for 1,184 speakers") was
**wrong**. See V2 Exp 4 for the actual cause.

Figure: `results/figures/v1_wav2vec2_finetune_loss_curves.png`.

---

## V2

Fixed speaker-level split, seed 42: train 769 speakers / 3,456 clips,
val 178 / 788, test 237 / 1,089. Clips truncated to 6 s. Val is used for
layer and checkpoint selection; test scored once per run.

Colab notebook with outputs: `notebooks/v2_layer_probe_and_finetune.ipynb`.

### Exp 3 — layer-wise linear probes

Mean-pooled hidden states of every layer, frozen encoder, same classifier
as Exp 1, StratifiedGroupKFold(5) on train+val speakers.

| Encoder | Layers | Best age layer | Age | Gender (best) |
|---|---|---|---|---|
| wav2vec2-base | 13 | 5 | 0.333 ± 0.019 | 0.966 (layers 3–4) |
| wavlm-base-plus | 13 | 6 | 0.363 ± 0.017 | 0.961 (layers 2–4) |
| whisper-base encoder | 7 | 3 | 0.339 ± 0.032 | 0.960 (layer 2) |

Per-layer values in `results/v2/summary.json`;
plots in `results/figures/v2_probe_*.png`.

Findings:
- Age peaks in the middle layers of every model and drops at the top.
  The default sequence-classification head reads only the last layer.
- All three probes and ECAPA (Exp 1) land in 0.33–0.36. Differences are
  within fold noise.
- Age F1 by gender at the best layer: wav2vec2 male 0.333 / female 0.334;
  WavLM male 0.374 / female 0.331. Female performance does not collapse
  despite 1:3 representation. Fourties is the weakest class for both.

### Exp 4 — wav2vec2-base fine-tuning, fixed recipe

Changes from Exp 2: `use_weighted_layer_sum=True`, class-weighted
cross-entropy, two stages (head only at lr 1e-3, then encoder unfrozen at
lr 2e-5), early stopping on val macro-F1, fp32.

| Classes | Stage | Epochs run | Best val | Test |
|---|---|---|---|---|
| 4-class | head only | 8 | 0.375 (ep 8) | 0.384 |
| 4-class | unfrozen, from head-only | 7 (early stop) | 0.381 (ep 3) | 0.388 |
| teens vs fourties | head only | 8 | 0.722 | 0.698 |
| teens vs fourties | unfrozen | 7 (early stop) | 0.801 (ep 4) | 0.778 |

Head-only, 4-class, val by epoch:
loss 1.338 → 1.338 (flat, ≈ ln 4), F1 0.278 → 0.375.
The model is right more often than chance with near-uniform
probabilities: small margins, weak signal.

Unfrozen, 4-class, val by epoch:
loss 1.39 → 1.56 → 1.57 → 1.92 → 2.15 → 2.42 → 3.04 while train loss
fell 1.94 → 0.19. Memorization. Test gain over head-only: +0.004.

Conclusions:
- Exp 2 failed because of (a) reading only the last layer, (b) one small
  learning rate shared by a random head and a pretrained encoder, (c) the
  class prior. Fixing (a) and (b) recovers 0.38 on the same data.
- For the 4-class task, a frozen encoder with a trained head is the right
  operating point. Unfreezing adds overfitting and no accuracy.
- Extreme bins (teens vs fourties) separate at 0.78. The 4-class loss is
  in adjacent bins (teens/twenties, thirties/fourties).
- Four representations and two training regimes converge in 0.33–0.39.
  This looks like a label / clip-length ceiling, not a model ceiling.

Caveat: probe scores are 5-fold CV over 947 speakers; fine-tune scores
are a single 237-speaker test split. "0.388 beats 0.363" is not
established until both are scored on the same speakers.

### Exp 5 — WavLM fine-tuning (crashed, not re-run)

`RuntimeError: size of tensor a (12) must match ... b (13)`.
Cause: WavLM config has `layerdrop=0.05`; in transformers 5 a dropped
layer emits no hidden state, so the weighted layer sum receives 12 states
for 13 weights. Fix: `layerdrop=0.0` at load time (in `src/finetune.py`).

---

## V3 — coarse age classes and age regression (Russian only)

Decision (2026-09-08): build a model that works before optimizing.
V2 showed the extremes separate (teens vs fourties 0.78) and the 4-class
loss is entirely in adjacent bins. So: collapse to three ordinal classes
and restore the 50+ speakers that V1 dropped.

| Class | Raw bins | Russian speakers (approx.) |
|---|---|---|
| young | teens | ~190 |
| adult | twenties, thirties, fourties | ~990 |
| older | 50+ | ~30 |

Not "child / adult / old": Common Voice has no children (teens = 13–19,
mostly post-pubertal), and "older" starts at fifties, where acoustic
aging cues are weak. Both caveats stay in the writeup.

Gender stays a separate model (0.95 already). Age x gender cells are the
product of the two outputs, not a 6-way label.

Implementation: `--scheme` in `probe_layers.py` and `finetune.py`
(`fine4`, `coarse3`, `coarse3_gap`). `make_splits.py` now keeps all raw
bins and writes `ru_splits_all.csv`; class mapping happens at load time.
The split therefore differs from V2's (50+ speakers added). V2 numbers
stay tied to the old CSV.

Baselines to beat under `coarse3`: majority-class macro-F1 ≈ 0.30.
Expectation before running: young ~0.7, adult ~0.9, older 0.3–0.6 with
a wide interval (30 speakers). Record the older-class support with every
number.

### Exp 6 — coarse3 probe, WavLM (2026-09-08)

1,215 speakers: adult 1,014 / young 170 / older 31. Test: 243 speakers,
older = 3 speakers / 15 clips. Majority-class macro-F1 0.30.

| | young | adult | older | macro |
|---|---|---|---|---|
| CV best layer (4) | – | – | – | 0.417 ± 0.029 |
| test, clip level | 0.31 | 0.83 | 0.00 | 0.379 |
| test, speaker level | | | | 0.394 |

By gender (CV): male 0.424, female 0.403; older F1 0.11 / 0.07.

Reading:
- older is unlearnable and unevaluable at 31 speakers. Data, not model.
- young collapsed because the boundary moved to teens vs twenties, which
  the model cannot hear. The V2 binary (teens vs fourties, 0.78) worked
  because of the distance between classes, not the number of classes.
- Coarsening by itself bought nothing. Age resolution is ~a decade; any
  boundary inside that resolution fails at that boundary.

Decision: predict age as a number (MAE in years) on ages 13–49 with the
speakers available; add German speakers later for an older class;
children out of scope (no under-13 speakers in Common Voice, "teens" bin
is 13–19 and cannot be split). `src/regress_age.py` implements the
regression probe (targets = bin midpoints; 50+ optional at 60).

### Exp 7 — age regression, WavLM (2026-09-08)

Ridge on bin midpoints, ages 13–49, 1,184 speakers. Best layer 6.

| | MAE (years) | Spearman |
|---|---|---|
| predict-the-mean baseline | 6.92 | – |
| clip level | 5.97 | 0.43 |
| speaker level | 5.69 | 0.53 |

Mean predicted age by true bin: teens 25.6, twenties 27.9, thirties 30.3,
fourties 32.9. Correct ordering, range squeezed to 25–33 (alpha 433,
twenties = 48% of speakers). MAE by bin: twenties 4.2, thirties 5.2,
teens 9.6, fourties 12.1. Female / male MAE equal (5.9 / 6.0).
Re-binned 4-class macro-F1 0.31.

Reading: the features order speakers by age with roughly a decade of
resolution; the tails are missing from the training data, so the model
never learns to predict them. Same conclusion as Exp 6 from the other
direction. Decision: fix the data before more modelling. Language audit
in `docs/LANGUAGE_AUDIT.md`; Catalan chosen as the primary addition.

Not run: `--balanced`, `coarse3_gap`. Both address symptoms of the same
data gap.

---

## V4 — multilingual, age-balanced training set (final)

Decision (2026-09-08): Catalan + German + Russian. Catalan is the only
Common Voice corpus balanced in both gender and age, with the largest
population over fifty; German adds language diversity. Audit in
`docs/LANGUAGE_AUDIT.md`.

Build (`src/data_prep.py --langs ca,de,ru`, cap 100 speakers per
language × age bin × gender, 4 clips / speaker, seed 42):

| Language | Speakers | Clips | Dropped (inconsistent labels) |
|---|---|---|---|
| ca | 1,385 | 5,392 | 49 |
| de | 1,138 | 4,206 | 18 |
| ru | 652 | 2,466 | 7 |
| total | 3,169 | 12,064 | |

Speakers over fifty: ~1,350 (was 31). Female teens: 188 (was 46).
Russian shrinks to 652 because its twenties/thirties male cells are
capped like everyone else's.

Rule enforced by the cap: every age bin is present in every language,
so a language classifier cannot double as an age classifier. The
per-language MAEs in Exp 8 (7.9–9.7) and the hold-out results in Exp 10
are consistent with that.

Download: shard by shard from the mirror (31 + 16 + 1 train shards,
~70 GB transferred, ~2 GB kept on disk).

Split (`splits_v4.csv`, seed 42): train 2,059 speakers / 7,821 clips,
val 476 / 1,818, test 634 / 2,425. All three languages in every split.

### Exp 8 — age regression, WavLM, V4 set (2026-09-09)

Ridge on bin midpoints, all bins. Best layer 5 (CV MAE 9.32 ± 0.22;
predict-the-mean 15.8).

| Test | MAE (years) | median AE | Spearman |
|---|---|---|---|
| predict-the-mean baseline | 16.2 | – | – |
| clip level | 9.23 | 7.8 | 0.79 |
| speaker level (634 speakers) | 8.23 | 7.2 | 0.84 |

Mean predicted age by true bin: teens 26.3, twenties 30.3, thirties 36.4,
fourties 43.7, fifties 49.3, sixties 55.6, seventies 62.9, eighties 71.6.
Monotonic across the full range (Exp 7 was squeezed into 25–33).

MAE by bin: thirties 6.8, fourties 7.3, twenties 8.2, fifties 8.9,
sixties 11.1, teens 11.3, seventies 13.0, eighties 13.4.
By gender: female 9.0, male 9.4. By language: ru 7.9, de 9.2, ca 9.7.
Re-binned 4-class macro-F1 (teens–fourties only): 0.416 (Exp 1: 0.360).

Reading:
- First working age model: human-level error (~8 years per speaker),
  rank correlation 0.84, equal across genders and languages.
- Extremes still pulled toward the middle (teens +10, seventies −12);
  alpha 433. Balanced weighting or a calibration line should reduce it.
- Bins with < 20 speakers (eighties 14, nineties 4) are unreliable;
  merge into 80+ or exclude from reports.
- The multilingual data also lifted the original 4-bin task (0.36 → 0.42)
  on a harder test set.

Results: `results/v4/regress_wavlm-base-plus.json`,
`results/figures/v4_regress_wavlm.png`, `results/figures/v4_age_by_bin.png`.

### Exp 9 — balanced weighting (2026-09-09)

Inverse-bin-frequency weights, same features, layer 5.
Speaker MAE 8.33 (plain 8.23), Spearman 0.83 (0.84). Seventies 13.0 →
10.2, eighties 13.4 → 11.1, teens 11.3 → 12.9, twenties 8.2 → 9.3.

Reading: the V4 set is already near-balanced from teens to seventies,
so the weights only inflate eighties (14 speakers) and nineties (4),
which stretched the top of the scale at everyone else's expense. Teens
stay at ~27 regardless: Common Voice teens are mostly 17–19, so the
midpoint 16 overstates the gap; the model's 4-year teens/twenties
separation is close to the real one. Not a balance problem.

Decision: the plain regressor (Exp 8) is the V4 model. Report eighties
and nineties merged as 80+. No further tuning before the transfer tests.

### Exp 10 — leave-one-language-out (2026-09-09)

Layer 5, plain ridge, trained on two languages (all splits), tested on
every speaker of the third.

| Held out | Train speakers | Test speakers | speaker MAE | Spearman | baseline |
|---|---|---|---|---|---|
| ca | 1,785 | 1,385 | 10.73 | 0.81 | 17.8 |
| de | 2,037 | 1,138 | 10.62 | 0.77 | 15.4 |
| ru | 2,522 | 652 | 8.14 | 0.57 | 14.9 |

In-language reference (Exp 8): 8.23 / 0.84.

Reading:
- Transfer to an unseen language works: ~2.5 years worse than
  in-language, ranking preserved (0.77–0.81). Slightly above the 2-year
  bar; real but not free.
- German held out shows a systematic offset, not noise: fifties → 41,
  sixties → 48, seventies → 54 (in-language 49 / 56 / 63). Ordering
  intact, scale shifted ~7 years younger. A linear calibration on a few
  labeled speakers of the target language would correct an offset like
  this.
- Russian's 0.57 is a range effect (ages 13–49 only); its MAE 8.1 equals
  in-language.

Results: `results/v4/regress_wavlm-base-plus_holdout-all.json`,
`results/figures/v4_holdout_language.png`.

### Adyghe (not run — future work)

The original target. Common Voice 26.0 Adyghe has ~168 speakers, 72
labeled female and 1 male; age labels are effectively female-only. The
audio is not on any public Hub mirror, and the project was closed with
the three-language transfer result instead.

Pre-registered protocol, implemented in `src/eval_adyghe.py`, all at
speaker level:
1. Zero-shot MAE and Spearman with a 1,000-resample bootstrap CI and a
   permutation test. Expectation from Exp 10: MAE ≈ 10–11.
2. Calibrated MAE: leave-one-speaker-out linear fit on the other
   speakers, to measure a language offset of the German kind.
3. Within-Adyghe leave-one-out ridge as a low-N in-domain reference.
4. Gender specificity: fraction of the women predicted female.
5. Mean predicted age per true bin; bins under 5 speakers not
   interpreted.

---

## Status (2026-09-10)

Closed with the V4 model:

| | Result |
|---|---|
| Gender | 0.95 macro-F1 |
| Age, in-language, per speaker | 8.2 years MAE, Spearman 0.84 (634 test speakers) |
| Age, unseen language | 10.6–10.7 years MAE, Spearman 0.77–0.81 |

Model: frozen WavLM-base-plus, layer 5, mean-pooled; ridge regression
(age) and logistic regression (gender); 3,169 speakers from Catalan,
German and Russian.

Not done, in order of value if the project is reopened:
1. Adyghe evaluation (protocol above; needs the audio).
2. Per-language calibration study: how many labeled speakers of a new
   language are needed to remove the offset seen on German.
3. A second feature family (openSMILE eGeMAPS) concatenated with WavLM,
   and Whisper-small, as cheap accuracy checks.
4. Confirm the 0.95 gender number on the V4 set with
   `probe_layers.py --splits splits_v4.csv` (V4 figure is from the V2
   and V3 Russian probes: 0.95–0.97 at every layer).
