"""Eiry local demo: record or upload a voice clip, get an age and gender estimate.

Runs entirely on this machine. Uses the fitted model in models/eiry.joblib
and the same windowed scoring as src/predict.py.

    python app.py            # then open http://127.0.0.1:7860
"""

import os
import sys

import gradio as gr
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from common import MODEL_FILE, SR, embed, load_encoder  # noqa: E402
from predict import windows  # noqa: E402

TYPICAL_ERROR = 8          # years, speaker-level MAE on the V4 test split
AGE_MIN, AGE_MAX = 10, 90  # range of the bar

models = joblib.load(MODEL_FILE)
fe, enc, device = load_encoder(models["encoder"])


def estimate(path):
    if not path:
        return empty("Record a few seconds of speech, or upload a clip, then press Estimate.")
    import librosa
    try:
        audio, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return empty("That file could not be read. Try a wav, mp3, m4a or ogg recording.")
    chunks = windows(audio)
    if not chunks:
        return empty("The clip is shorter than one second. Record at least a few seconds of speech.")
    F = np.stack([embed(c, fe, enc, device, models["layer"]) for c in chunks])
    seconds = len(audio) / SR
    age = float(np.mean(models["age"].predict(F)))
    p_f = float(np.mean(models["gender"].predict_proba(F)[:, 1]))
    return card(age, p_f, seconds, len(F))


def empty(msg):
    return f'<div class="eiry-card eiry-empty"><p>{msg}</p></div>'


def card(age, p_f, seconds, windows):
    lo, hi = age - TYPICAL_ERROR, age + TYPICAL_ERROR
    pct = lambda a: 100 * (min(max(a, AGE_MIN), AGE_MAX) - AGE_MIN) / (AGE_MAX - AGE_MIN)
    gender = "Female" if p_f >= 0.5 else "Male"
    conf = max(p_f, 1 - p_f)
    if conf >= 0.9:
        conf_txt = "confident"
    elif conf >= 0.7:
        conf_txt = "fairly confident"
    else:
        conf_txt = "unsure"
    clip_note = ("Short clip: one window scored. Longer speech gives a steadier estimate."
                 if windows < 3 else f"{windows} windows averaged over {seconds:.0f} seconds of audio.")
    return f"""
<div class="eiry-card">
  <div class="eiry-row">
    <div class="eiry-age">
      <span class="eiry-big">{age:.0f}</span>
      <span class="eiry-unit">years, give or take {TYPICAL_ERROR}</span>
    </div>
    <div class="eiry-gender">
      <span class="eiry-glabel">{gender}</span>
      <span class="eiry-gconf">{conf_txt} ({conf:.0%})</span>
    </div>
  </div>
  <div class="eiry-bar" role="img" aria-label="Estimated age {age:.0f}, likely between {lo:.0f} and {hi:.0f}">
    <div class="eiry-band" style="left:{pct(lo):.1f}%;width:{pct(hi)-pct(lo):.1f}%"></div>
    <div class="eiry-mark" style="left:{pct(age):.1f}%"></div>
  </div>
  <div class="eiry-ticks"><span>10</span><span>30</span><span>50</span><span>70</span><span>90</span></div>
  <p class="eiry-note">Most likely between {lo:.0f} and {hi:.0f}. {clip_note}</p>
</div>"""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
:root { --ink:#1E1B4B; --muted:#6B6890; --line:#E4E0F5; --ground:#F7F5FF; --card:#FFFFFF; --track:#E7E4FA;
        --indigo:#4F46E5; --violet:#7C3AED; --indigo-soft:#A5B4FC; }
body.dark { --ink:#EEEBFF; --muted:#A5A1C8; --line:#2A2750; --ground:#0F0E1F; --card:#181633; --track:#262347;
            --indigo:#818CF8; --violet:#A78BFA; --indigo-soft:#4F46E5; }
body, .gradio-container { background: var(--ground) !important; font-family: 'Manrope', ui-sans-serif, system-ui, sans-serif !important; color: var(--ink); }
.gradio-container { max-width: 600px !important; margin: 0 auto !important; padding: 56px 20px 40px !important; }
.eiry-head { margin-bottom: 20px; }
.eiry-head .eiry-name { font-size: 4.5rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; margin: 0; color: var(--violet); }
.eiry-head .eiry-name span { display: inline-block; width: 0.28em; height: 0.28em; border-radius: 50%; background: var(--indigo); margin-left: 0.08em; vertical-align: baseline; }
.eiry-head .eiry-tag { font-size: 1.05rem; color: var(--muted); line-height: 1.55; margin: 12px 0 0; max-width: 44ch; }
.eiry-card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 24px 24px 18px; margin-top: 8px; color: var(--ink); animation: eiry-in 320ms ease-out both; }
.eiry-empty p { color: var(--muted); margin: 0; font-size: 0.98rem; line-height: 1.5; }
.eiry-row { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; margin-bottom: 18px; }
.eiry-big { display: inline-block; font-size: 4rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em; color: var(--violet); font-variant-numeric: tabular-nums; }
.eiry-unit { display: block; color: var(--muted); font-size: 0.95rem; margin-top: 4px; }
.eiry-gender { text-align: right; }
.eiry-glabel { display: block; font-size: 1.5rem; font-weight: 700; color: var(--ink); }
.eiry-gconf { display: block; color: var(--muted); font-size: 0.9rem; }
.eiry-bar { position: relative; height: 10px; border-radius: 6px; background: var(--track); }
.eiry-band { position: absolute; top: 0; height: 100%; border-radius: 6px; background: linear-gradient(90deg, var(--indigo), var(--violet)); opacity: 0.5; transform-origin: left; animation: eiry-grow 480ms 120ms ease-out both; }
.eiry-mark { position: absolute; top: -5px; width: 20px; height: 20px; margin-left: -10px; border-radius: 50%; background: var(--violet); border: 3px solid var(--card); box-shadow: 0 0 0 2px var(--violet); animation: eiry-pop 360ms 420ms ease-out both; }
.eiry-ticks { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.78rem; margin: 10px 0 0; font-variant-numeric: tabular-nums; }
.eiry-note { color: var(--muted); font-size: 0.9rem; line-height: 1.5; margin: 14px 0 0; }
.eiry-foot { color: var(--muted); font-size: 0.85rem; line-height: 1.55; margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--line); }
.eiry-foot a { color: var(--indigo); text-decoration: none; }
.eiry-foot a:hover { text-decoration: underline; }
button.primary { background: linear-gradient(100deg, var(--indigo), var(--violet)) !important; border: none !important; color: #fff !important; font-weight: 700 !important; font-size: 1rem !important; border-radius: 12px !important; min-height: 48px; transition: filter 180ms ease, transform 120ms ease; }
button.primary:hover { filter: brightness(1.06); }
button.primary:active { transform: scale(0.98); }
button.primary:focus-visible { outline: 3px solid var(--violet); outline-offset: 2px; }
footer { display: none !important; }
@keyframes eiry-in { from { opacity: 0.6; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes eiry-grow { from { transform: scaleX(0.3); } to { transform: scaleX(1); } }
@keyframes eiry-pop { from { transform: scale(0.6); } to { transform: scale(1); } }
@media (prefers-reduced-motion: reduce) { .eiry-card, .eiry-band, .eiry-mark, button.primary { animation: none; transition: none; } }
"""

theme = gr.themes.Base(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "system-ui"],
                       radius_size="lg").set(
    body_background_fill="#F7F5FF", body_background_fill_dark="#0F0E1F",
    block_background_fill="#FFFFFF", block_background_fill_dark="#181633",
    block_border_color="#E4E0F5", block_border_color_dark="#2A2750",
    block_radius="18px",
    input_background_fill="#FFFFFF", input_background_fill_dark="#181633",
    input_border_color="#E4E0F5", input_border_color_dark="#2A2750",
    body_text_color="#1E1B4B", body_text_color_dark="#EEEBFF",
    block_label_text_color="#6B6890", block_label_text_color_dark="#A5A1C8")

with gr.Blocks(title="Eiry") as demo:
    gr.HTML("""<div class="eiry-head">
      <h1 class="eiry-name">Eiry<span aria-hidden="true"></span></h1>
      <p class="eiry-tag">Your voice carries more than you think. Record a few seconds and see how Eiry
      classifies your age and gender from your voice alone!</p></div>""")
    audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your voice",
                     show_label=True, waveform_options={"waveform_color": "#A5B4FC", "waveform_progress_color": "#7C3AED"})
    btn = gr.Button("Estimate", variant="primary")
    out = gr.HTML(empty("Record a few seconds of speech, or upload a clip, then press Estimate."))
    gr.HTML("""<p class="eiry-foot">Age is estimated to within about 8 years on average and is least reliable on short clips
      and on young men, who tend to read older. Ten seconds or more of natural speech works best.
      Model and results: <a href="https://github.com/abdulrahmanbandr/Erie">github.com/abdulrahmanbandr/Erie</a></p>""")
    btn.click(estimate, inputs=audio, outputs=out)
    audio.clear(lambda: empty("Record a few seconds of speech, or upload a clip, then press Estimate."), outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=theme, css=CSS,
                footer_links=[])
