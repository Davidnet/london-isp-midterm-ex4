"""Fig 2: raw vs preprocessed audio for the report.

Reproduces the notebook's preprocess() chain character-for-character
(order-4 Butterworth high-pass at 80 Hz via zero-phase sosfiltfilt,
RMS normalization to -20 dBFS with a 0.97 peak ceiling, and a 0.8 s
deterministic low-amplitude lead-in pad) on the author's own laptop
recording, then renders waveform + spectrogram panels before/after.

Run:
  uv run --no-project --with numpy --with scipy --with matplotlib \
      python report/tools/make_preprocess_fig.py
"""

import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt, spectrogram

REPO = Path(__file__).resolve().parents[2]
WAV = REPO / "Ex4_audio_files" / "OWN" / "own_sentence1.wav"
OUT = REPO / "report" / "figures" / "fig2-preprocess.png"


# --- exact copies of the notebook functions -------------------------------
def highpass(samples, sample_rate, cutoff_hz=80.0, order=4):
    sos = butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    padlen = min(3 * (sos.shape[0] * 2), len(samples) - 1)
    return sosfiltfilt(sos, samples, padlen=max(padlen, 0)).astype(np.float32)


def rms_normalize(samples, target_dbfs=-20.0, peak_ceiling=0.97):
    rms = np.sqrt(np.mean(samples**2)) + 1e-9
    out = samples * (10.0 ** (target_dbfs / 20.0) / rms)
    peak = np.max(np.abs(out)) + 1e-9
    if peak > peak_ceiling:
        out = out * (peak_ceiling / peak)
    return out.astype(np.float32)


def preprocess(samples, sample_rate, cutoff_hz=80.0, target_dbfs=-20.0, lead_sec=0.8):
    x = highpass(samples, sample_rate, cutoff_hz)
    x = rms_normalize(x, target_dbfs)
    pad = (np.random.RandomState(0).randn(int(lead_sec * sample_rate)) * 1e-4).astype(
        np.float32
    )
    return np.concatenate([pad, x])


def read_wave(path):
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "%s: must be mono" % path
        assert wf.getsampwidth() == 2, "%s: must be 16-bit PCM" % path
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


# --- figure ----------------------------------------------------------------
def spec_panel(ax, x, rate, title):
    f, t, sxx = spectrogram(x, fs=rate, nperseg=1024, noverlap=768)
    db = 10.0 * np.log10(sxx + 1e-12)
    m = ax.pcolormesh(t, f, db, shading="gouraud", cmap="magma",
                      vmin=db.max() - 80, vmax=db.max())
    ax.set_ylim(0, 1500)
    ax.axhline(80.0, color="w", ls="--", lw=1.0)
    ax.text(0.02, 92, "80 Hz cutoff", color="w", fontsize=8, va="bottom")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")
    return m


def wave_panel(ax, x, rate, title, lead_sec=None):
    t = np.arange(len(x)) / rate
    ax.plot(t, x, lw=0.4, color="#1a4f8b")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("amplitude")
    ax.set_xlabel("s")
    if lead_sec:
        ax.axvspan(0, lead_sec, color="0.75", alpha=0.35)
        ax.text(lead_sec / 2, 0.78, "0.8 s\nlead-in", ha="center",
                fontsize=8, color="0.25")


def main():
    raw, rate = read_wave(WAV)
    clean = preprocess(raw, rate)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 4.6))
    wave_panel(axes[0, 0], raw, rate, "Raw waveform (own_sentence1.wav)")
    wave_panel(axes[0, 1], clean, rate,
               "Preprocessed (HPF 80 Hz + RMS → −20 dBFS)",
               lead_sec=0.8)
    spec_panel(axes[1, 0], raw, rate, "Raw spectrogram")
    spec_panel(axes[1, 1], clean, rate, "Preprocessed spectrogram")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220)
    print("wrote", OUT)

    # report numbers used in prose sanity checks
    def dbfs(x):
        return 20.0 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)

    print("raw RMS dBFS  = %.1f" % dbfs(raw))
    print("clean RMS dBFS= %.1f (incl. pad)" % dbfs(clean))


if __name__ == "__main__":
    main()
