"""
plot_prediction.py  —  visualise fill detection for one recording

Usage (from labeled CSV):
    python plot_prediction.py cup1
    python plot_prediction.py cup3 7

Usage (from raw WAV file — includes spectrogram panel):
    python plot_prediction.py data/clean/cup1/20260602-085822.wav
"""

import matplotlib
matplotlib.use("TkAgg")
import argparse
import sys
from pathlib import Path

import joblib
import librosa
import librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── constants ────────────────────────────────────────────────────────
MODEL_PATH     = "models/best_model.pkl"
FEATURES_PATH  = "data/features_labeled.csv"
ROLLING_WIN    = 10
THRESHOLD      = 0.35
FULL_THRESHOLD = 0.90
FRAME_DURATION = 0.05
HOP_DURATION   = 0.025
N_MFCC         = 13
FMIN           = 1000
FMAX           = 8000


# ── feature extraction (for WAV mode) ────────────────────────────────
def extract_features_from_wav(wav_path: Path):
    """Returns (df, y_active, sr, hop_len, centroid, t_offset)"""
    y, sr = librosa.load(str(wav_path), sr=None)
    hop_len   = int(HOP_DURATION * sr)
    frame_len = int(FRAME_DURATION * sr)

    intervals = librosa.effects.split(y, top_db=40, frame_length=frame_len, hop_length=hop_len)
    start = int(intervals[0][0]) if len(intervals) else 0
    end   = int(intervals[-1][1]) if len(intervals) else len(y)
    y_a   = y[start:end]
    t_off = start / sr

    stft        = np.abs(librosa.stft(y_a, n_fft=frame_len, hop_length=hop_len))
    freqs       = librosa.fft_frequencies(sr=sr, n_fft=frame_len)
    mask        = (freqs >= FMIN) & (freqs <= FMAX)
    stft_b      = stft[mask]; freqs_b = freqs[mask]
    dom_f       = freqs_b[np.argmax(stft_b, axis=0)]

    centroid  = librosa.feature.spectral_centroid(y=y_a, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y_a, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y_a, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    zcr       = librosa.feature.zero_crossing_rate(y_a, frame_length=frame_len, hop_length=hop_len)[0]
    rms       = librosa.feature.rms(y=y_a, frame_length=frame_len, hop_length=hop_len)[0]
    mfccs     = librosa.feature.mfcc(y=y_a, sr=sr, n_mfcc=N_MFCC, n_fft=frame_len,
                                      hop_length=hop_len, fmin=FMIN, fmax=FMAX)
    d_mfcc    = librosa.feature.delta(mfccs)
    d2_mfcc   = librosa.feature.delta(mfccs, order=2)
    d_cent    = librosa.feature.delta(centroid.reshape(1, -1))[0]
    d2_cent   = librosa.feature.delta(centroid.reshape(1, -1), order=2)[0]
    contrast  = librosa.feature.spectral_contrast(y=y_a, sr=sr, n_fft=frame_len, hop_length=hop_len)
    S_b = stft_b ** 2; S_n = S_b / (S_b.sum(axis=0, keepdims=True) + 1e-10)
    mu  = (freqs_b[:, None] * S_n).sum(axis=0)
    sig = np.sqrt(((freqs_b[:, None] - mu[None, :]) ** 2 * S_n).sum(axis=0) + 1e-10)
    skew = ((freqs_b[:, None] - mu[None, :]) ** 3 * S_n).sum(axis=0) / (sig ** 3 + 1e-10)
    kurt = ((freqs_b[:, None] - mu[None, :]) ** 4 * S_n).sum(axis=0) / (sig ** 4 + 1e-10)

    n      = centroid.shape[0]
    times  = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_len) + t_off
    fill   = np.linspace(0, 100, n)  # NOTE: linear approximation only — not real fill level

    rows = []
    for f in range(n):
        row = {"frame": f, "time_sec": round(float(times[f]), 4),
               "fill_level_pct": round(fill[f], 2),
               "t_actual_full": round(float(times[-1]), 4),
               "centroid": centroid[f], "rolloff": rolloff[f],
               "bandwidth": bandwidth[f], "zcr": zcr[f], "rms": rms[f],
               "dominant_freq": dom_f[f] if f < len(dom_f) else np.nan,
               "delta_centroid": d_cent[f], "delta2_centroid": d2_cent[f],
               "spectral_skewness": skew[f] if f < len(skew) else np.nan,
               "spectral_kurtosis": kurt[f] if f < len(kurt) else np.nan}
        for i in range(N_MFCC):
            row[f"mfcc{i+1}"]        = mfccs[i, f]
            row[f"delta_mfcc{i+1}"]  = d_mfcc[i, f]
            row[f"delta2_mfcc{i+1}"] = d2_mfcc[i, f]
        for b in range(contrast.shape[0]):
            row[f"contrast_band{b+1}"] = contrast[b, f] if f < contrast.shape[1] else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    df["file"] = wav_path.stem
    df["cup"]  = wav_path.parent.name
    return df, y_a, sr, hop_len, centroid, t_off


# ── load data: WAV or CSV mode ────────────────────────────────────────
def load_recording(args):
    """Returns (df, wav_extras) where wav_extras is None in CSV mode."""
    if len(args.target) == 1 and args.target[0].endswith(".wav"):
        wav_path = Path(args.target[0])
        if not wav_path.exists():
            print(f"File not found: {wav_path}"); sys.exit(1)
        print(f"WAV mode: {wav_path}")
        df, y_a, sr, hop_len, centroid, t_off = extract_features_from_wav(wav_path)
        return df, {"y": y_a, "sr": sr, "hop_len": hop_len, "centroid": centroid, "t_off": t_off}

    cup   = args.target[0]
    index = int(args.target[1]) if len(args.target) > 1 else 0

    df_all = pd.read_csv(FEATURES_PATH)
    df_all["is_full"] = (df_all["time_sec"] >= df_all["t_actual_full"] * FULL_THRESHOLD).astype(int)
    df_all["group"]   = df_all["cup"] + "_" + df_all["file"]

    available = sorted(df_all["cup"].unique())
    if cup not in available:
        print(f"Cup '{cup}' not found. Available: {available}"); sys.exit(1)

    recordings = sorted(df_all[df_all["cup"] == cup]["file"].unique())
    if index >= len(recordings):
        print(f"Index {index} out of range — '{cup}' has {len(recordings)} recordings (0-{len(recordings)-1}).")
        sys.exit(1)

    rec_file = recordings[index]
    print(f"CSV mode: {rec_file}  ({cup}  #{index} of {len(recordings)})")
    return df_all[(df_all["cup"] == cup) & (df_all["file"] == rec_file)].copy(), None


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="+",
                        help="Either:  cup1  |  cup3 7  |  path/to/file.wav")
    args = parser.parse_args()

    df, wav_extras = load_recording(args)
    model = joblib.load(MODEL_PATH)

    meta_cols    = ["file", "cup", "frame", "time_sec", "t_actual_full",
                    "fill_level_pct", "is_full", "group"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    probs    = model.predict_proba(df[feature_cols])[:, 1]
    smoothed = np.convolve(probs, np.ones(ROLLING_WIN) / ROLLING_WIN,
                           mode="full")[:len(probs)]
    smoothed = np.clip(smoothed, 0, 1)

    times         = df["time_sec"].values
    fill_pct      = df["fill_level_pct"].values
    t_actual_full = df["t_actual_full"].iloc[0]
    t_stop_target = t_actual_full * FULL_THRESHOLD
    t_stop_pred   = next((t for t, p in zip(times, smoothed) if p >= THRESHOLD), None)

    print(f"Duration       : {times[-1]:.2f}s")
    print(f"Should stop at : {t_stop_target:.2f}s  (= {FULL_THRESHOLD*100:.0f}% full)")
    if t_stop_pred:
        dt = t_stop_pred - t_stop_target
        print(f"Model fires at : {t_stop_pred:.2f}s  ({dt:+.2f}s  {'early' if dt < 0 else 'late'})")
    else:
        print("Model never triggered!")

    # ── shared settings ───────────────────────────────────────────────
    wav_mode   = wav_extras is not None
    fill_label = "Fill level (estimated - linear)" if wav_mode else "Fill level (labeled)"
    title      = f"{df['file'].iloc[0]}  ({df['cup'].iloc[0]})"
    run_dir    = Path("predictedExperiment") / f"{df['cup'].iloc[0]}_{df['file'].iloc[0]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def vlines(ax, lw=1.5):
        ax.axvline(t_stop_target, color="orange", linestyle="--", linewidth=lw,
                   label=f"Should stop @ {t_stop_target:.2f}s")
        if t_stop_pred:
            ax.axvline(t_stop_pred, color="green", linestyle="--", linewidth=lw,
                       label=f"Model fires @ {t_stop_pred:.2f}s")
        ax.set_xlabel("Time (seconds)", fontsize=11)

    # ── figure 1: fill level ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.fill_between(times, fill_pct, alpha=0.25, color="steelblue")
    ax1.plot(times, fill_pct, color="steelblue", linewidth=2, label=fill_label)
    ax1.axhline(FULL_THRESHOLD * 100, color="orange", linestyle="--", linewidth=1.5,
                label=f"Stop target ({FULL_THRESHOLD*100:.0f}%)")
    vlines(ax1)
    ax1.set_ylabel("Cup capacity (%)", fontsize=11)
    ax1.set_ylim(0, 108)
    ax1.set_yticks([0, 25, 50, 75, 90, 100])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title(f"Fill Level - {title}" + (" [estimated]" if wav_mode else ""), fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(run_dir / "01_fill_level.png", dpi=150)
    plt.close(fig1)
    print(f"  Saved -> {run_dir / '01_fill_level.png'}")

    # ── figure 2: model probability ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(times, probs,    alpha=0.3, color="steelblue", label="Raw probability")
    ax2.plot(times, smoothed, color="steelblue", linewidth=2,
             label=f"Smoothed (window={ROLLING_WIN})")
    ax2.axhline(THRESHOLD, color="red", linestyle=":", linewidth=1.5,
                label=f"Threshold ({THRESHOLD})")
    vlines(ax2)
    ax2.set_ylabel("P(cup is full)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_title(f"Model Probability - {title}", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(run_dir / "02_probability.png", dpi=150)
    plt.close(fig2)
    print(f"  Saved -> {run_dir / '02_probability.png'}")

    # ── figure 3: spectrogram + centroid ──────────────────────────────
    centroid_vals   = wav_extras["centroid"] if wav_mode else df["centroid"].values
    centroid_smooth = np.convolve(centroid_vals, np.ones(10) / 10, mode="same")
    cent_times      = times[:len(centroid_vals)]

    fig3, ax3 = plt.subplots(figsize=(11, 5))

    # try to load audio — WAV extras or locate the file in data/clean/
    if wav_mode:
        y_spec, sr_spec = wav_extras["y"], wav_extras["sr"]
        hop_spec        = wav_extras["hop_len"]
        off_spec        = wav_extras["t_off"]
    else:
        wav_path = Path("data/clean") / df["cup"].iloc[0] / (df["file"].iloc[0] + ".wav")
        if wav_path.exists():
            y_spec, sr_spec = librosa.load(str(wav_path), sr=None)
            hop_spec = int(HOP_DURATION * sr_spec)
            intervals = librosa.effects.split(y_spec, top_db=40,
                                              frame_length=int(FRAME_DURATION * sr_spec),
                                              hop_length=hop_spec)
            start    = int(intervals[0][0]) if len(intervals) else 0
            y_spec   = y_spec[start:]
            off_spec = start / sr_spec
        else:
            y_spec = None

    if y_spec is not None:
        frame_len_spec = int(FRAME_DURATION * sr_spec)
        D   = librosa.amplitude_to_db(
                np.abs(librosa.stft(y_spec, n_fft=frame_len_spec, hop_length=hop_spec)),
                ref=np.max)
        img = librosa.display.specshow(D, sr=sr_spec, hop_length=hop_spec,
                                       x_axis="time", y_axis="log", ax=ax3, cmap="magma",
                                       x_coords=librosa.frames_to_time(
                                           np.arange(D.shape[1]), sr=sr_spec,
                                           hop_length=hop_spec) + off_spec)
        fig3.colorbar(img, ax=ax3, format="%+2.0f dB")
    else:
        ax3.set_facecolor("#1a1a2e")

    cent_color = "cyan"

    ax3.plot(cent_times, centroid_smooth, color=cent_color, linewidth=2.5,
             label="Spectral centroid (smoothed)")
    vlines(ax3, lw=2)
    ax3.set_xlim(times[0], times[-1])
    y_max = max(centroid_smooth.max() * 1.15, 8000)   # at least 8kHz, 15% headroom above centroid
    ax3.set_ylim(200, y_max)
    ax3.set_ylabel("Frequency (Hz)", fontsize=11)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.set_title(f"Spectrogram + Spectral Centroid - {title}", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(run_dir / "03_spectrogram_centroid.png", dpi=150)
    plt.close(fig3)
    print(f"  Saved -> {run_dir / '03_spectrogram_centroid.png'}")

    print(f"Done - folder: {run_dir}")


if __name__ == "__main__":
    main()
