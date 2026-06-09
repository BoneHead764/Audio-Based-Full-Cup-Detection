import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import sys


def trim_to_pour(y, sr, frame_len, hop_len):
    """Trim signal to the actual pour region using RMS energy."""
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    noise_floor = np.sort(rms)[:max(1, len(rms) // 10)].mean()
    threshold   = noise_floor * 4
    active      = np.where(rms > threshold)[0]
    if len(active) == 0:
        return y, 0.0
    start_sample = int(librosa.frames_to_samples(active[0],  hop_length=hop_len))
    end_sample   = int(librosa.frames_to_samples(active[-1], hop_length=hop_len))
    return y[start_sample:end_sample], start_sample / sr


def plot_spectrogram(wav_path: str):
    wav_path = Path(wav_path)
    y, sr = librosa.load(str(wav_path), sr=None)

    frame_len = int(0.05 * sr)
    hop_len   = int(0.025 * sr)

    y_active, t_offset = trim_to_pour(y, sr, frame_len, hop_len)
    # cut last 5% of samples — always the quietest tail where centroid drops
    y_active = y_active

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"Spectrogram Analysis — {wav_path.name}", fontsize=13)

    # --- 1. Mel spectrogram ---
    S_mel = librosa.feature.melspectrogram(y=y_active, sr=sr, n_mels=128, hop_length=hop_len)
    S_db  = librosa.power_to_db(S_mel, ref=np.max)
    img   = librosa.display.specshow(S_db, sr=sr, hop_length=hop_len,
                                     x_axis="time", y_axis="mel", ax=axes[0])
    fig.colorbar(img, ax=axes[0], format="%+2.0f dB")
    axes[0].set_title("Mel Spectrogram — frequency content over time")

    # --- 2. STFT + centroid ---
    centroid  = librosa.feature.spectral_centroid(y=y_active, sr=sr,
                                                   n_fft=frame_len, hop_length=hop_len)[0]
    times     = librosa.times_like(centroid, sr=sr, hop_length=hop_len) + t_offset
    S_stft    = np.abs(librosa.stft(y_active, n_fft=frame_len, hop_length=hop_len))
    S_stft_db = librosa.amplitude_to_db(S_stft, ref=np.max)
    librosa.display.specshow(S_stft_db, sr=sr, hop_length=hop_len,
                             x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].plot(times, centroid, color="yellow", linewidth=2, label="Spectral centroid")
    axes[1].set_ylim(0, sr // 2)
    axes[1].set_title("STFT Spectrogram + Spectral Centroid (yellow) — centroid should rise as cup fills")
    axes[1].legend(loc="upper right")

    # --- 3. Centroid vs fill level ---
    window          = 20
    # use "valid" convolution to avoid edge artifacts, then align times
    centroid_smooth = np.convolve(centroid, np.ones(window) / window, mode="valid")
    pad             = (len(centroid) - len(centroid_smooth)) // 2
    times_smooth    = times[pad: pad + len(centroid_smooth)]
    fill_pct        = np.linspace(0, 100, len(times))

    ax3 = axes[2]
    ax3.plot(times, centroid,                    alpha=0.3, color="steelblue", label="Raw centroid")
    ax3.plot(times_smooth, centroid_smooth, color="steelblue", linewidth=2, label="Smoothed centroid")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Frequency (Hz)", color="steelblue")
    ax3.tick_params(axis="y", labelcolor="steelblue")

    ax3b = ax3.twinx()
    ax3b.plot(times, fill_pct, color="red", linestyle="--", linewidth=1.5, label="Fill level (linear)")
    ax3b.set_ylabel("Fill Level (%)", color="red")
    ax3b.tick_params(axis="y", labelcolor="red")
    ax3b.set_ylim(0, 100)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax3.set_title("Spectral Centroid vs Fill Level — if correlated, acoustic change is real (not just time)")

    plt.tight_layout()
    out = wav_path.with_name(wav_path.stem + "_spectrogram.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_spectrogram.py <path_to_wav>")
        print("Example: python plot_spectrogram.py data\\clean\\cup1\\20260602-085822.wav")
    else:
        plot_spectrogram(sys.argv[1])
