import numpy as np
import pandas as pd
import librosa
from pathlib import Path


FRAME_DURATION = 0.05    # 50ms per frame
HOP_DURATION   = 0.025   # 25ms hop (50% overlap)
N_MFCC         = 13
FMIN           = 1000    # Hz — lower bound of water resonance band
FMAX           = 8000    # Hz — upper bound of water resonance band


# ------------------------------------------------------------------
# Improvement 1: Dynamic Edge Trimming using librosa.effects.split
# ------------------------------------------------------------------
def detect_active_region(y: np.ndarray, sr: int, hop_len: int):
    """
    Use librosa.effects.split to find the exact pour start/end.
    Returns (start_sample, end_sample).
    Falls back to full signal if no split found.
    """
    intervals = librosa.effects.split(y, top_db=40, frame_length=int(FRAME_DURATION * sr), hop_length=hop_len)
    if len(intervals) == 0:
        return 0, len(y)
    # take from first active interval start to last active interval end
    return int(intervals[0][0]), int(intervals[-1][1])


def extract_frame_features(wav_path: Path) -> pd.DataFrame:
    y, sr = librosa.load(str(wav_path), sr=None)

    hop_len   = int(HOP_DURATION * sr)
    frame_len = int(FRAME_DURATION * sr)

    # --- Improvement 1: trim to active pour region ---
    start_sample, end_sample = detect_active_region(y, sr, hop_len)
    y_active = y[start_sample:end_sample]
    t_start  = start_sample / sr
    t_end    = end_sample / sr

    # --- Improvement 2A: frequency-bounded STFT (1000–8000 Hz) ---
    stft  = np.abs(librosa.stft(y_active, n_fft=frame_len, hop_length=hop_len))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_len)
    freq_mask  = (freqs >= FMIN) & (freqs <= FMAX)
    stft_bound = stft[freq_mask, :]
    freqs_bound = freqs[freq_mask]
    dominant_f  = freqs_bound[np.argmax(stft_bound, axis=0)]

    # --- base features (on trimmed signal) ---
    centroid  = librosa.feature.spectral_centroid(y=y_active, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y_active, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y_active, sr=sr, n_fft=frame_len, hop_length=hop_len)[0]
    zcr       = librosa.feature.zero_crossing_rate(y_active, frame_length=frame_len, hop_length=hop_len)[0]
    rms       = librosa.feature.rms(y=y_active, frame_length=frame_len, hop_length=hop_len)[0]
    mfccs     = librosa.feature.mfcc(y=y_active, sr=sr, n_mfcc=N_MFCC,
                                      n_fft=frame_len, hop_length=hop_len,
                                      fmin=FMIN, fmax=FMAX)

    # --- Improvement 2B: delta & delta-delta for MFCCs and centroid ---
    delta_mfccs    = librosa.feature.delta(mfccs)
    delta2_mfccs   = librosa.feature.delta(mfccs, order=2)
    delta_centroid  = librosa.feature.delta(centroid.reshape(1, -1))[0]
    delta2_centroid = librosa.feature.delta(centroid.reshape(1, -1), order=2)[0]

    # --- spectral shape features ---
    contrast = librosa.feature.spectral_contrast(y=y_active, sr=sr,
                                                   n_fft=frame_len, hop_length=hop_len)
    S_bound  = stft_bound ** 2
    S_norm   = S_bound / (S_bound.sum(axis=0, keepdims=True) + 1e-10)
    mu       = (freqs_bound[:, None] * S_norm).sum(axis=0)
    sigma2   = ((freqs_bound[:, None] - mu[None, :]) ** 2 * S_norm).sum(axis=0)
    sigma    = np.sqrt(sigma2 + 1e-10)
    skewness = ((freqs_bound[:, None] - mu[None, :]) ** 3 * S_norm).sum(axis=0) / (sigma ** 3 + 1e-10)
    kurtosis = ((freqs_bound[:, None] - mu[None, :]) ** 4 * S_norm).sum(axis=0) / (sigma ** 4 + 1e-10)

    n_frames    = centroid.shape[0]
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_len) + t_start

    # --- Improvement 1: linear label strictly within trimmed region ---
    fill_level = np.linspace(0.0, 100.0, n_frames)

    # t_actual_full = last active frame time (assumed = 95% full per project assumption)
    t_actual_full = float(frame_times[-1])

    rows = []
    for f in range(n_frames):
        row = {
            "file":             wav_path.stem,
            "cup":              wav_path.parent.name,
            "frame":            f,
            "time_sec":         round(frame_times[f], 4),
            "t_actual_full":    round(t_actual_full, 4),
            "fill_level_pct":   round(fill_level[f], 2),
            "centroid":         centroid[f],
            "rolloff":          rolloff[f],
            "bandwidth":        bandwidth[f],
            "zcr":              zcr[f],
            "rms":              rms[f],
            "dominant_freq":    dominant_f[f] if f < len(dominant_f) else np.nan,
            "delta_centroid":   delta_centroid[f],
            "delta2_centroid":  delta2_centroid[f],
            "spectral_skewness": skewness[f] if f < len(skewness) else np.nan,
            "spectral_kurtosis": kurtosis[f] if f < len(kurtosis) else np.nan,
        }
        for i in range(N_MFCC):
            row[f"mfcc{i+1}"]        = mfccs[i, f]
            row[f"delta_mfcc{i+1}"]  = delta_mfccs[i, f]
            row[f"delta2_mfcc{i+1}"] = delta2_mfccs[i, f]
        for b in range(contrast.shape[0]):
            row[f"contrast_band{b+1}"] = contrast[b, f] if f < contrast.shape[1] else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    clean_dir = Path("data/clean")
    wav_files = [f for f in clean_dir.rglob("*.wav") if "test" not in f.stem]

    if not wav_files:
        print("No WAV files found in data/clean/ — run whitenoise.py first.")
        return

    print(f"Processing {len(wav_files)} files (frame-level extraction)...")

    all_frames = []
    for i, path in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] {path.relative_to(clean_dir)}", end=" ... ", flush=True)
        try:
            df = extract_frame_features(path)
            all_frames.append(df)
            print(f"done ({len(df)} frames, {df['time_sec'].iloc[-1] - df['time_sec'].iloc[0]:.1f}s active)")
        except Exception as e:
            print(f"ERROR: {e}")

    result = pd.concat(all_frames, ignore_index=True)
    out = Path("data/features_labeled.csv")
    result.to_csv(out, index=False)
    print(f"\nSaved {len(result)} frames x {len(result.columns)} columns → {out.resolve()}")
    print(result[["cup", "file", "frame", "time_sec", "fill_level_pct", "t_actual_full"]].head(10))


if __name__ == "__main__":
    main()
