import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

WINDOWS_CSV = "data/processed/windows_index.csv"
OUTPUT_CSV = "data/processed/features.csv"

SAMPLE_RATE = 22050


# =========================================================
# פילטר Band-Pass
# שומר את תחום התדרים החשוב ומקטין רעשי ברז/סביבה
# =========================================================
def bandpass_filter(y, sr, low=200, high=5000, order=4):
    nyquist = sr / 2

    low_norm = low / nyquist
    high_norm = high / nyquist

    b, a = butter(order, [low_norm, high_norm], btype="band")
    y_filtered = filtfilt(b, a, y)

    return y_filtered


# =========================================================
# חישוב 1/f slope
# בודק איך האנרגיה יורדת עם התדר
# =========================================================
def spectral_slope_1overf(y, sr):
    spectrum = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)

    mask = freqs > 0
    freqs = freqs[mask]
    spectrum = spectrum[mask]

    spectrum = spectrum + 1e-12

    log_f = np.log(freqs)
    log_p = np.log(spectrum)

    slope, intercept = np.polyfit(log_f, log_p, 1)

    return slope


# =========================================================
# חילוץ מאפיינים מחלון אודיו אחד
# =========================================================
def extract_features_from_window(y_window, sr):
    features = {}

    # ---------- מאפייני אנרגיה ----------
    features["rms"] = float(np.mean(librosa.feature.rms(y=y_window)))

    # ---------- Zero Crossing ----------
    features["zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(y_window)))

    # ---------- מאפיינים ספקטרליים ----------
    features["spectral_centroid"] = float(np.mean(
        librosa.feature.spectral_centroid(y=y_window, sr=sr)
    ))

    features["spectral_bandwidth"] = float(np.mean(
        librosa.feature.spectral_bandwidth(y=y_window, sr=sr)
    ))

    features["spectral_rolloff"] = float(np.mean(
        librosa.feature.spectral_rolloff(y=y_window, sr=sr)
    ))

    features["spectral_flatness"] = float(np.mean(
        librosa.feature.spectral_flatness(y=y_window)
    ))

    # ---------- 1/f slope ----------
    features["slope_1overf"] = float(spectral_slope_1overf(y_window, sr))

    # ---------- MFCC ----------
    mfcc = librosa.feature.mfcc(y=y_window, sr=sr, n_mfcc=13)

    for i in range(13):
        features[f"mfcc_{i+1}"] = float(np.mean(mfcc[i]))

    return features


# =========================================================
# טעינת אודיו מלא
# =========================================================
def load_audio(path, sr=SAMPLE_RATE):
    y, sr = librosa.load(path, sr=sr, mono=True)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y, sr


# =========================================================
# בניית features.csv
# =========================================================
def build_features():
    windows_df = pd.read_csv(WINDOWS_CSV)

    all_rows = []

    current_audio_path = None
    current_audio = None
    current_sr = None

    for idx, row in windows_df.iterrows():

        audio_path = row["audio_path"]

        # -------------------------------------------------
        # טעינה מחדש רק כאשר עוברים לקובץ אודיו חדש
        # זה חוסך זמן ריצה
        # -------------------------------------------------
        if audio_path != current_audio_path:
            current_audio_path = audio_path
            current_audio, current_sr = load_audio(audio_path)

            # פילטר אחרי טעינה
            current_audio = bandpass_filter(current_audio, current_sr)

            print(f"Loaded and filtered: {audio_path}")

        start = int(row["start_sample"])
        end = int(row["end_sample"])

        y_window = current_audio[start:end]

        if len(y_window) == 0:
            continue

        feats = extract_features_from_window(y_window, current_sr)

        # מוסיפים metadata ו-label
        feats["audio_path"] = row["audio_path"]
        feats["cup_id"] = row["cup_id"]
        feats["take_id"] = row["take_id"]
        feats["window_id"] = row["window_id"]
        feats["start_time"] = row["start_time"]
        feats["end_time"] = row["end_time"]
        feats["fill_percent"] = row["fill_percent"]

        all_rows.append(feats)

    features_df = pd.DataFrame(all_rows)
    features_df.to_csv(OUTPUT_CSV, index=False)

    print("======================================")
    print(f"Saved features to: {OUTPUT_CSV}")
    print(f"Total feature rows: {len(features_df)}")
    print(f"Columns: {len(features_df.columns)}")
    print("======================================")


if __name__ == "__main__":
    build_features()