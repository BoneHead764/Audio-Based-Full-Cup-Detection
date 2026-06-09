import noisereduce as nr
import numpy as np
from scipy.io import wavfile
from pathlib import Path


def reduce_noise(input_path: Path, output_path: Path, noise_clip_duration: float = 0.5):
    rate, data = wavfile.read(input_path)

    noise_samples = int(rate * noise_clip_duration)
    noise_clip = data[:noise_samples]

    if data.ndim == 2:
        reduced = np.stack([
            nr.reduce_noise(y=data[:, ch].astype(np.float32), sr=rate, y_noise=noise_clip[:, ch].astype(np.float32))
            for ch in range(data.shape[1])
        ], axis=1)
    else:
        reduced = nr.reduce_noise(y=data.astype(np.float32), sr=rate, y_noise=noise_clip.astype(np.float32))

    dtype = data.dtype
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
    reduced = np.clip(reduced, info.min, info.max).astype(dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, rate, reduced)


def main():
    raw_dir = Path("data/raw")
    clean_dir = Path("data/clean")

    wav_files = list(raw_dir.rglob("*.wav"))
    if not wav_files:
        print("No WAV files found in data/raw/")
        return

    print(f"Found {len(wav_files)} files. Processing...")

    for i, input_path in enumerate(wav_files, 1):
        relative = input_path.relative_to(raw_dir)
        output_path = clean_dir / relative
        print(f"[{i}/{len(wav_files)}] {relative}", end=" ... ", flush=True)
        try:
            reduce_noise(input_path, output_path)
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nFinished. Clean files saved to: {clean_dir.resolve()}")


if __name__ == "__main__":
    main()
