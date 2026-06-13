# 🥤 Acoustic Cup-Fill Detection

> Detect the exact moment a cup is full — using only sound.

An automated machine learning system that determines the optimal moment to stop pouring water into a cup using **passive acoustic sensing** only — no cameras, no weight sensors, just a microphone.

---

## 📖 Overview

When water is poured into a cup, the acoustic resonance of the vessel changes as the water level rises. This project captures those changes using a microphone, extracts audio features, and trains a classifier to distinguish between "not full" and "nearly full" states.

Submitted as a Machine Learning course project at **Sami Shamoon College of Engineering (SCE)**, Department of Electrical and Electronics Engineering.

**Lecturer:** Dmitry Bakhovsky  
**Authors:** Tomer Nur Milkov, Nikita Budovski, Oren, Or

---

## ⚙️ How It Works

The system operates through a sequential pipeline:

1. **Record & Clean** — Capture liquid-pouring audio and apply non-stationary noise profiling to strip out ambient faucet/pump hum.
2. **Trim & Frame** — Isolate the true boundaries of the pour event automatically, slicing the signal into 50 ms frames with 50% overlap.
3. **Extract Transient Features** — Map frequencies within the 1000–8000 Hz band to compute 13 MFCCs alongside their velocity (Δ) and acceleration (Δ²) derivatives.
4. **Train & Tune Gradient Boosters** — Train a LightGBM binary classifier via optimized parameter search using GroupKFold cross-validation to recognize the final ≥ 90% capacity point.
5. **Real-Time Guardrails** — Run model probability projections through a 10-frame causal moving average to prevent sudden splash artifacts from triggering early faucet closures.

---

## 🗂️ Dataset

- **120 recordings** across 8 cup types
- **Cup materials:** Glass, Plastic, Cardboard
- **~15 recordings per cup** for balanced representation
- **1,495 samples** total after windowing

Recordings were made under consistent conditions: same room, same tap, fixed microphone distance, and a quiet environment.

---

## 🎛️ Features Extracted

For each 50 ms audio frame, feature extraction is bounded to the primary liquid resonance passband (1000–8000 Hz):

| Feature | Description |
|---|---|
| **MFCC (×13)** | Mel-frequency cepstral coefficients — capture timbral/spectral shape |
| **MFCC Δ / Δ² (×26)** | First- and second-order temporal derivatives of MFCCs |
| **RMS Energy** | Loudness / signal power |
| **Zero Crossing Rate** | How often the signal crosses zero — relates to frequency content |
| **Spectral Centroid + Δ/Δ²** | "Center of gravity" of the spectrum and its rate of change |
| **Spectral Bandwidth** | Spread of energy around the centroid |
| **Spectral Rolloff** | Frequency below which 85% of energy falls |
| **Spectral Contrast (×7 bands)** | Peak-to-valley difference per sub-band |
| **Spectral Skewness / Kurtosis** | Shape statistics of the bounded power spectrum |
| **Dominant Frequency** | Frequency bin with peak energy in the bounded STFT |

**Total: 56 features per sample**

---

## 🧠 Models Evaluated

All models used **GroupKFold cross-validation** (5 folds), grouping by recording file to prevent window leakage between train and test sets.

| Model | Precision | Recall | F1-Score | AUC-ROC | Avg Latency (Δt) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **72.10%** | **75.11%** | **0.735** | **0.968** | **−1.00 s** |
| XGBoost | 69.14% | 77.79% | 0.731 | 0.963 | −1.74 s |
| Gradient Boosting | 76.00% | 67.35% | 0.712 | 0.967 | −0.69 s |
| Random Forest | 67.81% | 73.06% | 0.703 | 0.947 | −1.90 s |
| Extra Trees | 68.34% | 72.37% | 0.702 | 0.949 | −1.76 s |
| HistGradientBoosting | 60.92% | 80.49% | 0.692 | 0.955 | −2.19 s |
| KNN | 73.62% | 61.09% | 0.666 | 0.947 | −0.41 s |
| SVC | 68.07% | 65.04% | 0.664 | 0.957 | −2.59 s |

*Δt is measured relative to the true full point — negative values indicate early triggering, positive values late. LightGBM achieves the best overall F1 and AUC.*

---

## 🔍 Feature Ablation & Importance

Ablation analysis (`models/ablation_results.csv`) confirmed that modeling the **temporal velocity** of the acoustic change is crucial:

| Feature Set | Features | F1 | AUC-ROC |
|---|:---:|:---:|:---:|
| All Features | 56 | **0.868** | **0.961** |
| No Delta / Delta-Delta | 28 | 0.585 | 0.932 |
| MFCC Baseline Only | 19 | 0.573 | 0.911 |

Removing the delta features causes a **28% drop in F1**, proving the classifier is tracking the *dynamic path* of the Helmholtz resonance shift as the empty volume of the cup contracts — not just static spectral values.

Feature importance charts are in `images/feature_importance_LightGBM.png` and `images/feature_importance_Gradient_Boosting.png`.

---

## 📂 Project Structure

```text
.
├── data/
│   ├── raw/                    # Original .wav recordings (organized by cup)
│   ├── clean/                  # Noise-reduced .wav files
│   ├── processed/
│   │   └── features.csv        # Windowed baseline dataset
│   └── features_labeled.csv    # Final frame-level classification dataset
│
├── src/
│   ├── whitenoise.py           # Step 1: Non-stationary noise reduction
│   ├── label_and_extract.py    # Step 2: MFCC + delta feature extraction
│   ├── tune_models.py          # Step 3: Model training & hyperparameter tuning
│   ├── plot_prediction.py      # Step 4: Real-time probability dashboards
│   └── plot_spectrogram.py     # Step 5: Helmholtz resonance visualization
│
├── models/
│   ├── best_model.pkl          # Saved production LightGBM pipeline
│   ├── results_summary.csv     # Cross-validation performance comparisons
│   ├── results_extended.csv    # Full metric breakdown per model
│   └── ablation_results.csv    # Feature ablation analysis
│
├── images/
│   ├── 01_fill_level.png
│   ├── 02_probability.png
│   ├── 03_spectrogram_centroid.png
│   ├── model_comparison.png
│   ├── feature_importance_LightGBM.png
│   └── feature_importance_Gradient_Boosting.png
│
├── docs/
│   ├── Cup_Fill_Detection.pptx     # Project presentation slides
│   └── [Hebrew project report].pdf # Final project report (Hebrew)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Usage

### Requirements

Python 3.10+ is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Running the Pipeline

Place your raw `.wav` recordings inside `data/raw/` organized into per-cup subfolders, then run the scripts in order from the project root:

**Step 1 — Clean Background Noise**
```bash
python src/whitenoise.py
```

**Step 2 — Extract & Label Features**
```bash
python src/label_and_extract.py
```

**Step 3 — Train & Tune Models**
```bash
python src/tune_models.py
```

**Step 4 — Generate Prediction Dashboards**
```bash
python src/plot_prediction.py
```

**Step 5 — Visualize Acoustic Resonance**
```bash
python src/plot_spectrogram.py
```

---

## 📊 Key Findings

- **Acoustic-only detection works.** LightGBM achieves an F1 of 0.735 and AUC-ROC of 0.968 using only a microphone — no camera, no scale.
- **Spectral dynamics are the real signal.** Delta and delta-delta MFCC features account for a 28% improvement in F1 over static features alone, confirming the model is tracking the *rate of change* in Helmholtz resonance, not just its absolute value.
- **Smoothing is essential.** A 10-frame causal rolling average prevents sudden splashing sounds from triggering false-positive stops.
- **GroupKFold prevents data leakage.** Without grouping by recording, windows from the same file appear in both train and test splits, artificially inflating scores.

---

## 📈 Visualizations

![Model Comparison](images/model_comparison.png)
![Fill Level Tracking](images/01_fill_level.png)
![Prediction Probability](images/02_probability.png)
![Spectrogram & Spectral Centroid](images/03_spectrogram_centroid.png)

---

## 🔮 Future Improvements

- Larger, more diverse dataset (more cup types, materials, and sizes)
- Data augmentation with background noise to improve robustness in noisy environments
- Real-time streaming implementation for embedded hardware (e.g., Raspberry Pi)
- Online adaptation to new cup types using few-shot learning
