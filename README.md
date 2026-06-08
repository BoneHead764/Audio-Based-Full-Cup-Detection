# 🥤 Acoustic Cup-Fill Detection

An automated machine learning system capable of determining the optimal moment to stop pouring water into a cup using only passive acoustic sensing (audio signals) — no cameras, no weight sensors, just sound.

---

## 📖 Overview

When water is poured into a cup, the acoustic resonance of the vessel changes as the water level rises. This project captures those changes using a microphone, extracts audio features, and trains a classifier to distinguish between "not full" and "nearly full" states.

Submitted as a Machine Learning course project at **Sami Shamoon College of Engineering (SCE)**, Department of Electrical and Electronics Engineering.

**Lecturer:** Dmitry Bakhovsky  
**Authors:** Tomer Nur Milkov, Nikita Budovski, Oren, Or

---

## ⚙️ How It Works

Our system operates through a sequential pipeline:
1. **Record** water being poured into a cup (empty → full).
2. **Slice** each recording into 0.5-second time windows.
3. **Extract** acoustic features (MFCCs, Spectral Centroid, etc.) from every window.
4. **Train** a classifier using GroupKFold cross-validation to predict the fill percentage.
5. **Real-Time Logic:** Apply a moving average and enforce physical monotonicity to trigger a reliable "stop" command when the cup hits 90% capacity.

---

## 🗂️ Dataset

- **120 recordings** across 8 cup types.
- **Cup materials:** Glass, Plastic, Cardboard.
- **~15 recordings per cup type** for balanced representation.
- **1,495 samples** total after windowing.

Recordings were made under consistent conditions: same room, same tap, fixed microphone distance, and a quiet environment.

---

## 🎛️ Features Extracted

For each 0.5-second audio window, the following features are computed (mean + std = 2 values each):

| Feature | Description |
|---|---|
| **MFCC (×13)** | Mel-frequency cepstral coefficients — capture timbral/spectral shape |
| **RMS Energy** | Loudness / signal power |
| **Zero Crossing Rate** | How often the signal crosses zero — relates to frequency content |
| **Spectral Centroid** | "Center of gravity" of the frequency spectrum |
| **Spectral Bandwidth** | Spread of energy around the centroid |
| **Spectral Rolloff** | Frequency below which 85% of energy falls |

**Total: 32 features per sample**

---

## 🧠 Models Evaluated

All models were evaluated with **GroupKFold Cross-Validation**, grouping by recording to prevent data leakage between windows from the same file.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 95.85% | 77.71% | 53.37% | 62.76% |
| k-NN (k=5) | 96.12% | 75.29% | 62.58% | 68.07% |
| **Random Forest** | **96.32%** | **83.93%** | 56.47% | **67.18%** |

**Random Forest** was selected as the final baseline model for its high accuracy and precision, though advanced tests with LightGBM have pushed F1 scores even higher (F1 = 0.868, AUC-ROC = 0.961).

### Confusion Matrix (Random Forest)

| | Predicted: Not Full | Predicted: Full |
|---|---|---|
| **Actual: Not Full** | 1384 ✅ | 12 ❌ |
| **Actual: Full** | 43 ❌ | 56 ✅ |

---

## 🔍 Top Features (Feature Importance)

The model revealed that **MFCC features dominate** the decision — particularly their standard deviation, which captures how much the spectral shape varies within each window.

1. `mfcc_0_std` (~0.083)
2. `mfcc_2_mean` (~0.060)
3. `mfcc_5_mean` (~0.057)
4. `mfcc_1_std` (~0.053)
5. `bandwidth_mean` (~0.051)

This confirms the model is detecting **resonance changes** (frequency structure), perfectly aligning with Helmholtz resonance theory, rather than just listening for volume.

---

## 📂 Project Structure

```text
.
├── data/
│   ├── raw/                 # Put your raw .wav recordings here 
│   └── processed/           # Generated windows_index.csv and features.csv
│
├── src/                     # Python pipeline scripts
│   ├── build_dataset.py     # Step 1: Slices audio into windows
│   ├── features.py          # Step 2: Extracts MFCCs, Rolloff, etc.
│   ├── train.py             # Step 3: Trains models & calculates metrics
│   ├── realtime_decision.py # Step 4: Applies moving average and 90% stop logic
│   └── plots.py             # Step 5: Generates graphs for reports
│
├── results/                 # Output predictions, logs, and metrics
│   └── figures/             # Auto-generated graphs (Confusion Matrix, RMSE, etc.)
│
├── docs/                    # Project reports and documentation
│   ├── lab_report.docx      # English Laboratory Report
│   ├── ML_Report_Hebrew.pdf # Final Project Report (Hebrew)
│   └── documantation.docx   # Internal engineering logic and graph explanations
│
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
---

## 🚀 Setup & Usage

### Requirements
Ensure you have Python 3.10+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```
### Running the Pipeline
Place your raw `.wav` recordings inside `data/raw/` (organized by cup folders), then run the scripts from the root directory in sequential order:

**1. Build the dataset index**
```bash
python src/build_dataset.py
```

**2. Extract acoustic features**
```bash
python src/features.py
```

**3. Train models and generate predictions**
```bash
python src/train.py
```

**4. Apply smoothing and real-time stop logic**
```bash
python src/realtime_decision.py
```
**5. Generate visual evaluation charts**
```bash
python src/plots.py
```

---

## 📊 Key Findings

- **Acoustic-only detection works.** A classifier can accurately trigger a stop command with an average latency of just +0.08 to +0.10 seconds without visual or weight sensors.
- **Spectral shape is key.** The spectral shape of the sound, not its volume, is what gives away the fill level.
- **Physical Smoothing is essential.** Applying a rolling average and enforcing monotonicity prevents sudden splashing noises from triggering false-positive stops.
- **GroupKFold prevents leakage.** Without grouping by recording, windows from the same recording appear in both train and test sets, artificially inflating scores.

---

## 🔮 Future Improvements

- Larger, more diverse dataset (more cup types, sizes, materials).
- Data augmentation with background noise to improve robustness in active environments.
- Real-time streaming implementation for embedded hardware deployment.
- Online adaptation to new cup types using few-shot learning.
