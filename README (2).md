# 🥤 Cup Fill Level Detector

A machine learning system that identifies whether a cup is nearly full by analyzing audio recordings — no cameras, no weight sensors, just sound.

---

## Overview

When water is poured into a cup, the acoustic resonance of the vessel changes as the water level rises. This project captures those changes using a microphone, extracts audio features, and trains a classifier to distinguish between "not full" and "nearly full" states.

Submitted as a Machine Learning course project at **Sami Shamoon College of Engineering (SCE)**, Department of Electrical and Electronics Engineering.

**Presenter:** Dmitri Bakhobsky  
**Authors:** Tomer Nor Milkov, Nikita Bodovsky, Oren, Or

---

## How It Works

1. **Record** water being poured into a cup (empty → full)
2. **Slice** each recording into 0.5-second windows
3. **Extract** acoustic features from every window
4. **Label** windows: last 10% of the recording = `1` (full), rest = `0` (not full)
5. **Train** a classifier on those labeled features
6. **Predict** fill state from new audio in real time

---

## Dataset

- **119 recordings** across 8 cup types
- **Cup materials:** Glass (×4), Plastic, Cardboard (×3)
- **~15 recordings per cup type** for balanced representation
- **1,495 samples** total after windowing

Recordings were made under consistent conditions: same room, same tap, fixed microphone distance, quiet environment.

---

## Features Extracted

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

## Models Evaluated

All models were evaluated with **5-Fold GroupKFold Cross-Validation**, grouping by recording to prevent data leakage between windows from the same file.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 95.85% | 77.71% | 53.37% | 62.76% |
| k-NN (k=5) | 96.12% | 75.29% | 62.58% | 68.07% |
| **Random Forest** | **96.32%** | **83.93%** | 56.47% | **67.18%** |

**Random Forest** was selected as the final model for its highest accuracy and precision, and its ability to provide feature importance analysis.

### Confusion Matrix (Random Forest)

|  | Predicted: Not Full | Predicted: Full |
|---|---|---|
| **Actual: Not Full** | 1384 ✅ | 12 ❌ |
| **Actual: Full** | 43 ❌ | 56 ✅ |

---

## Top Features (Feature Importance)

The Random Forest model revealed that **MFCC features dominate** the decision — particularly their standard deviation, which captures how much the spectral shape varies within each window.

```
1. mfcc_0_std       ~0.083
2. mfcc_2_mean      ~0.060
3. mfcc_5_mean      ~0.057
4. mfcc_1_std       ~0.053
5. bandwidth_mean   ~0.051
6. centroid_std     ~0.045
7. rms_mean         ~0.041
8. mfcc_0_mean      ~0.039
9. mfcc_3_mean      ~0.038
10. rolloff_std     ~0.035
```

This confirms the model is detecting **resonance changes** (frequency structure), not just volume.

---

## Project Structure

```
.
├── recordings/              # Raw WAV recordings, organized by cup type
│   ├── Glass 1/
│   ├── Glass 2/
│   ├── Plastic/
│   └── ...
│
├── build_dataset.py         # Slice recordings → extract features → save CSV
├── train_model.py           # Train & evaluate all models with cross-validation
├── RandomForest.py          # Train and save the final production model
├── FirstCode.py             # Exploratory: waveform visualization
│
├── dataset_graph.py         # Plot recording counts per cup type
├── confusion_matrix_graph.py# Plot the Random Forest confusion matrix
├── model_comparison_graph.py# Bar chart comparing model metrics
│
├── water_dataset.csv        # Generated feature dataset
├── final_water_model.pkl    # Saved Random Forest model (joblib)
│
└── dataset_distribution.png
    confusion_matrix.png
    feature_importance.png
    model_comparison.png
```

---

## Setup & Usage

### Requirements

```bash
pip install librosa numpy pandas scikit-learn matplotlib joblib
```

### 1. Build the dataset

Place your WAV recordings under `recordings/<cup_type>/` then run:

```bash
python build_dataset.py
```

This produces `water_dataset.csv`.

### 2. Train and evaluate models

```bash
python train_model.py
```

Prints cross-validation scores, confusion matrix, and classification report for all three models. Also outputs feature importance rankings.

### 3. Save the final model

```bash
python RandomForest.py
```

Saves `final_water_model.pkl` for deployment.

### 4. Generate charts

```bash
python dataset_graph.py
python model_comparison_graph.py
python confusion_matrix_graph.py
```

---

## Key Findings

- **Acoustic-only detection works.** A Random Forest classifier achieves ~96% accuracy with no visual or weight sensors.
- **MFCC features are most informative.** The spectral shape of the sound, not its volume, is what gives away the fill level.
- **GroupKFold is essential.** Without grouping by recording, windows from the same recording appear in both train and test sets, inflating scores artificially.
- **Recall for "full" class is the main challenge** (56%). The "full" class is rare (last 10% of recordings), creating class imbalance. Future work: SMOTE oversampling, adjusted class weights, or a lower classification threshold.

---

## Future Improvements

- Larger, more diverse dataset (more cup types, sizes, materials)
- More precise labeling of the actual moment the cup becomes full
- Evaluation in noisy environments
- Real-time inference pipeline (stream audio → predict → alert)
- Deep learning approaches (CNN on spectrograms, LSTM on feature sequences)
