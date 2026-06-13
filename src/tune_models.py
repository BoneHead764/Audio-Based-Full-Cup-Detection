"""
Extended model search — classical ML only, no deep learning.
Runs RandomizedSearchCV with GroupKFold for every candidate,
then re-evaluates the best config of each family with the same
GroupKFold pipeline used in train_model.py so metrics are comparable.
Binary classification: is_full = (time_sec >= t_actual_full * 0.90)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

DATA_PATH      = Path("data/features_labeled.csv")
MODEL_DIR      = Path("models")
N_FOLDS        = 5
ROLLING_WIN    = 10
FULL_THRESHOLD = 0.90   # time_sec >= t_actual_full * 0.90 → is_full=1
STOP_THRESHOLD = 0.35   # probability threshold to trigger stop
N_ITER         = 10  # במקום 30
RANDOM_STATE   = 42


# ── helpers ────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["time_sec", "t_actual_full"])
    df["is_full"] = (df["time_sec"] >= df["t_actual_full"] * FULL_THRESHOLD).astype(int)
    df["group"]   = df["cup"] + "_" + df["file"]
    meta_cols     = ["file", "cup", "frame", "time_sec", "t_actual_full",
                     "fill_level_pct", "is_full", "group"]
    feature_cols  = [c for c in df.columns if c not in meta_cols]
    return df, feature_cols


def rolling_average(predictions: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.convolve(predictions, np.ones(window) / window, mode="full")
    return smoothed[window - 1: window - 1 + len(predictions)]


def find_stop_time(times, smooth_probs, threshold=STOP_THRESHOLD):
    for t, p in zip(times, smooth_probs):
        if p >= threshold:
            return float(t)
    return None


def compute_delta_t(t_pred_stop, t_actual_full):
    if t_pred_stop is None:
        return None
    return t_pred_stop - (t_actual_full * FULL_THRESHOLD)


def evaluate_pipeline(name, pipeline, df, feature_cols):
    cv     = GroupKFold(n_splits=N_FOLDS)
    X      = df[feature_cols].values
    y      = df["is_full"].values
    groups = df["group"].values

    f1s, aucs, precisions, recalls = [], [], [], []
    delta_ts = []
    early_count, late_count, never_count = 0, 0, 0

    for train_idx, test_idx in cv.split(X, y, groups):
        pipeline.fit(X[train_idx], y[train_idx])
        probs = pipeline.predict_proba(X[test_idx])[:, 1]

        test_df           = df.iloc[test_idx].copy()
        test_df["prob"]   = probs
        test_df["smooth"] = np.nan

        for _, rec in test_df.groupby("group"):
            idx      = rec.index
            smoothed = np.clip(rolling_average(rec["prob"].values, ROLLING_WIN), 0, 1)
            test_df.loc[idx, "smooth"] = smoothed

            t_actual = rec["t_actual_full"].iloc[-1]
            t_stop   = find_stop_time(rec["time_sec"].values, smoothed)
            dt       = compute_delta_t(t_stop, t_actual)

            if dt is None:
                never_count += 1
            elif dt < 0:
                early_count += 1
                delta_ts.append(dt)
            else:
                late_count += 1
                delta_ts.append(dt)

        y_test = y[test_idx]
        y_pred = (test_df["smooth"].values >= STOP_THRESHOLD).astype(int)
        smooth = test_df["smooth"].values

        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_test, smooth))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))

    delta_ts = np.array(delta_ts)
    return {
        "Precision"    : np.mean(precisions),
        "Recall"       : np.mean(recalls),
        "F1"           : np.mean(f1s),
        "AUC_ROC"      : np.mean(aucs),
        "Delta_t_mean" : delta_ts.mean() if len(delta_ts) else float("nan"),
        "Delta_t_std"  : delta_ts.std()  if len(delta_ts) else float("nan"),
        "Early"        : early_count,
        "Late"         : late_count,
        "Never"        : never_count,
    }


# ── candidate model definitions ────────────────────────────────────────────

def build_candidates():
    cw = "balanced"   # handles the 10% positive rate
    candidates = {}

    candidates["Random Forest"] = {
        "pipeline": Pipeline([("model", RandomForestClassifier(
            class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1))]),
        "param_dist": {
            "model__n_estimators":     [100, 200, 400, 600],
            "model__max_depth":        [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features":     ["sqrt", "log2", 0.5],
        },
    }

    candidates["Extra Trees"] = {
        "pipeline": Pipeline([("model", ExtraTreesClassifier(
            class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1))]),
        "param_dist": {
            "model__n_estimators":     [100, 200, 400, 600],
            "model__max_depth":        [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features":     ["sqrt", "log2", 0.5],
        },
    }

    candidates["HistGradientBoosting"] = {
        "pipeline": Pipeline([("model", HistGradientBoostingClassifier(
            class_weight=cw, random_state=RANDOM_STATE))]),
        "param_dist": {
            "model__max_iter":          [100, 200, 400],
            "model__learning_rate":     [0.01, 0.05, 0.1, 0.2],
            "model__max_depth":         [None, 5, 10, 15],
            "model__min_samples_leaf":  [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
    }

    candidates["Gradient Boosting"] = {
        "pipeline": Pipeline([("model", GradientBoostingClassifier(
            random_state=RANDOM_STATE))]),
        "param_dist": {
            "model__n_estimators":     [100, 200, 400],
            "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "model__max_depth":        [3, 5, 7],
            "model__subsample":        [0.7, 0.8, 1.0],
            "model__min_samples_leaf": [1, 5, 10],
        },
    }

    candidates["SVC"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", class_weight=cw,
                          probability=True, random_state=RANDOM_STATE)),
        ]),
        "param_dist": {
            "model__C":     [0.1, 1, 10, 50, 100],
            "model__gamma": ["scale", "auto", 0.001, 0.01],
        },
    }

    candidates["KNN"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_jobs=-1)),
        ]),
        "param_dist": {
            "model__n_neighbors": [3, 5, 10, 15, 20, 30],
            "model__weights":     ["uniform", "distance"],
            "model__metric":      ["euclidean", "manhattan"],
        },
    }

    candidates["Logistic Regression"] = {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight=cw, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "param_dist": {
            "model__C":      [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver":  ["lbfgs", "saga"],
        },
    }

    candidates["AdaBoost"] = {
        "pipeline": Pipeline([("model", AdaBoostClassifier(
            random_state=RANDOM_STATE))]),
        "param_dist": {
            "model__n_estimators":  [50, 100, 200],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
    }


    if HAS_XGB:
        candidates["XGBoost"] = {
            "pipeline": Pipeline([("model", XGBClassifier(
                tree_method="hist", random_state=RANDOM_STATE,
                verbosity=0, n_jobs=-1,
                scale_pos_weight=9))]),  # ~90/10 ratio
            "param_dist": {
                "model__n_estimators":     [100, 200, 400],
                "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
                "model__max_depth":        [3, 5, 7, 9],
                "model__subsample":        [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__reg_alpha":        [0, 0.1, 1.0],
                "model__reg_lambda":       [1, 5, 10],
            },
        }

    if HAS_LGB:
        candidates["LightGBM"] = {
            "pipeline": Pipeline([("model", LGBMClassifier(
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
                class_weight=cw))]),
            "param_dist": {
                "model__n_estimators":     [100, 200, 400],
                "model__learning_rate":    [0.01, 0.05, 0.1, 0.2],
                "model__num_leaves":       [31, 63, 127],
                "model__max_depth":        [-1, 5, 10],
                "model__subsample":        [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.6, 0.8, 1.0],
                "model__reg_alpha":        [0, 0.1, 1.0],
                "model__reg_lambda":       [0, 1.0, 5.0],
            },
        }

    return candidates


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df, feature_cols = load_data()
    X      = df[feature_cols].values
    y      = df["is_full"].values
    groups = df["group"].values
    print(f"  {len(df)} frames | {len(feature_cols)} features | "
          f"{df['group'].nunique()} recordings | "
          f"positive rate: {y.mean()*100:.1f}%")

    MODEL_DIR.mkdir(exist_ok=True)
    candidates = build_candidates()

    # ── Phase 1: hyperparameter search ────────────────────────────────────
    print(f"\n=== Phase 1: RandomizedSearchCV ({N_ITER} iters, {N_FOLDS}-fold GroupKFold) ===")
    gkf = GroupKFold(n_splits=N_FOLDS)
    best_pipelines = {}

    for name, spec in candidates.items():
        print(f"\n  Tuning: {name} ...", flush=True)
        search = RandomizedSearchCV(
            spec["pipeline"],
            param_distributions=spec["param_dist"],
            n_iter=N_ITER,
            cv=list(gkf.split(X, y, groups)),
            scoring="f1",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
        )
        search.fit(X, y)
        best_pipelines[name] = search.best_estimator_
        print(f"    Best CV F1:     {search.best_score_:.3f}")
        print(f"    Best params:    {search.best_params_}")

    # ── Phase 2: full evaluation with smoothing + Δt ──────────────────────
    print("\n=== Phase 2: Full evaluation (smoothing + Δt) ===")
    all_results = {}
    for name, pipeline in best_pipelines.items():
        print(f"\n  Evaluating: {name} ...", flush=True)
        metrics = evaluate_pipeline(name, pipeline, df, feature_cols)
        all_results[name] = metrics
        print(f"    Precision: {metrics['Precision']:.3f}")
        print(f"    Recall:    {metrics['Recall']:.3f}")
        print(f"    F1:        {metrics['F1']:.3f}")
        print(f"    AUC-ROC:   {metrics['AUC_ROC']:.3f}")
        print(f"    Δt mean:   {metrics['Delta_t_mean']:+.3f}s")
        print(f"    Δt std:    {metrics['Delta_t_std']:.3f}s")
        print(f"    Early: {metrics['Early']} | Late: {metrics['Late']} | Never: {metrics['Never']}")

    # ── Summary ────────────────────────────────────────────────────────────
    summary = pd.DataFrame(all_results).T
    summary.index.name = "Model"
    summary.to_csv(MODEL_DIR / "results_extended.csv")
    print("\n=== Results Summary ===")
    print(summary.round(3).to_string())

    # ── Pick winner: best F1 with normalized late-stop and timing penalties ──
    def score(k):
        r = all_results[k]
        total = r["Early"] + r["Late"] + r["Never"] + 1e-9
        late_rate = r["Late"] / total
        dt_penalty = abs(r["Delta_t_mean"]) if not np.isnan(r["Delta_t_mean"]) else 10.0
        return r["F1"] - 0.1 * late_rate - 0.02 * dt_penalty

    best_name = max(all_results, key=score)
    print(f"\nBest model: {best_name}")
    for k, v in all_results[best_name].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Retrain on full data and save ──────────────────────────────────────
    print(f"\nRetraining {best_name} on full dataset...")
    best_pipelines[best_name].fit(X, y)
    joblib.dump(best_pipelines[best_name], MODEL_DIR / "best_model.pkl")
    print("  Saved → models/best_model.pkl")

    # ── Feature importance plot ────────────────────────────────────────────
    last_step = best_pipelines[best_name].steps[-1][1]
    if hasattr(last_step, "feature_importances_"):
        importances = last_step.feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        plt.figure(figsize=(10, 6))
        plt.barh([feature_cols[i] for i in top_idx], importances[top_idx])
        plt.xlabel("Importance")
        plt.title(f"Top 20 Features — {best_name}")
        plt.tight_layout()
        out = MODEL_DIR / f"feature_importance_{best_name.replace(' ', '_')}.png"
        plt.savefig(out)
        plt.close()
        print(f"  Feature importance → {out}")

    # ── Comparison chart ───────────────────────────────────────────────────
    model_names = list(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, max(6, len(model_names) * 0.5)))
    metrics_to_plot = ["F1", "AUC_ROC", "Late"]
    titles          = ["F1 ↑", "AUC-ROC ↑", "Late stops ↓"]

    for ax, metric, title in zip(axes, metrics_to_plot, titles):
        vals   = [all_results[m][metric] for m in model_names]
        colors = ["#2ecc71" if m == best_name else "#3498db" for m in model_names]
        ax.barh(model_names, vals, color=colors)
        ax.set_title(title)
        ax.invert_yaxis()

    plt.suptitle("Model Comparison — Extended Search (Binary)", fontsize=13)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "model_comparison.png", dpi=120)
    plt.close()
    print("  Comparison chart → models/model_comparison.png")


if __name__ == "__main__":
    main()
