import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# קבצי קלט
# =========================================================

# קובץ אחרי החלקה + מונוטוניות + החלטת עצירה
DECISION_CSV = "results/decision_random_forest.csv"

# קובץ מדדים של כל המודלים
METRICS_CSV = "results/metrics.csv"

# תיקיית גרפים
FIGURES_DIR = "results/figures"


# =========================================================
# יצירת תיקיית figures אם לא קיימת
# =========================================================
def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# =========================================================
# בחירת הקלטה אחת לדוגמה
#
# נבחר את ההקלטה הראשונה בקובץ.
# אפשר אחר כך לשנות ידנית ל־cup_id/take_id אחר.
# =========================================================
def get_example_recording(df):
    first_row = df.iloc[0]

    cup_id = first_row["cup_id"]
    take_id = first_row["take_id"]

    example = df[
        (df["cup_id"] == cup_id) &
        (df["take_id"] == take_id)
    ].copy()

    example = example.sort_values("start_time")

    return example, cup_id, take_id


# =========================================================
# גרף 1:
# Ground Truth מול Prediction
# =========================================================
def plot_prediction_vs_ground_truth():
    df = pd.read_csv(DECISION_CSV)

    example, cup_id, take_id = get_example_recording(df)

    plt.figure(figsize=(10, 5))

    # אמת
    plt.plot(
        example["end_time"],
        example["fill_percent"],
        label="Ground Truth"
    )

    # תחזית אחרי החלקה ומונוטוניות
    plt.plot(
        example["end_time"],
        example["prediction_monotonic"],
        label="Prediction after smoothing + monotonicity"
    )

    # קו יעד עצירה
    plt.axhline(
        y=90,
        linestyle="--",
        label="Stop threshold = 90%"
    )

    plt.xlabel("Time [sec]")
    plt.ylabel("Fill percentage [%]")
    plt.title(f"Prediction vs Ground Truth | {cup_id}, {take_id}")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(FIGURES_DIR, "prediction_vs_ground_truth.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# גרף 2:
# השוואת תחזית גולמית, מוחלקת ומונוטונית
# =========================================================
def plot_smoothing_demo():
    df = pd.read_csv(DECISION_CSV)

    example, cup_id, take_id = get_example_recording(df)

    plt.figure(figsize=(10, 5))

    # תחזית גולמית
    plt.plot(
        example["end_time"],
        example["prediction"],
        label="Raw prediction"
    )

    # תחזית אחרי Moving Average
    plt.plot(
        example["end_time"],
        example["prediction_smooth"],
        label="Moving average"
    )

    # תחזית אחרי מונוטוניות
    plt.plot(
        example["end_time"],
        example["prediction_monotonic"],
        label="Monotonic prediction"
    )

    plt.axhline(
        y=90,
        linestyle="--",
        label="Stop threshold = 90%"
    )

    plt.xlabel("Time [sec]")
    plt.ylabel("Predicted fill percentage [%]")
    plt.title(f"Decision Logic Demonstration | {cup_id}, {take_id}")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(FIGURES_DIR, "prediction_smoothing_demo.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# גרף 3:
# השוואת ביצועי מודלים לפי RMSE ו־R2
# =========================================================
def plot_metrics_comparison():
    metrics = pd.read_csv(METRICS_CSV)

    # מחשבים ממוצע לכל מודל על פני כל ה־folds
    summary = metrics.groupby("model")[["RMSE", "R2"]].mean().reset_index()

    # -----------------------------
    # גרף RMSE
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.bar(summary["model"], summary["RMSE"])

    plt.xlabel("Model")
    plt.ylabel("Mean RMSE [%]")
    plt.title("Model Comparison - RMSE")
    plt.xticks(rotation=30)
    plt.grid(axis="y")

    output_path = os.path.join(FIGURES_DIR, "metrics_rmse_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    # -----------------------------
    # גרף R2
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.bar(summary["model"], summary["R2"])

    plt.xlabel("Model")
    plt.ylabel("Mean R²")
    plt.title("Model Comparison - R²")
    plt.xticks(rotation=30)
    plt.grid(axis="y")

    output_path = os.path.join(FIGURES_DIR, "metrics_r2_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# הרצת כל הגרפים
# =========================================================
def main():
    ensure_figures_dir()

    plot_prediction_vs_ground_truth()
    plot_smoothing_demo()
    plot_metrics_comparison()

    print()
    print("All figures saved successfully.")


# =========================================================
# נקודת כניסה
# =========================================================
if __name__ == "__main__":
    main()