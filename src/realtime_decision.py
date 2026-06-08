import os
import numpy as np
import pandas as pd

# =========================================================
# קובץ התחזיות של המודל
# =========================================================
PREDICTIONS_CSV = "results/predictions_random_forest.csv"

# קובץ פלט אחרי החלקה + מונוטוניות + החלטת עצירה
OUTPUT_CSV = "results/decision_random_forest.csv"

# אחוז המילוי שבו נרצה לעצור את הברז
TARGET_STOP_PERCENT = 90

# כמה חלונות אחורה נשתמש להחלקה
# 5 חלונות = בערך 1.25 שניות אם hop=0.25sec
SMOOTHING_WINDOW = 5

# כמה חלונות רצופים צריכים להיות מעל 90%
# כדי שלא נעצור בגלל קפיצה חד־פעמית של רעש
CONSECUTIVE_WINDOWS = 3


# =========================================================
# Moving Average
#
# מקבל סדרה של תחזיות:
# 40, 42, 80, 43, 45
#
# ומחליק אותן:
# 40, 41, 54, 51, 50
#
# הרעיון:
# רעש פתאומי לא יקפיץ מיד את ההחלטה.
# =========================================================
def moving_average(values, window=5):
    return (
        pd.Series(values)
        .rolling(window=window, min_periods=1)
        .mean()
        .values
    )


# =========================================================
# אכיפת מונוטוניות
#
# פיזיקלית, בזמן מילוי כוס, אחוז המילוי לא אמור לרדת.
#
# אבל מודל ML יכול לחזות:
# 30, 35, 33, 40
#
# זה לא הגיוני פיזיקלית.
#
# לכן נתקן ל:
# 30, 35, 35, 40
# =========================================================
def enforce_monotonicity(values):

    # np.maximum.accumulate מחזיר בכל נקודה
    # את הערך המקסימלי שנראה עד עכשיו.
    return np.maximum.accumulate(values)


# =========================================================
# מציאת זמן עצירה
#
# המערכת לא עוצרת כאשר יש חלון אחד מעל 90%,
# כי ייתכן שזה רעש או טעות רגעית.
#
# לכן דורשים כמה חלונות רצופים מעל הסף.
# =========================================================
def find_stop_time(df):

    # בדיקה לכל חלון:
    # האם התחזית המונוטונית גדולה או שווה ל־90%
    above_threshold = df["prediction_monotonic"] >= TARGET_STOP_PERCENT

    count = 0

    # מעבר על החלונות לפי סדר זמן
    for i, is_above in enumerate(above_threshold):

        if is_above:
            count += 1
        else:
            count = 0

        # אם קיבלנו מספיק חלונות רצופים מעל הסף
        if count >= CONSECUTIVE_WINDOWS:

            # זמן העצירה הוא סוף החלון הנוכחי
            stop_time = df.iloc[i]["end_time"]

            # אחוז המילוי האמיתי באותו רגע
            true_fill_at_stop = df.iloc[i]["fill_percent"]

            return stop_time, true_fill_at_stop

    # אם אף פעם לא עברנו את הסף בצורה יציבה
    return None, None


# =========================================================
# הפעלת לוגיקת החלטה על כל ההקלטות
# =========================================================
def apply_decision_logic():

    # קריאת התחזיות שהגיעו מהמודל
    df = pd.read_csv(PREDICTIONS_CSV)

    all_results = []
    stop_events = []

    # -----------------------------------------------------
    # חשוב:
    # עובדים על כל הקלטה בנפרד.
    #
    # לא מחליקים בין הקלטות שונות,
    # כי זה יהיה ערבוב לא נכון של דאטה.
    # -----------------------------------------------------
    grouped = df.groupby(["cup_id", "take_id"])

    for (cup_id, take_id), group in grouped:

        # סידור החלונות לפי זמן
        group = group.sort_values("start_time").copy()

        # התחזית המקורית של המודל
        raw_pred = group["prediction"].values

        # -------------------------------------------------
        # שלב 1:
        # החלקת התחזית כדי להקטין רעשי פתע
        # -------------------------------------------------
        group["prediction_smooth"] = moving_average(
            raw_pred,
            window=SMOOTHING_WINDOW
        )

        # -------------------------------------------------
        # שלב 2:
        # תיקון פיזיקלי — אחוז מילוי לא יורד בזמן מילוי
        # -------------------------------------------------
        group["prediction_monotonic"] = enforce_monotonicity(
            group["prediction_smooth"].values
        )

        # -------------------------------------------------
        # שלב 3:
        # מציאת הנקודה שבה המערכת הייתה אומרת "עצור"
        # -------------------------------------------------
        stop_time, true_fill_at_stop = find_stop_time(group)

        # עמודה שמסמנת האם העצירה כבר הופעלה
        group["stop_triggered"] = False

        if stop_time is not None:

            # כל חלון אחרי זמן העצירה יסומן כ־True
            group.loc[group["end_time"] >= stop_time, "stop_triggered"] = True

        # שומרים את כל החלונות אחרי העיבוד
        all_results.append(group)

        # שומרים סיכום קצר לכל הקלטה
        stop_events.append({
            "cup_id": cup_id,
            "take_id": take_id,
            "stop_time_sec": stop_time,
            "true_fill_at_stop": true_fill_at_stop,
            "target_stop_percent": TARGET_STOP_PERCENT
        })

    # איחוד כל ההקלטות לטבלה אחת
    result_df = pd.concat(all_results, ignore_index=True)

    # טבלת אירועי עצירה
    stop_df = pd.DataFrame(stop_events)

    # יצירת תיקיית results אם לא קיימת
    os.makedirs("results", exist_ok=True)

    # שמירת קובץ מלא:
    # כולל prediction רגיל, מוחלק, מונוטוני וסימון עצירה
    result_df.to_csv(OUTPUT_CSV, index=False)

    # שמירת קובץ סיכום:
    # לכל הקלטה — מתי עצרנו ומה היה אחוז המילוי האמיתי
    stop_df.to_csv("results/stop_events_random_forest.csv", index=False)

    print("Saved:")
    print(OUTPUT_CSV)
    print("results/stop_events_random_forest.csv")

    print()
    print("Stop events:")
    print(stop_df)


# =========================================================
# נקודת כניסה
# =========================================================
if __name__ == "__main__":

    apply_decision_logic()