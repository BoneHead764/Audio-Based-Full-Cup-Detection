import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


# =========================================================
# קבצים ותיקיות
# =========================================================

# קובץ המאפיינים שיצרנו בשלב הקודם
FEATURES_CSV = "data/processed/features.csv"

# תיקיית תוצאות כללית
RESULTS_DIR = "results"

# תיקייה לשמירת מודלים אם נרצה בהמשך
MODELS_DIR = "results/models"

# קובץ שאליו נשמור את מדדי הביצועים
METRICS_CSV = "results/metrics.csv"


# =========================================================
# טעינת הדאטה
# =========================================================
def load_data():

    # קריאת טבלת המאפיינים
    df = pd.read_csv(FEATURES_CSV)

    # -----------------------------------------------------
    # עמודות metadata:
    #
    # אלה עמודות חשובות לדוח, לגרפים ול־GroupKFold,
    # אבל הן לא אמורות להיכנס למודל כ־features.
    #
    # לדוגמה:
    # אם נכניס cup_id למודל, הוא עלול ללמוד "זהות כוס"
    # במקום ללמוד קשר אמיתי בין צליל לאחוז מילוי.
    # -----------------------------------------------------
    meta_cols = [
        "audio_path",
        "cup_id",
        "take_id",
        "window_id",
        "start_time",
        "end_time",
        "fill_percent"
    ]

    # -----------------------------------------------------
    # כל עמודה שאינה metadata נחשבת feature מספרי
    # למשל: rms, zcr, spectral_centroid, mfcc_1 וכו'
    # -----------------------------------------------------
    feature_cols = [col for col in df.columns if col not in meta_cols]

    # מטריצת הקלט למודל
    X = df[feature_cols].values

    # התווית שהמודל צריך לחזות: אחוז מילוי
    y = df["fill_percent"].values

    # קבוצות לאימות:
    # כל הדוגמאות מאותה כוס מקבלות אותו group.
    groups = df["cup_id"].values

    return df, X, y, groups, feature_cols


# =========================================================
# הגדרת מודלים קלאסיים
# =========================================================
def get_models():

    # -----------------------------------------------------
    # Ridge Regression:
    # מודל ליניארי פשוט.
    # טוב כ־baseline כדי להראות אם יש קשר בסיסי בין features ל־fill%.
    # -----------------------------------------------------

    # -----------------------------------------------------
    # SVR:
    # מודל קלאסי לא ליניארי.
    # חייב StandardScaler כי הוא רגיש לסקאלה של המאפיינים.
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Random Forest:
    # מודל חזק, יציב, לא דורש scaling.
    # טוב לדאטה קטן־בינוני וליחסים לא ליניאריים.
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Gradient Boosting:
    # מודל boosting קלאסי.
    # לעיתים נותן תחזיות חלקות ומדויקות יותר מ־Random Forest.
    # -----------------------------------------------------

    models = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "svr_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=10, epsilon=2))
        ]),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    return models


# =========================================================
# אימון והערכת מודל אחד
# =========================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):

    # אימון על כוסות האימון בלבד
    model.fit(X_train, y_train)

    # חיזוי על כוסות שלא נראו בזמן האימון
    pred = model.predict(X_test)

    # -----------------------------------------------------
    # מדדים:
    #
    # MSE  - שגיאה ריבועית ממוצעת
    # RMSE - שורש MSE, באחוזי מילוי
    # MAE  - שגיאה מוחלטת ממוצעת
    # R2   - כמה מהשונות המודל מסביר
    # -----------------------------------------------------
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return pred, mse, rmse, mae, r2


# =========================================================
# אימון עם GroupKFold
# =========================================================
def train_with_groupkfold():

    # יצירת תיקיות תוצאות אם הן לא קיימות
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # טעינת הדאטה
    df, X, y, groups, feature_cols = load_data()

    # קבלת המודלים
    models = get_models()

    # -----------------------------------------------------
    # מספר ה־folds לא יכול להיות גדול ממספר הכוסות.
    #
    # אם יש 3 כוסות → יהיו 3 folds.
    # אם יש 5 ומעלה → נשתמש ב־5 folds.
    # -----------------------------------------------------
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)

    if n_splits < 2:
        raise ValueError("צריך לפחות 2 כוסות שונות בשביל GroupKFold")

    # -----------------------------------------------------
    # GroupKFold:
    #
    # בכל fold, כוס שלמה נשארת לבדיקה.
    # כך מונעים data leakage.
    # -----------------------------------------------------
    gkf = GroupKFold(n_splits=n_splits)

    all_metrics = []

    # מעבר על כל מודל
    for model_name, model in models.items():

        print()
        print("======================================")
        print(f"Training model: {model_name}")
        print("======================================")

        fold_predictions = []

        # -------------------------------------------------
        # split לפי groups:
        # sklearn דואג שאותה כוס לא תהיה גם באימון וגם בבדיקה.
        # -------------------------------------------------
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # הכוסות שנמצאות בסט הבדיקה ב־fold הנוכחי
            test_cups = np.unique(groups[test_idx])

            # אימון והערכת המודל
            pred, mse, rmse, mae, r2 = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test
            )

            print(f"Fold {fold + 1}")
            print(f"Test cups: {test_cups}")
            print(f"MSE:  {mse:.3f}")
            print(f"RMSE: {rmse:.3f}")
            print(f"MAE:  {mae:.3f}")
            print(f"R2:   {r2:.3f}")
            print()

            # שמירת מדדים לדוח
            all_metrics.append({
                "model": model_name,
                "fold": fold + 1,
                "test_cups": ",".join(test_cups),
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

            # -------------------------------------------------
            # שמירת תחזיות לכל חלון.
            #
            # זה חשוב מאוד לגרפים:
            # Ground Truth מול Prediction לאורך זמן.
            # -------------------------------------------------
            fold_df = df.iloc[test_idx].copy()
            fold_df["prediction"] = pred
            fold_df["model"] = model_name
            fold_df["fold"] = fold + 1

            fold_predictions.append(fold_df)

        # איחוד כל התחזיות של המודל
        predictions_df = pd.concat(fold_predictions, ignore_index=True)

        # שמירת התחזיות לקובץ CSV
        pred_path = f"results/predictions_{model_name}.csv"
        predictions_df.to_csv(pred_path, index=False)

        print(f"Saved predictions: {pred_path}")

    # -----------------------------------------------------
    # שמירת כל המדדים של כל המודלים
    # -----------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV, index=False)

    print()
    print("======================================")
    print(f"Saved metrics to: {METRICS_CSV}")
    print("======================================")

    # הדפסת ממוצע ביצועים לכל מודל
    print(metrics_df.groupby("model")[["MSE", "RMSE", "MAE", "R2"]].mean())


# =========================================================
# נקודת כניסה
# =========================================================
if __name__ == "__main__":

    train_with_groupkfold()