"""Train the final Random Forest model and save it as models/liver_disease_model.pkl."""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ─── Column definitions ───────────────────────────────────────────────────────
NUMERIC = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "LiverFunctionTest"]
NOMINAL = ["Gender", "Smoking", "Diabetes", "Hypertension"]
ORDINAL = ["GeneticRisk"]
TARGET  = "Diagnosis"

COLUMN_MAPS = {
    "Gender":       {0: "Male",     1: "Female"},
    "Smoking":      {0: "No",       1: "Yes"},
    "GeneticRisk":  {0: "Low",      1: "Medium", 2: "High"},
    "Diabetes":     {0: "No",       1: "Yes"},
    "Hypertension": {0: "No",       1: "Yes"},
    "Diagnosis":    {0: "Negative", 1: "Positive"},
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col, mapping in COLUMN_MAPS.items():
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].map(mapping)
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC),
        ("ord", Pipeline([("ordinal", OrdinalEncoder(categories=[["Low", "Medium", "High"]]))]), ORDINAL),
        ("nom", Pipeline([("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))]), NOMINAL),
    ], verbose_feature_names_out=False)

    rf = RandomForestClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        n_estimators=100,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])


def main():
    root = Path(__file__).resolve().parent.parent
    csv_path = root / "data" / "liver_disease_data.csv"
    model_path = root / "models" / "liver_disease_model.pkl"

    print("Loading data…")
    df = load_data(csv_path)
    print(f"  {len(df)} rows loaded. Target distribution:\n{df[TARGET].value_counts().to_string()}")

    X = df[NUMERIC + ORDINAL + NOMINAL]
    y = (df[TARGET] == "Positive").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    print("Training model…")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"\nTest F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Model saved: {model_path}")


if __name__ == "__main__":
    main()
