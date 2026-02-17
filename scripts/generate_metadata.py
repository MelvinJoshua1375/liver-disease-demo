"""Train all models, compute metrics, and save models/model_metadata.json."""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

NUMERIC  = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity", "LiverFunctionTest"]
NOMINAL  = ["Gender", "Smoking", "Diabetes", "Hypertension"]
ORDINAL  = ["GeneticRisk"]
TARGET   = "Diagnosis"

COLUMN_MAPS = {
    "Gender":       {0: "Male",     1: "Female"},
    "Smoking":      {0: "No",       1: "Yes"},
    "GeneticRisk":  {0: "Low",      1: "Medium", 2: "High"},
    "Diabetes":     {0: "No",       1: "Yes"},
    "Hypertension": {0: "No",       1: "Yes"},
    "Diagnosis":    {0: "Negative", 1: "Positive"},
}

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree":       CalibratedClassifierCV(DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=4, random_state=42), method="sigmoid"),
    "svc":                 SVC(probability=True, random_state=42),
    "naive_bayes":         GaussianNB(),
    "random_forest":       RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=4, random_state=42),
}


def get_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC),
        ("ord", Pipeline([("ordinal", OrdinalEncoder(categories=[["Low", "Medium", "High"]]))]), ORDINAL),
        ("nom", Pipeline([("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))]), NOMINAL),
    ], verbose_feature_names_out=False)


def load_data(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data" / "liver_disease_data.csv")
    for col, mapping in COLUMN_MAPS.items():
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].map(mapping)
    return df


def main():
    root = Path(__file__).resolve().parent.parent
    df = load_data(root)

    X = df[NUMERIC + ORDINAL + NOMINAL]
    y = (df[TARGET] == "Positive").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_comparison = []
    roc_curves = {}
    best_pipeline = None
    best_auc = 0.0

    for name, clf in MODELS.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", get_preprocessor()),
            ("classifier", clf),
        ])
        pipeline.fit(X_train, y_train)

        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_proba)
        acc = (y_test == y_pred).mean()

        # Store ROC curve points (downsample to ~50 points for JSON size)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        step = max(1, len(fpr) // 50)
        roc_curves[name.replace("_", " ").title()] = {
            "fpr": [round(float(x), 4) for x in fpr[::step]],
            "tpr": [round(float(x), 4) for x in tpr[::step]],
            "auc": round(float(auc), 4),
        }

        model_comparison.append({
            "model": name.replace("_", " ").title(),
            "accuracy": round(float(acc), 4),
            "f1_weighted": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
        })

        if name == "random_forest":
            best_pipeline = pipeline
            best_f1  = f1
            best_auc = auc
            best_acc = acc
            y_pred_best  = y_pred
            y_proba_best = y_proba

        print(f"  {name}: F1={f1:.3f}, AUC={auc:.3f}")

    # Feature importances from RF
    rf_clf = best_pipeline.named_steps["classifier"]
    prep   = best_pipeline.named_steps["preprocessor"]
    feat_names = list(prep.get_feature_names_out())
    importances = dict(zip(feat_names, rf_clf.feature_importances_.tolist(), strict=False))

    # Classification report
    cr = classification_report(y_test, y_pred_best, target_names=["Negative", "Positive"], output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best).tolist()

    metadata = {
        "model_name":    "Random Forest",
        "version":       "1.0.0",
        "training_date": str(date.today()),
        "dataset_size":  len(df),
        "accuracy":      round(float(best_acc), 4),
        "f1_score":      round(float(best_f1), 4),
        "roc_auc":       round(float(best_auc), 4),
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
        },
        "feature_importances": importances,
        "classification_report": cr,
        "confusion_matrix": cm,
        "roc_curves": roc_curves,
        "model_comparison": model_comparison,
    }

    meta_path = root / "models" / "model_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
