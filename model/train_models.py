import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = PROJECT_DIR / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"
CLEAN_DATA_FILE = PROJECT_DIR / "data" / "wdbc_clean.csv"
TEST_DATA_FILE = PROJECT_DIR / "test_data.csv"
MODEL_DIR = PROJECT_DIR / "model"
METRICS_FILE = MODEL_DIR / "metrics.json"

RANDOM_STATE = 42
REQUIRED_METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

base_features = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

FEATURE_COLUMNS = []
for suffix in ["mean", "se", "worst"]:
    for feature in base_features:
        FEATURE_COLUMNS.append(f"{feature}_{suffix}")

ALL_COLUMNS = ["id", "diagnosis"] + FEATURE_COLUMNS


def load_dataset(raw_file=RAW_DATA_FILE):
    data = pd.read_csv(raw_file, header=None, names=ALL_COLUMNS)
    data = data.drop(columns=["id"])
    data["diagnosis"] = data["diagnosis"].map({"B": 0, "M": 1})
    return data


def split_dataset(data, test_size=0.2):
    X = data[FEATURE_COLUMNS]
    y = data["diagnosis"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models():
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_model(model, X_test, y_test):
    predicted_class = model.predict(X_test)
    malignant_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predicted_class),
        "AUC": roc_auc_score(y_test, malignant_probability),
        "Precision": precision_score(y_test, predicted_class, zero_division=0),
        "Recall": recall_score(y_test, predicted_class, zero_division=0),
        "F1": f1_score(y_test, predicted_class, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predicted_class),
    }

    confusion = confusion_matrix(y_test, predicted_class)
    report = classification_report(
        y_test,
        predicted_class,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    return metrics, confusion, report


def validate_test_data(data):
    missing_columns = []
    for column in FEATURE_COLUMNS:
        if column not in data.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            "Uploaded CSV is missing these columns: " + ", ".join(missing_columns)
        )

    checked_data = data.copy()
    if "diagnosis" in checked_data.columns:
        if checked_data["diagnosis"].dtype == object:
            checked_data["diagnosis"] = checked_data["diagnosis"].map(
                {"B": 0, "M": 1, "0": 0, "1": 1}
            )
        checked_data["diagnosis"] = checked_data["diagnosis"].astype(int)

    return checked_data


def model_file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".pkl"

def train_and_save_artifacts():
    MODEL_DIR.mkdir(exist_ok=True)
    CLEAN_DATA_FILE.parent.mkdir(exist_ok=True)

    data = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(data)

    test_data = X_test.copy()
    test_data["diagnosis"] = y_test

    data.to_csv(CLEAN_DATA_FILE, index=False)
    test_data.to_csv(TEST_DATA_FILE, index=False)

    all_metrics = {}
    all_reports = {}

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)

        metrics, confusion, report = evaluate_model(model, X_test, y_test)
        rounded_metrics = {}
        for metric_name, value in metrics.items():
            rounded_metrics[metric_name] = round(float(value), 4)

        all_metrics[model_name] = rounded_metrics
        all_reports[model_name] = {
            "confusion_matrix": confusion.tolist(),
            "classification_report": report,
        }

        joblib.dump(model, MODEL_DIR / model_file_name(model_name))

    METRICS_FILE.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (MODEL_DIR / "reports.json").write_text(
        json.dumps(all_reports, indent=2),
        encoding="utf-8",
    )

    return all_metrics

if __name__ == "__main__":
    metrics_table = train_and_save_artifacts()
    print(pd.DataFrame(metrics_table).T.to_string())
