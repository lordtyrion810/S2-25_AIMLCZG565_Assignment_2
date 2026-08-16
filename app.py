import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model.train_models import (
    FEATURE_COLUMNS,
    METRICS_FILE,
    TEST_DATA_FILE,
    evaluate_model,
    model_file_name,
    validate_test_data,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
]


@st.cache_resource
def load_trained_models():
    models = {}
    for name in MODEL_NAMES:
        models[name] = joblib.load(MODEL_DIR / model_file_name(name))
    return models


@st.cache_data
def load_default_test_data():
    return pd.read_csv(TEST_DATA_FILE)


def load_metrics_table():
    if not METRICS_FILE.exists():
        return pd.DataFrame()

    saved_metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return pd.DataFrame(saved_metrics).T


def show_metric_cards(metrics):
    columns = st.columns(len(metrics))
    for column, metric_name in zip(columns, metrics):
        column.metric(metric_name, f"{metrics[metric_name]:.4f}")


st.set_page_config(
    page_title="Breast Cancer Classification Models",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Breast Cancer Classification Models")
st.caption("Dataset: Wisconsin Diagnostic Breast Cancer")

models = load_trained_models()
comparison_table = load_metrics_table()

uploaded_csv = st.sidebar.file_uploader("Upload test CSV", type="csv")
selected_model_name = st.sidebar.selectbox("Select model", MODEL_NAMES)
selected_model = models[selected_model_name]

if uploaded_csv is None:
    test_data = load_default_test_data()
    st.sidebar.info("Using the default test_data.csv from the repository.")
else:
    test_data = pd.read_csv(uploaded_csv)

try:
    test_data = validate_test_data(test_data)
except ValueError as error:
    st.error(str(error))
    st.stop()

X_test = test_data[FEATURE_COLUMNS]
predictions = selected_model.predict(X_test)

st.subheader("Model Comparison")
if comparison_table.empty:
    st.warning("Metrics file not found")
else:
    st.dataframe(comparison_table, width="stretch")

st.subheader(f"Selected Model: {selected_model_name}")

if "diagnosis" in test_data.columns:
    y_test = test_data["diagnosis"]
    metrics, confusion, report = evaluate_model(selected_model, X_test, y_test)

    show_metric_cards(metrics)

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.markdown("#### Confusion Matrix")
        confusion_table = pd.DataFrame(
            confusion,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"],
        )
        st.dataframe(confusion_table, width="stretch")

    with right_column:
        st.markdown("#### Classification Report")
        st.dataframe(pd.DataFrame(report).T, width="stretch")
else:
    st.info("No diagnosis column found, so only predictions are shown.")

st.subheader("Prediction Preview")
preview_data = test_data.copy()
preview_data["prediction"] = [
    "Malignant" if prediction == 1 else "Benign" for prediction in predictions
]
st.dataframe(preview_data.head(25), width="stretch")
