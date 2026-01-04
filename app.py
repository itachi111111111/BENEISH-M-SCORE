# ============================================================
# Earnings Manipulation Detection – Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(page_title="Earnings Manipulation Detection", layout="wide")
st.title("Detecting Earnings Manipulation in Indian Firms")

# ============================================================
# SIDEBAR – EXECUTION MODE
# ============================================================

st.sidebar.header("Execution Settings")

execution_mode = st.sidebar.radio(
    "Mode",
    ["Fast Mode", "Detailed Mode"],
    help="Fast Mode disables SHAP and heavy computations"
)

st.caption("Machine Learning–based Early Warning System")

# ============================================================
# CONSTANTS
# ============================================================

FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]
TARGET = "Manipulator"
RANDOM_STATE = 42
HEAVY_MODELS = ["Ensemble (Logistic + RF)"]

# ============================================================
# DATA LOADING (CACHED)
# ============================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

# ============================================================
# PREPROCESSING (CACHED)
# ============================================================

@st.cache_data(show_spinner="Preprocessing data...")
def preprocess_data(df, test_size):

    X = df[FEATURES]
    y = df[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25, stratify=y_train_full,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (
        X_train_s, X_val_s, X_test_s,
        y_train, y_val, y_test,
        scaler, X_train.columns
    )

# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(choice, y_train):

    if choice == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    if choice == "Decision Tree (CART)":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)

    if choice == "Random Forest":
        return RandomForestClassifier(
            n_estimators=150, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE
        )

    if choice == "XGBoost":
        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE
        )

    if choice == "Ensemble (Logistic + RF)":
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        rf = RandomForestClassifier(
            n_estimators=120, max_depth=5,
            class_weight="balanced", random_state=RANDOM_STATE
        )
        return VotingClassifier(estimators=[("lr", lr), ("rf", rf)], voting="soft")

# ============================================================
# MODEL TRAINING (CACHED)
# ============================================================

@st.cache_resource(show_spinner="Training model...")
def train_model(model_choice, X_train, y_train):

    model = get_model(model_choice, y_train)
    model.fit(X_train, y_train)

    return model


# ============================================================
# DATA UPLOAD
# ============================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload the earnings manipulation dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)

if df[TARGET].dtype == object:
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

# ============================================================
# BENEISH BENCHMARK
# ============================================================

def compute_beneish_m_score(row):
    return (
        -4.84 + 0.92 * row["DSRI"] + 0.528 * row["GMI"]
        + 0.404 * row["AQI"] + 0.892 * row["SGI"]
        + 0.115 * row["DEPI"] - 0.172 * row["SGAI"]
        + 4.679 * row["ACCR"] - 0.327 * row["LEVI"]
    )

df["Beneish_M_Score"] = df.apply(compute_beneish_m_score, axis=1)
df["Beneish_Flag"] = (df["Beneish_M_Score"] > -2.22).astype(int)

st.write(
    f"Percentage of firms flagged by Beneish Model: "
    f"{df['Beneish_Flag'].mean() * 100:.2f}%"
)

# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.header("Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree (CART)",
        "Random Forest",
        "XGBoost",
        "Ensemble (Logistic + RF)"
    ]
)

test_size = st.sidebar.slider("Test Set Size", 0.2, 0.4, 0.05, 0.3)

(
    X_train_s, X_val_s, X_test_s,
    y_train, y_val, y_test,
    scaler, feature_names
) = preprocess_data(df, test_size)

model = train_model(model_choice, X_train_s, y_train)

# ============================================================
# VALIDATION-BASED THRESHOLD
# ============================================================

y_val_prob = model.predict_proba(X_val_s)[:, 1]
thresholds = np.arange(0.2, 0.8, 0.05)

scores = [
    {"Threshold": t,
     "Recall": recall_score(y_val, (y_val_prob >= t).astype(int)),
     "F1": f1_score(y_val, (y_val_prob >= t).astype(int))}
    for t in thresholds
]

threshold_df = pd.DataFrame(scores)
best_threshold = threshold_df.sort_values(
    ["Recall", "F1"], ascending=False
).iloc[0]["Threshold"]

st.write(f"**Selected Threshold (Validation-based): {best_threshold:.2f}**")

# ============================================================
# TEST EVALUATION
# ============================================================

st.header("3. Model Performance Evaluation")

y_test_prob = model.predict_proba(X_test_s)[:, 1]
y_test_pred = (y_test_prob >= best_threshold).astype(int)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Value": [
        accuracy_score(y_test, y_test_pred),
        precision_score(y_test, y_test_pred, zero_division=0),
        recall_score(y_test, y_test_pred),
        f1_score(y_test, y_test_pred),
        roc_auc_score(y_test, y_test_prob)
    ]
})

st.table(metrics_df)

# ============================================================
# SHAP EXPLAINABILITY (OPTIMIZED)
# ============================================================

st.header("5. Model Explainability (SHAP)")

if model_choice in HEAVY_MODELS:
    st.info("SHAP skipped for ensemble models.")

elif execution_mode == "Fast Mode":
    st.info("SHAP disabled in Fast Mode.")

else:
    if st.button("Run SHAP Analysis"):
        with st.spinner("Computing SHAP values..."):
            X_shap = pd.DataFrame(X_test_s, columns=feature_names).sample(100, random_state=42)

            if model_choice == "Logistic Regression":
                explainer = shap.LinearExplainer(model, X_shap)
                shap_values = explainer.shap_values(X_shap)
                shap.summary_plot(shap_values, X_shap, show=False)

            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_shap)
                shap.summary_plot(shap_values, X_shap, show=False)

            st.pyplot(bbox_inches="tight")
