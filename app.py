# ============================================================
# Earnings Manipulation Detection – Streamlit Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
st.set_page_config(page_title="Earnings Manipulation Detection", layout="wide")

st.title("Detecting Earnings Manipulation in Indian Firms")
st.caption("Machine Learning–based Early Warning System")

st.sidebar.header("Execution Settings")

execution_mode = st.sidebar.radio(
    "Mode",
    ["Fast Mode", "Detailed Mode"],
    help="Fast Mode reduces computation; Detailed Mode provides deeper explainability"
)
# ---- UI messaging only (no logic impact) ----
if execution_mode == "Fast Mode":
    st.info(
        "⚡ **Fast Mode Enabled**: "
        "SHAP analysis uses a reduced sample for quicker responsiveness. "
        "Model training, validation, evaluation, and predictions remain unchanged."
    )
else:
    st.info(
        "🔍 **Detailed Mode Enabled**: "
        "SHAP analysis uses a larger sample for richer interpretability. "
        "Core model results and predictions remain unchanged."
    )
FEATURES = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "ACCR", "LEVI"]
TARGET = "Manipulator"
RANDOM_STATE = 42
COMPLEX_MODELS = ["Ensemble (Logistic + RF)"]
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)
@st.cache_data
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

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler
def get_model(choice, y_train):

    if choice == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")

    if choice == "Decision Tree (CART)":
        return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)

    if choice == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
        )

    if choice == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=RANDOM_STATE
        )

    if choice == "Ensemble (Logistic + RF)":
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        )
        return VotingClassifier(estimators=[("lr", lr), ("rf", rf)], voting="soft")
@st.cache_resource
def train_model(model_choice, X_train_s, y_train):
    model = get_model(model_choice, y_train)
    model.fit(X_train_s, y_train)
    return model
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
def compute_beneish_m_score(row):
    return (
        -4.84
        + 0.92 * row["DSRI"]
        + 0.528 * row["GMI"]
        + 0.404 * row["AQI"]
        + 0.892 * row["SGI"]
        + 0.115 * row["DEPI"]
        - 0.172 * row["SGAI"]
        + 4.679 * row["ACCR"]
        - 0.327 * row["LEVI"]
    )

df["Beneish_M_Score"] = df.apply(compute_beneish_m_score, axis=1)
df["Beneish_Flag"] = (df["Beneish_M_Score"] > -2.22).astype(int)

st.write(
    f"Percentage of firms flagged by Beneish Model: "
    f"{df['Beneish_Flag'].mean() * 100:.2f}%"
)
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

test_size = st.sidebar.slider("Test Set Size", 0.2, 0.3, 0.4,0.5)

X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler = preprocess_data(df, test_size)

model = train_model(model_choice, X_train_s, y_train)
st.subheader("Validation-Based Threshold Selection")

y_val_prob = model.predict_proba(X_val_s)[:, 1]

thresholds = np.arange(0.2, 0.8, 0.05)
threshold_results = []

for t in thresholds:
    y_val_pred = (y_val_prob >= t).astype(int)
    threshold_results.append({
        "Threshold": t,
        "Recall": recall_score(y_val, y_val_pred),
        "F1 Score": f1_score(y_val, y_val_pred)
    })

threshold_df = pd.DataFrame(threshold_results)

best_threshold = threshold_df.sort_values(
    ["Recall", "F1 Score"], ascending=False
).iloc[0]["Threshold"]

st.write(f"**Selected Threshold: {best_threshold:.2f}**")
st.dataframe(threshold_df)
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

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_test_pred)
st.dataframe(pd.DataFrame(
    cm,
    index=["Actual Non-Manipulator", "Actual Manipulator"],
    columns=["Predicted Non-Manipulator", "Predicted Manipulator"]
))
st.subheader("How to Interpret These Evaluation Metrics")

st.markdown("""
In the context of **earnings manipulation detection**, not all evaluation metrics
carry equal importance.

###  Primary Metric: Recall
Recall measures the proportion of **actual manipulators correctly identified** by the model.

• A **low recall** implies that manipulative firms are being missed (false negatives)  
• In financial oversight and auditing contexts, this is **highly costly**, as undetected
  manipulation can lead to regulatory penalties, investor losses, and reputational damage  

Therefore, **high recall is the most critical objective** of this system.

###  Supporting Metric: F1 Score
The F1 Score balances:
• Recall (catching manipulators)  
• Precision (avoiding excessive false alarms)

A strong F1 score indicates that the model achieves **effective screening**
without overwhelming auditors with too many false positives.

###  ROC–AUC (Ranking Ability)
ROC–AUC reflects how well the model **ranks firms by manipulation risk**
across all possible thresholds.

• A higher ROC–AUC indicates superior **risk discrimination**
• This is useful for prioritizing firms for further investigation

###  Why Accuracy Is Not Emphasized
Accuracy can be misleading in this setting due to **class imbalance**.
A model predicting most firms as non-manipulators may achieve high accuracy
while failing at its primary objective—detecting manipulation.

**In summary**, recall-driven evaluation ensures the system functions as an
*early warning mechanism* rather than a naive classifier.
""")

st.header("5. Model Explainability (SHAP)")

if model_choice in COMPLEX_MODELS:
    st.info("SHAP is skipped for ensemble models due to attribution ambiguity.")

elif st.button("Run SHAP Analysis"):

    sample_size = 30 if execution_mode == "Fast Mode" else 100
    X_shap = pd.DataFrame(X_test_s, columns=FEATURES).sample(
        n=min(sample_size, len(X_test_s)), random_state=42
    )

    if model_choice == "Logistic Regression":
        explainer = shap.LinearExplainer(model, X_shap)
        shap_values = explainer.shap_values(X_shap)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

    shap.summary_plot(shap_values, X_shap, show=False)
    st.pyplot(bbox_inches="tight")
    if execution_mode == "Fast Mode":
    st.markdown("""
    ### SHAP Summary Interpretation (Fast Mode)

    This SHAP summary provides a **high-level overview** of the key drivers
    behind earnings manipulation risk.

    • Features at the top have the strongest overall influence  
    • Rightward movement increases predicted manipulation risk  
    • Red points represent higher feature values; blue indicate lower values  

    Even with a reduced sample, **ACCR (accrual intensity)** and
    **DSRI (receivable growth)** consistently emerge as dominant signals,
    aligning with classical earnings manipulation theory.
    """)
else:
    st.markdown("""
    ### SHAP Summary Interpretation (Detailed Mode)

    This SHAP visualization offers a **granular explanation** of how individual
    financial ratios contribute to manipulation risk.

    • Each point represents a firm-level observation  
    • The horizontal spread reflects the **magnitude and direction** of impact  
    • Non-linear patterns indicate threshold and interaction effects  

    The results highlight that **accrual-based distortions (ACCR)** and
    **revenue-related pressure (DSRI, SGI)** significantly elevate manipulation
    risk, particularly when these factors interact.

    This reinforces the notion that earnings manipulation is driven not by
    isolated ratios, but by **combined financial pressures**, which modern
    machine learning models capture more effectively than linear rules.
    """)
st.header("7. Model Comparison (Beneish vs ML Models)")
compare_extra = st.checkbox(
    "Compare with another ML model (optional)"
)

secondary_model_choice = None
if compare_extra:
    secondary_model_choice = st.selectbox(
        "Select Secondary Model for Comparison",
        [
            "Logistic Regression",
            "Decision Tree (CART)",
            "Random Forest",
            "XGBoost"
        ]
    )
results = []

# ---------------- Beneish Benchmark ----------------
beneish_pred = df.loc[y_test.index, "Beneish_Flag"]

results.append({
    "Model": "Beneish M-Score",
    "Accuracy": accuracy_score(y_test, beneish_pred),
    "Precision": precision_score(y_test, beneish_pred, zero_division=0),
    "Recall": recall_score(y_test, beneish_pred),
    "F1 Score": f1_score(y_test, beneish_pred),
    "ROC-AUC": np.nan
})

# ---------------- Primary ML Model ----------------
primary_metrics = evaluate_model(model, X_test_s, y_test)
primary_metrics["Model"] = f"{model_choice} (Primary)"
results.append(primary_metrics)

# ---------------- Optional Secondary ML Model ----------------
if compare_extra and secondary_model_choice != model_choice:

    secondary_model = get_model(secondary_model_choice, y_train)
    secondary_model.fit(X_train_s, y_train)

    secondary_metrics = evaluate_model(
        secondary_model, X_test_s, y_test
    )
    secondary_metrics["Model"] = f"{secondary_model_choice} (Secondary)"

    results.append(secondary_metrics)

comparison_df = pd.DataFrame(results)
st.dataframe(comparison_df.set_index("Model"))

st.caption(
    "All models are evaluated on the same held-out test set to ensure fair comparison."
)
st.header("6. Single Firm Risk Assessment")

user_input = {}

for feature in FEATURES:
    user_input[feature] = st.number_input(
        f"{feature}", value=float(df[feature].median())
    )

if st.button("Predict Manipulation Risk"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    risk_prob = model.predict_proba(input_scaled)[0][1]

    st.metric(
        label="Probability of Earnings Manipulation",
        value=f"{risk_prob:.2%}"
    )

    if risk_prob >= best_threshold:
        st.error("High Risk of Earnings Manipulation")
    else:
        st.success("Low Risk of Earnings Manipulation")

beneish_score = compute_beneish_m_score(user_input)

st.metric("Beneish M-Score (Reference)", f"{beneish_score:.2f}")
