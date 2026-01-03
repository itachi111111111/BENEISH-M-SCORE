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
# EVALUATION HELPER (DEFINE ONCE)
# ============================================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Earnings Manipulation Detection",
    layout="wide"
)

st.title("Detecting Earnings Manipulation in Indian Firms")
st.caption("Machine Learning–based Early Warning System")

# ============================================================
# CONSTANTS
# ============================================================

FEATURES = [
    "DSRI", "GMI", "AQI", "SGI",
    "DEPI", "SGAI", "ACCR", "LEVI"
]

TARGET = "Manipulator"
RANDOM_STATE = 42

# ============================================================
# SIDEBAR – MODEL SELECTION
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

test_size = st.sidebar.slider(
    "Test Set Size",
    min_value=0.2,
    max_value=0.4,
    step=0.05,
    value=0.3
)
# ============================================================
# BENEISH M-SCORE (REFERENCE BENCHMARK)
# ============================================================

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

# ============================================================
# DATA UPLOAD
# ============================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload the earnings manipulation dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload the dataset to proceed.")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_csv(uploaded_file)
    # ------------------------------------------------------------
# TARGET ENCODING (Yes/No → 1/0)
# ------------------------------------------------------------

if df[TARGET].dtype == object:
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

# Safety check
if not set(df[TARGET].unique()).issubset({0, 1}):
    st.error("Target variable must be binary (Yes/No or 1/0).")
    st.stop()


st.subheader("Dataset Preview")
st.dataframe(df.head())
# Compute Beneish benchmark
df["Beneish_M_Score"] = df.apply(compute_beneish_m_score, axis=1)
df["Beneish_Flag"] = (df["Beneish_M_Score"] > -2.22).astype(int)
st.subheader("Beneish Model – Benchmark Summary")

st.write(
    f"Percentage of firms flagged by Beneish Model: "
    f"{df['Beneish_Flag'].mean() * 100:.2f}%"
)

# ============================================================
# VALIDATION
# ============================================================

required_cols = FEATURES + [TARGET]
missing = set(required_cols) - set(df.columns)

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# ============================================================
# PREPROCESSING
# ============================================================

# ============================================================
# TRAIN – VALIDATION – TEST SPLIT
# ============================================================

X = df[FEATURES]
y = df[TARGET]

# Step 1: Train + Test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    stratify=y,
    random_state=RANDOM_STATE
)

# Step 2: Train + Validation (from training only)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,   # 25% of train → validation
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

# Step 3: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrames (for SHAP & plots)
X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURES)
X_val_df = pd.DataFrame(X_val_scaled, columns=FEATURES)
X_test_df = pd.DataFrame(X_test_scaled, columns=FEATURES)
st.subheader("Data Splitting Strategy")

st.markdown("""
The dataset is divided into **training**, **validation**, and **test** subsets.

• **Training set**: Used to fit model parameters  
• **Validation set**: Used for model selection and threshold tuning  
• **Test set**: Used only for final performance reporting  

This separation ensures unbiased evaluation and prevents information leakage.
""")


# ============================================================
# MODEL FACTORY
# ============================================================

def get_model(choice):
    if choice == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

    if choice == "Decision Tree (CART)":
        return DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

    if choice == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE
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
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
        return VotingClassifier(
            estimators=[("lr", lr), ("rf", rf)],
            voting="soft"
        )

# ============================================================
# TRAIN MODEL
# ============================================================

st.header("2. Model Training")

model = get_model(model_choice)

with st.spinner("Training model..."):
    model.fit(X_train_scaled, y_train)

st.success("Model training completed.")
# ============================================================
# VALIDATION-BASED THRESHOLD TUNING
# ============================================================

y_val_prob = model.predict_proba(X_val_scaled)[:, 1]

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

st.write(f"**Selected Threshold (Validation-based): {best_threshold:.2f}**")
st.dataframe(threshold_df)
# ============================================================
# MODEL EVALUATION
# ============================================================

st.header("3. Model Performance Evaluation")

# Use validation-tuned threshold
y_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
    "Value": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        roc_auc_score(y_test, y_prob)
    ]
})

st.table(metrics_df)

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Manipulator", "Actual Manipulator"],
    columns=["Predicted Non-Manipulator", "Predicted Manipulator"]
)

st.dataframe(cm_df)

st.subheader("Recommended Evaluation Metric for This Use Case")

st.markdown("""
**Primary Metric: Recall**

The objective of this system is to act as an *early warning mechanism* for earnings manipulation.
Missing a manipulative firm (false negative) is significantly more costly than incorrectly flagging
a non-manipulator (false positive).

**Supporting Metrics:**
- **F1 Score**: balances detection sensitivity with screening efficiency
- **ROC-AUC**: assesses overall ranking ability across thresholds

**Not Emphasized:**
- **Accuracy**: may be misleading due to class imbalance
""")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header("4. Feature Importance")

if hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

elif model_choice == "Logistic Regression":
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_[0]
    }).sort_values(by="Coefficient", ascending=False)

    st.dataframe(coef_df)

else:
    st.info("Feature importance not available for this model.")
# ============================================================
# SHAP EXPLAINABILITY (SELECTIVE & VALID)
# ============================================================

st.header("5. Model Explainability (SHAP)")

# ---------- CASE 1: Logistic Regression ----------
if model_choice == "Logistic Regression":

    st.subheader("Global Feature Impact (Test Set)")

    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_test_df,
        show=False
    )
    st.pyplot(fig)

    # 📍 SHAP EXPLANATION (TEXT GOES HERE)
    st.markdown("""
    ### How to interpret this SHAP plot

    • Features at the top have the strongest influence on manipulation risk  
    • Red points indicate higher feature values; blue indicate lower values  
    • Rightward movement increases predicted manipulation risk  

    The plot shows that **accrual intensity (ACCR)** and **receivable growth (DSRI)**
    exert the strongest positive influence on predicted manipulation risk,
    consistent with classical accounting theory.
    """)

# ---------- CASE 2: Random Forest ----------
elif model_choice == "Random Forest":

    st.subheader("Global Feature Impact (Test Set)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values[1],  # class = Manipulator
        X_test_df,
        show=False
    )
    st.pyplot(fig)

    # 📍 SHAP EXPLANATION (TEXT GOES HERE)
    st.markdown("""
    ### How to interpret this SHAP plot

    • Each point represents a firm-level observation  
    • Rightward shifts indicate increased manipulation risk  
    • ACCR and DSRI display strong non-linear effects  

    This suggests that manipulation risk rises sharply when accrual-based
    earnings adjustments intensify, reinforcing the relevance of Beneish-style
    indicators under modern machine learning.
    """)

elif model_choice == "XGBoost":

    st.subheader("Global Feature Impact (Test Set)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)

    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values,
        X_test_df,
        show=False
    )
    st.pyplot(fig)

    st.markdown("""
    ### How to interpret this SHAP plot

    • XGBoost captures complex, non-linear interactions  
    • ACCR and DSRI show strong amplification effects  
    • Risk rises sharply beyond threshold levels  

    This indicates that earnings manipulation risk is driven
    by interaction effects rather than linear ratio movements.
    """)
# ---------- CASE 4: Ensemble / Stacking ----------
else:
    st.info(
        "SHAP explainability is intentionally skipped for ensemble and stacking models. "
        "These models combine multiple learners and do not have a unique, "
        "well-defined feature attribution structure."
    )
st.header("7. Model Comparison (Beneish vs ML Models)")

results = []

# Beneish benchmark
beneish_pred = df.loc[X_test.index, "Beneish_Flag"]

results.append({
    "Model": "Beneish M-Score",
    "Accuracy": accuracy_score(y_test, beneish_pred),
    "Precision": precision_score(y_test, beneish_pred),
    "Recall": recall_score(y_test, beneish_pred),
    "F1 Score": f1_score(y_test, beneish_pred),
    "ROC-AUC": np.nan
})

# ML models to compare
models_to_compare = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
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
}

for name, mdl in models_to_compare.items():
    mdl.fit(X_train_scaled, y_train)
    metrics = evaluate_model(mdl, X_test_scaled, y_test)
    metrics["Model"] = name
    results.append(metrics)

comparison_df = pd.DataFrame(results)
st.dataframe(comparison_df.set_index("Model"))
# ============================================================
# SINGLE FIRM PREDICTION
# ============================================================

st.header("6. Single Firm Risk Assessment")

user_input = {}

for feature in FEATURES:
    user_input[feature] = st.number_input(
        f"{feature}",
        value=float(df[feature].median())
    )

if st.button("Predict Manipulation Risk"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)

    risk_prob = model.predict_proba(input_scaled)[0][1]

    st.metric(
        label="Probability of Earnings Manipulation",
        value=f"{risk_prob:.2%}"
    )

    if risk_prob >= 0.5:
        st.error("High Risk of Earnings Manipulation")
    else:
        st.success("Low Risk of Earnings Manipulation")
        # Beneish score for single firm
beneish_score = compute_beneish_m_score(user_input)

st.metric(
    label="Beneish M-Score (Reference Benchmark)",
    value=f"{beneish_score:.2f}"
)

if beneish_score > -2.22:
    st.warning("Beneish Model Assessment: Likely Manipulator")
else:
    st.success("Beneish Model Assessment: Non-Manipulator")


# ============================================================
# END OF APPLICATION
# ============================================================
