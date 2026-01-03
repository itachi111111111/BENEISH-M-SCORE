Earnings Manipulation Detection using Machine Learning

An end-to-end Streamlit-based decision support system for detecting earnings manipulation risk in firms using classical accounting theory (Beneish M-Score) and modern machine learning models.

This project is designed as an early-warning screening tool, not a fraud confirmation system, and emphasizes recall-oriented detection, validation-based evaluation, and interpretability.

 Project Overview

Earnings manipulation poses significant risks to investors, auditors, and regulators. Traditional models such as the Beneish M-Score provide useful red flags but suffer from high false positives and limited adaptability.

This project enhances detection by:

Using machine learning classifiers trained on Beneish-style financial ratios

Explicitly separating training, validation, and test datasets

Performing validation-based threshold tuning

Comparing ML models against the Beneish benchmark

Providing firm-level risk assessment via an interactive web interface

Objectives

Detect firms with high likelihood of earnings manipulation

Minimize false negatives (missed manipulators)

Compare classical vs ML-based detection

Ensure methodological transparency and interpretability

Provide a decision-support dashboard, not a black-box predictor
Evaluation Strategy (Key Strength)
Dataset Splitting

The dataset is divided into three mutually exclusive subsets:

Training set → Model fitting

Validation set → Model selection & threshold tuning

Test set → Final, unbiased performance reporting

This prevents information leakage and ensures reliable evaluation.

Primary Evaluation Metric

Recall (Sensitivity) is emphasized because:

Missing a manipulator (false negative) is more costly than false alarms

The system is intended for audit/regulatory screening

Supporting metrics:

F1 Score – balances recall and precision

ROC-AUC – model comparison

Accuracy – reported but not emphasized due to class imbalance

Interpretability

SHAP (SHapley Additive exPlanations) is applied selectively:

✔ Logistic Regression

✔ Random Forest

✔ XGBoost

SHAP is intentionally skipped for ensemble/stacking models due to lack of a unified attribution structure.

This ensures theoretical validity of explanations.

Beneish M-Score Benchmark

The Beneish M-Score is calculated using the original formula:

𝑀
=
−
4.84
+
0.92
⋅
𝐷
𝑆
𝑅
𝐼
+
0.528
⋅
𝐺
𝑀
𝐼
+
0.404
⋅
𝐴
𝑄
𝐼
+
0.892
⋅
𝑆
𝐺
𝐼
+
0.115
⋅
𝐷
𝐸
𝑃
𝐼
−
0.172
⋅
𝑆
𝐺
𝐴
𝐼
+
4.679
⋅
𝐴
𝐶
𝐶
𝑅
−
0.327
⋅
𝐿
𝐸
𝑉
𝐼
M=−4.84+0.92⋅DSRI+0.528⋅GMI+0.404⋅AQI+0.892⋅SGI+0.115⋅DEPI−0.172⋅SGAI+4.679⋅ACCR−0.327⋅LEVI

Firms with M-Score > −2.22 are flagged as likely manipulators

Used strictly as a reference benchmark, not the final decision rule
