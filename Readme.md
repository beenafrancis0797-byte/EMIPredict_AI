# 💸 EMIPredict AI — EMI Eligibility & EMI Estimator

This project is a **Streamlit + MLflow powered app** that predicts whether a user is *eligible* for a loan based on their income, expenses, and financial ratios — and estimates the possible monthly EMI payment.

The project consists of:
- A trained **classification model** for eligibility prediction.
- A trained **regression model** for EMI amount estimation.
- A **preprocessing pipeline** (`preprocessor_coltransformer.joblib`) and **feature names** (`feature_names.json`).
- A **Streamlit app** (`app.py`) for real-time predictions.
- **MLflow experiment logs** under `mlruns/`.

---

## 🚀 Features

✅ Predicts EMI eligibility (Eligible / Not Eligible / High Risk)  
✅ Estimates monthly EMI amount using regression  
✅ Shows model confidence and reasoning  
✅ Includes DTI (Debt-to-Income) sanity check layer  
✅ MLflow integration for model tracking  
✅ Easy deployment to Streamlit Cloud or Docker  

---

## 🧩 Folder Structure

EMIPredict_AI_Experiment/
├── app.py
├── preprocessor_coltransformer.joblib
├── model_results/
│ ├── best_classifier_XGBoostClassifier.joblib
│ ├── best_regressor_XGBRegressor.joblib
├── feature_names.json
├── requirements.txt
├── README.md
└── mlruns/
