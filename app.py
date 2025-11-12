# app.py
# Final tightened Streamlit app for EMIPredict AI with enforced sanity overrides
# Paste over your current app.py and restart Streamlit (streamlit run app.py)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os, json, re, traceback
from pathlib import Path

st.set_page_config(page_title="EMIPredict AI", layout="centered")

# ---------------- Config ----------------
BASE_DIR = Path(".").resolve()
RESULT_DIR = BASE_DIR / "model_results"
ARTIFACTS_DIR = BASE_DIR
PREPROCESSOR_FILE = "preprocessor_coltransformer.joblib"
FEATURE_NAMES_FILE = "feature_names.json"

RESULT_DIR.mkdir(exist_ok=True)

# ---------------- Helpers ----------------
def find_model_file(prefix):
    p = RESULT_DIR
    if not p.exists():
        return None
    candidates = sorted(list(p.glob(f"{prefix}_*.joblib")) + list(p.glob(f"*{prefix}*.joblib")))
    return candidates[0] if candidates else None

@st.cache_resource(ttl=3600)
def safe_joblib_load(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return ("__LOAD_ERROR__", f"{e}\n{traceback.format_exc()}")

@st.cache_resource(ttl=3600)
def load_models_and_preproc():
    output = {}
    pre_path = ARTIFACTS_DIR / PREPROCESSOR_FILE
    if not pre_path.exists():
        candidate = find_model_file("preprocessor") or find_model_file("preproc")
        pre_path = candidate if candidate else pre_path

    clf_path = find_model_file("best_classifier") or find_model_file("clf")
    reg_path = find_model_file("best_regressor") or find_model_file("reg")

    # --- fallback: check repo root for exact filenames (useful if files were uploaded to repo root) ---
    if (not clf_path) and (ARTIFACTS_DIR / "best_classifier_XGBoostClassifier.joblib").exists():
        clf_path = ARTIFACTS_DIR / "best_classifier_XGBoostClassifier.joblib"
    if (not reg_path) and (ARTIFACTS_DIR / "best_regressor_XGBRegressor.joblib").exists():
        reg_path = ARTIFACTS_DIR / "best_regressor_XGBRegressor.joblib"
    # also fallback for preprocessor exact filename in root (already set above, but ensure)
    if (pre_path is None or (isinstance(pre_path, Path) and not pre_path.exists())) and (ARTIFACTS_DIR / PREPROCESSOR_FILE).exists():
        pre_path = ARTIFACTS_DIR / PREPROCESSOR_FILE

    # preprocessor
    if pre_path and str(pre_path) not in ("", "None") and Path(pre_path).exists():
        pre_loaded = safe_joblib_load(pre_path)
        if isinstance(pre_loaded, tuple) and pre_loaded[0] == "__LOAD_ERROR__":
            output['preprocessor'] = None
            output['preprocessor_error'] = pre_loaded[1]
        else:
            output['preprocessor'] = pre_loaded
            output['preprocessor_error'] = None
    else:
        output['preprocessor'] = None
        output['preprocessor_error'] = None

    # classifier & regressor
    for label, path in (("classifier", clf_path), ("regressor", reg_path)):
        if path and str(path) not in ("", "None") and Path(path).exists():
            loaded = safe_joblib_load(path)
            if isinstance(loaded, tuple) and loaded[0] == "__LOAD_ERROR__":
                output[label] = None
                output[f"{label}_error"] = loaded[1]
            else:
                output[label] = loaded
                output[f"{label}_error"] = None
        else:
            output[label] = None
            output[f"{label}_error"] = None

    output['paths'] = {
        "preprocessor": str(pre_path) if pre_path and Path(pre_path).exists() else None,
        "clf": str(clf_path) if clf_path and Path(clf_path).exists() else None,
        "reg": str(reg_path) if reg_path and Path(reg_path).exists() else None
    }
    return output

artifacts = load_models_and_preproc()
preprocessor = artifacts.get('preprocessor')
clf = artifacts.get('classifier')
reg = artifacts.get('regressor')

# ---------------- Load expected feature names if present ----------------
EXPECTED_FEATURE_NAMES = None
if os.path.exists(FEATURE_NAMES_FILE):
    try:
        with open(FEATURE_NAMES_FILE, "r", encoding="utf-8") as f:
            EXPECTED_FEATURE_NAMES = json.load(f)
        EXPECTED_FEATURE_NAMES = [str(x) for x in EXPECTED_FEATURE_NAMES]
    except Exception:
        EXPECTED_FEATURE_NAMES = None

# ---------------- UI header & sidebar ----------------
st.sidebar.header("App info")
st.sidebar.write("Project folder:", str(BASE_DIR))
if EXPECTED_FEATURE_NAMES:
    st.sidebar.success(f"Loaded feature_names.json (len={len(EXPECTED_FEATURE_NAMES)})")
else:
    st.sidebar.info("No feature_names.json — best-effort alignment will be used.")

if artifacts.get('preprocessor_error'):
    st.sidebar.error("Preprocessor load error - see logs")
if artifacts.get('classifier_error'):
    st.sidebar.error("Classifier load error - see logs")
if artifacts.get('regressor_error'):
    st.sidebar.error("Regressor load error - see logs")

st.sidebar.success("Preprocessor loaded" if preprocessor else "Preprocessor NOT found")
st.sidebar.success("Classifier loaded" if clf else "Classifier NOT found")
st.sidebar.success("Regressor loaded" if reg else "Regressor NOT found")

# allow model uploads (useful during dev)
st.sidebar.markdown("---")
st.sidebar.subheader("Upload model artifact (optional)")
uploaded_pre = st.sidebar.file_uploader("Upload preprocessor (.joblib)", type=["joblib"])
uploaded_clf = st.sidebar.file_uploader("Upload classifier (.joblib)", type=["joblib"])
uploaded_reg = st.sidebar.file_uploader("Upload regressor (.joblib)", type=["joblib"])

def load_uploaded(fileobj, target_name):
    if fileobj is None:
        return None, None
    try:
        save_path = RESULT_DIR / f"uploaded_{target_name}.joblib"
        with open(save_path, "wb") as f:
            f.write(fileobj.getbuffer())
        loaded = joblib.load(save_path)
        return loaded, None
    except Exception as e:
        return None, str(e)

if uploaded_pre:
    preprocessor_uploaded, err = load_uploaded(uploaded_pre, "preprocessor")
    if preprocessor_uploaded:
        preprocessor = preprocessor_uploaded
        st.sidebar.success("Uploaded preprocessor loaded")
    else:
        st.sidebar.error(f"Upload preprocessor failed: {err}")

if uploaded_clf:
    clf_uploaded, err = load_uploaded(uploaded_clf, "classifier")
    if clf_uploaded:
        clf = clf_uploaded
        st.sidebar.success("Uploaded classifier loaded")
    else:
        st.sidebar.error(f"Upload classifier failed: {err}")

if uploaded_reg:
    reg_uploaded, err = load_uploaded(uploaded_reg, "regressor")
    if reg_uploaded:
        reg = reg_uploaded
        st.sidebar.success("Uploaded regressor loaded")
    else:
        st.sidebar.error(f"Upload regressor failed: {err}")

DEBUG_SHOW_FEATURES = st.sidebar.checkbox("Show debug features (X_model)", value=False)
SHOW_CACHE_CLEAR = st.sidebar.checkbox("Show cache clear button", value=True)
if SHOW_CACHE_CLEAR:
    if st.sidebar.button("Clear cached model artifacts"):
        st.cache_resource.clear()
        st.sidebar.success("Cleared cache — please reload page (F5) and rerun.")

st.title("EMIPredict AI — EMI Eligibility & Max EMI Predictor")
st.markdown("Fill the form and press **Predict**. App will inject derived features and align to saved preprocessor.")

# ---------------- Input form ----------------
with st.form("input_form"):
    st.subheader("Personal & Employment")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 18, 100, 32)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        family_size = st.number_input("Family Size", 1, 20, 3)
    with c2:
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        company_type = st.text_input("Company Type (optional)", "MNC")
        years_of_employment = st.number_input("Years of Employment", 0, 60, 6)
    with c3:
        monthly_salary = st.number_input("Monthly Salary (INR)", 0, 10_000_000, 120000, step=1000)
        house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 500000, 0)

    st.subheader("Monthly Obligations (INR)")
    c4, c5, c6 = st.columns(3)
    with c4:
        current_emi_amount = st.number_input("Existing monthly EMI total", 0, 1_000_000, 8000, step=500)
        school_fees = st.number_input("School fees", 0, 500000, 3000)
        college_fees = st.number_input("College fees", 0, 1_000_000, 0)
    with c5:
        travel_expenses = st.number_input("Travel expenses", 0, 100000, 2000)
        groceries_utilities = st.number_input("Groceries & utilities", 0, 200000, 5000)
        other_monthly_expenses = st.number_input("Other monthly expenses", 0, 1_000_000, 2000)
    with c6:
        existing_loans = st.selectbox("Existing loans?", ["No", "Yes"])
        dependents = st.number_input("Dependents", 0, 20, 0)
        bank_balance = st.number_input("Bank balance (INR)", 0, 50_000_000, 250000, step=1000)

    st.subheader("Credit & Savings")
    credit_score = st.slider("Credit score", min_value=300, max_value=850, value=780)
    emergency_fund = st.number_input("Emergency fund (INR)", 0, 50_000_000, 150000, step=1000)

    st.subheader("Loan Requested")
    emi_scenario = st.selectbox("EMI Scenario", ["E-commerce", "Home Appliances", "Vehicle", "Personal Loan", "Education"])
    requested_amount = st.number_input("Requested loan amount (INR)", 1000, 50_000_000, 300000, step=1000)
    requested_tenure = st.number_input("Requested tenure (months)", 1, 360, 24)

    submitted = st.form_submit_button("Predict")

# ---------------- Utilities & transformers ----------------
def build_input_df():
    row = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "company_type": company_type,
        "house_type": house_type,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure
    }
    return pd.DataFrame([row])

def add_derived_features(df):
    df = df.copy()
    tenure = df['requested_tenure'].replace(0, 1)
    df['requested_monthly_equiv'] = df['requested_amount'] / tenure
    df['total_existing_emi'] = df['current_emi_amount'].fillna(0)
    df['debt_to_income'] = (df['total_existing_emi'] + df['requested_monthly_equiv']) / (df['monthly_salary'].replace(0, np.nan))
    df['essentials_expense'] = df[['school_fees','college_fees','travel_expenses','groceries_utilities']].sum(axis=1)
    df['expense_to_income'] = df['essentials_expense'] / (df['monthly_salary'].replace(0, np.nan))
    df['other_expense_to_income'] = df['other_monthly_expenses'] / (df['monthly_salary'].replace(0, np.nan))
    df['savings_buffer_ratio'] = df['emergency_fund'] / (df['monthly_salary'].replace(0, np.nan))
    df['available_balance_ratio'] = df['bank_balance'] / (df['monthly_salary'].replace(0, np.nan))
    df['has_existing_loans'] = df['existing_loans'].astype(str).str.lower().isin(['yes','y','true','1']).astype(int)
    df['is_renter'] = df['house_type'].astype(str).str.lower().str.contains('rent', na=False).astype(int)
    edu_map = {"High School": 0, "Graduate": 1, "Post Graduate": 2, "Professional": 3}
    df['education_ord'] = df['education'].map(edu_map).fillna(0).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

_normalize_re = re.compile(r'^(num__|cat__)', flags=re.IGNORECASE)
def normalize_name(n):
    if n is None:
        return ""
    s = str(n)
    s = _normalize_re.sub("", s)
    s = s.lower()
    s = re.sub(r'[^0-9a-z]+', '_', s)
    s = re.sub(r'__+', '_', s)
    return s.strip('_')

def transform_for_model(df_input):
    if preprocessor is None:
        st.error("Preprocessor not loaded.")
        return None

    numeric_features = [
        'age','monthly_salary','monthly_rent','family_size','dependents',
        'school_fees','college_fees','travel_expenses','groceries_utilities',
        'other_monthly_expenses','current_emi_amount','credit_score','bank_balance','emergency_fund',
        'requested_amount','requested_tenure','requested_monthly_equiv','debt_to_income','expense_to_income',
        'other_expense_to_income','savings_buffer_ratio','available_balance_ratio','total_existing_emi'
    ]
    categorical_features = [c for c in ['gender','marital_status','education','employment_type','company_type','house_type','emi_scenario'] if c in df_input.columns]
    cols_to_transform = [c for c in numeric_features + categorical_features if c in df_input.columns]

    try:
        X_trans = preprocessor.transform(df_input[cols_to_transform])
    except Exception as e:
        st.error(f"Preprocessor transform failed: {e}")
        return None

    try:
        import scipy.sparse as sp
        if sp.issparse(X_trans):
            X_trans = X_trans.toarray()
    except Exception:
        pass

    if X_trans.ndim == 1:
        X_trans = X_trans.reshape(1, -1)

    pre_names = None
    try:
        pre_names = list(preprocessor.get_feature_names_out(cols_to_transform))
    except Exception:
        try:
            pre_names = list(preprocessor.get_feature_names_out())
        except Exception:
            pre_names = None

    if not EXPECTED_FEATURE_NAMES:
        if pre_names is not None and len(pre_names) == X_trans.shape[1]:
            return pd.DataFrame(X_trans, columns=pre_names)
        else:
            return pd.DataFrame(X_trans, columns=[f"f{i}" for i in range(X_trans.shape[1])])

    expected = [str(x) for x in EXPECTED_FEATURE_NAMES]
    expected_norm = [normalize_name(x) for x in expected]

    if pre_names is not None and len(pre_names) == X_trans.shape[1]:
        pre_norm = [normalize_name(x) for x in pre_names]
    else:
        pre_norm = [f"trans_{i}" for i in range(X_trans.shape[1])]

    pre_map = {n: idx for idx, n in enumerate(pre_norm)}
    rows = X_trans.shape[0]
    final = np.zeros((rows, len(expected)), dtype=X_trans.dtype)

    unmatched = []
    for j, en in enumerate(expected_norm):
        if en in pre_map:
            final[:, j] = X_trans[:, pre_map[en]]
        else:
            found = False
            for pn_idx, pn in enumerate(pre_norm):
                if en and pn and (en in pn or pn in en):
                    final[:, j] = X_trans[:, pn_idx]
                    found = True
                    break
            if not found:
                unmatched.append(expected[j])

    X_df = pd.DataFrame(final, columns=expected)

    for col in ['education_ord','is_renter','has_existing_loans']:
        if col not in X_df.columns:
            X_df[col] = df_input[col].values if col in df_input.columns else 0

    if unmatched:
        st.info(f"Note: {len(unmatched)} expected features were not found and were filled with zeros. Examples: {unmatched[:6]}")

    return X_df

# ---------------- EMI helper ----------------
def calc_approx_emi(principal, tenure_months, annual_rate=0.10):
    if tenure_months <= 0:
        return float(principal)
    r = annual_rate / 12.0
    P = float(principal)
    n = float(tenure_months)
    try:
        emi = (P * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    except Exception:
        emi = P / n
    return emi

# ---------------- Main predict flow ----------------
if submitted:
    # basic input validation
    if requested_tenure <= 0:
        st.error("Requested tenure must be >= 1 month.")
    else:
        raw = build_input_df()
        df = add_derived_features(raw)

        st.subheader("Input Summary")
        st.dataframe(df.T, height=300)

        X_model = transform_for_model(df)
        if X_model is None:
            st.error("Failed to construct model features.")
        else:
            if DEBUG_SHOW_FEATURES:
                st.subheader("Debug: Features passed to model (X_model)")
                st.write("Columns count:", len(X_model.columns))
                st.dataframe(X_model.T)

            st.subheader("Predictions (model + enforced business rules)")

            final_label = None
            reg_pred = None

            # ---------- classifier (model outputs for reference) ----------
            model_probs = None
            model_readable = None
            model_eligible_prob = 0.0
            if clf is not None:
                try:
                    proba = None
                    try:
                        proba = clf.predict_proba(X_model.fillna(0))
                    except Exception:
                        proba = None

                    pred_raw = clf.predict(X_model.fillna(0))[0]

                    classes = list(getattr(clf, "classes_", []))
                    if classes and all(isinstance(c, str) for c in classes):
                        model_readable = [str(c) for c in classes]
                    else:
                        default_map = {0: "Eligible", 1: "Not Eligible", 2: "High Risk"}
                        model_readable = [default_map.get(int(c), str(c)) for c in classes] if classes else ["0","1"]

                    if proba is not None:
                        model_probs = proba[0]
                        prob_table = pd.DataFrame({"class": model_readable, "probability": model_probs})
                        prob_table["probability_pct"] = (prob_table["probability"] * 100).round(2).astype(str) + "%"
                        st.table(prob_table[["class", "probability_pct"]])

                    # compute eligible prob for reference
                    try:
                        eligible_idx = [i for i, c in enumerate(model_readable) if "eligible" in str(c).lower()]
                        model_eligible_prob = float(model_probs[eligible_idx[0]]) if (proba is not None and eligible_idx) else 0.0
                    except Exception:
                        model_eligible_prob = 0.0

                except Exception as e:
                    st.error(f"Classifier prediction error (for reference): {e}")
                    model_eligible_prob = 0.0
            else:
                st.warning("No classifier found. Model-based eligibility skipped (rules will still apply).")

            # ------------------ RUN SANITY CHECKS (ENFORCED) ------------------
            # numeric inputs (explicitly show these so you can verify)
            monthly_salary_val = float(df['monthly_salary'].iloc[0]) if 'monthly_salary' in df.columns else 0.0
            requested_amount_val = float(df['requested_amount'].iloc[0]) if 'requested_amount' in df.columns else 0.0
            requested_monthly_equiv_val = float(df['requested_monthly_equiv'].iloc[0]) if 'requested_monthly_equiv' in df.columns else (requested_amount_val / max(1, float(df['requested_tenure'].iloc[0]) if 'requested_tenure' in df.columns else 1))
            dti_val = float(df['debt_to_income'].iloc[0]) if 'debt_to_income' in df.columns else None

            st.write("**Sanity-check inputs (visible)**")
            st.write(f"- Monthly salary: ₹{monthly_salary_val:,.0f}")
            st.write(f"- Requested amount: ₹{requested_amount_val:,.0f}")
            st.write(f"- Requested monthly equiv (simple): ₹{requested_monthly_equiv_val:,.0f}")
            if dti_val is not None:
                st.write(f"- Debt-to-Income (approx): {dti_val:.2f}")

            # business-rule params
            DTI_HARD_THRESHOLD = 0.6
            EMI_SHARE_THRESHOLD = 0.60
            LOW_SALARY_LIMIT = 10000
            HIGH_LOAN_MULTIPLIER = 200
            HIGH_LOAN_LIMIT = 1_000_000

            override_reasons = []

            # legacy rules
            if dti_val is not None and dti_val > DTI_HARD_THRESHOLD:
                override_reasons.append(f"Debt-to-Income ratio is {dti_val:.2f} (>{DTI_HARD_THRESHOLD:.2f}).")
            if monthly_salary_val > 0 and requested_monthly_equiv_val > monthly_salary_val * EMI_SHARE_THRESHOLD:
                percent = (requested_monthly_equiv_val / (monthly_salary_val + 1e-9)) * 100
                override_reasons.append(f"Requested EMI ≈ ₹{requested_monthly_equiv_val:,.0f} (~{percent:.0f}% of salary) exceeds {EMI_SHARE_THRESHOLD*100:.0f}% of income.")
            if monthly_salary_val < LOW_SALARY_LIMIT and requested_amount_val > HIGH_LOAN_LIMIT:
                override_reasons.append(f"Monthly salary ₹{monthly_salary_val:,.0f} is too low for a loan of ₹{requested_amount_val:,.0f}.")

            # strong checks: disproportionate loan multiplier + estimated EMI
            try:
                if monthly_salary_val > 0 and requested_amount_val > (monthly_salary_val * HIGH_LOAN_MULTIPLIER):
                    override_reasons.append(f"Requested loan ₹{requested_amount_val:,.0f} is disproportionate to monthly salary ₹{monthly_salary_val:,.0f} (>{HIGH_LOAN_MULTIPLIER}x).")

                est_emi = calc_approx_emi(requested_amount_val, int(max(1, requested_tenure)), annual_rate=0.10)
                st.write(f"- Estimated realistic EMI (10% p.a.): ₹{est_emi:,.0f}/mo")
                if monthly_salary_val > 0 and est_emi > monthly_salary_val * EMI_SHARE_THRESHOLD:
                    override_reasons.append(f"Estimated realistic EMI ₹{est_emi:,.0f}/mo exceeds {EMI_SHARE_THRESHOLD*100:.0f}% of salary (₹{monthly_salary_val:,.0f}).")
            except Exception as e:
                st.write("Sanity-check EMI calc failed:", e)

            # Enforce overrides BEFORE showing model label
            if override_reasons:
                final_label = "Not Eligible"
                st.error("❌ BUSINESS RULE OVERRIDE: Application marked NOT ELIGIBLE")
                st.markdown("**Reason(s) (enforced):**")
                for r in override_reasons:
                    st.write(f"- {r}")
                if model_eligible_prob is not None:
                    st.info(f"Model Eligible probability (for reference only): {model_eligible_prob*100:.2f}%")
            else:
                # No overrides -> use model thresholds (if available) to set label
                if model_eligible_prob >= 0.50:
                    final_label = "Eligible"
                    st.success(f"✅ EMI Eligibility: **{final_label}** — Model confidence for Eligible is {model_eligible_prob*100:.2f}% (≥50%).")
                elif model_eligible_prob >= 0.35:
                    final_label = "Borderline"
                    st.warning(f"⚠️ EMI Eligibility: **{final_label}** — Eligible probability {model_eligible_prob*100:.2f}%.")
                else:
                    final_label = "Not Eligible"
                    st.error(f"❌ EMI Eligibility: **{final_label}** — Model confidence for Eligible only {model_eligible_prob*100:.2f}% (<35%).")

            # ---------- regressor ----------
            if reg is not None:
                try:
                    reg_pred = reg.predict(X_model.fillna(0))[0]
                    st.markdown(f"**Estimated Max Monthly EMI (₹) by regressor:**  {reg_pred:,.0f}")
                    dti = df['debt_to_income'].iloc[0] if 'debt_to_income' in df.columns else None
                    if dti is not None:
                        st.write(f"Debt-to-Income ratio (approx): {dti:.2f}")
                        if dti > 0.5:
                            st.error("Warning: High DTI (>0.5) — applicant may be risky.")
                        elif dti > 0.35:
                            st.warning("Moderate DTI (0.35-0.5) — caution recommended.")
                        else:
                            st.success("Healthy DTI (<0.35).")
                except Exception as e:
                    st.error(f"Regressor prediction error: {e}")
            else:
                st.warning("No regressor found. Skipping EMI prediction.")

            # Save audit
            if st.button("Save input & prediction (audit)"):
                out = df.copy()
                out['pred_label'] = final_label
                out['pred_max_monthly_emi'] = locals().get('reg_pred', "")
                fname = RESULT_DIR / f"prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                try:
                    out.to_csv(fname, index=False)
                    st.success(f"Saved to {fname}")
                    with open(fname, "rb") as f:
                        st.download_button("Download audit CSV", f, file_name=fname.name, mime="text/csv")
                except Exception as e:
                    st.error(f"Failed to save audit: {e}")

# footer info
st.sidebar.markdown("---")
st.sidebar.write("Preprocessor file:", artifacts['paths'].get("preprocessor"))
st.sidebar.write("Classifier file:", artifacts['paths'].get("clf"))
st.sidebar.write("Regressor file:", artifacts['paths'].get("reg"))
st.sidebar.caption("Tip: enable 'Show debug features (X_model)' for troubleshooting feature alignment.")
