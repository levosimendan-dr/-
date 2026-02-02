import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration (Standard English)
# ==========================================
st.set_page_config(page_title="HFpEF w/ CKD Risk Calculator", layout="wide")

# Set font to standard sans-serif (No more Chinese font issues)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus signs display correctly

# English Name Mapping
NAME_MAPPING = {
    "egfr": "eGFR (mL/min/1.73m2)",  # Used 'm2' to avoid unicode issues
    "E_over_e_prime": "E/e' Ratio",
    "d_dimer": "D-Dimer (mg/L)",
    "serum_creatinine": "Serum Creatinine (umol/L)",
    "nyha_class": "NYHA Class",
    "serum_uric_acid": "Uric Acid (umol/L)",
    "blood_urea_nitrogen": "BUN (mmol/L)",
    "nt_probnp": "NT-proBNP (pg/mL)",
    "homocysteine": "Homocysteine (umol/L)",
    "hs_crp": "hs-CRP (mg/L)"
}

# Reverse mapping for lookups
REVERSE_MAPPING = {v: k for k, v in NAME_MAPPING.items()}


# ==========================================
# 2. Load Resources
# ==========================================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('final_model.pkl')
        background_data = joblib.load('train_data_sample.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, background_data, feature_names
    except FileNotFoundError:
        st.error(
            "Error: Model files not found. Please ensure 'final_model.pkl', 'train_data_sample.pkl', and 'feature_names.pkl' are in the directory.")
        return None, None, None


model, X_train_bg, model_features = load_resources()

# ==========================================
# 3. Sidebar: Patient Data Input
# ==========================================
st.sidebar.header("🏥 Patient Data Input")

input_dict = {}

if model_features:
    # Use English labels for inputs

    st.sidebar.subheader("Renal Function")
    input_dict['egfr'] = st.sidebar.number_input(NAME_MAPPING['egfr'], min_value=5.0, max_value=150.0, value=30.0,
                                                 step=1.0)
    input_dict['serum_creatinine'] = st.sidebar.number_input(NAME_MAPPING['serum_creatinine'], min_value=20.0,
                                                             max_value=1000.0, value=150.0)
    input_dict['blood_urea_nitrogen'] = st.sidebar.number_input(NAME_MAPPING['blood_urea_nitrogen'], min_value=1.0,
                                                                max_value=50.0, value=10.0)

    st.sidebar.subheader("Cardiac Function")
    input_dict['E_over_e_prime'] = st.sidebar.number_input(NAME_MAPPING['E_over_e_prime'], min_value=1.0,
                                                           max_value=50.0, value=15.0)
    input_dict['nt_probnp'] = st.sidebar.number_input(NAME_MAPPING['nt_probnp'], min_value=10.0, max_value=35000.0,
                                                      value=2000.0, step=100.0)
    input_dict['nyha_class'] = st.sidebar.selectbox(NAME_MAPPING['nyha_class'], options=[1, 2, 3, 4], index=2)

    st.sidebar.subheader("Biomarkers")
    input_dict['d_dimer'] = st.sidebar.number_input(NAME_MAPPING['d_dimer'], min_value=0.0, max_value=20.0, value=0.5,
                                                    step=0.1)
    input_dict['serum_uric_acid'] = st.sidebar.number_input(NAME_MAPPING['serum_uric_acid'], min_value=50.0,
                                                            max_value=1000.0, value=400.0)
    input_dict['homocysteine'] = st.sidebar.number_input(NAME_MAPPING['homocysteine'], min_value=1.0, max_value=100.0,
                                                         value=15.0)
    input_dict['hs_crp'] = st.sidebar.number_input(NAME_MAPPING['hs_crp'], min_value=0.0, max_value=200.0, value=5.0)

# ==========================================
# 4. Main Interface
# ==========================================
st.title("❤️ Readmission Risk Calculator")
st.markdown("**Target Population:** HFpEF patients with comorbid CKD")
st.markdown("---")

if st.button("🚀 Calculate Risk", type="primary"):
    if model is None:
        st.stop()

    # 1. Prepare Input Data
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_features]  # Ensure correct column order

    # 2. Prediction
    try:
        prob = model.predict_proba(input_df)[:, 1][0]
    except:
        st.error("Model prediction failed. Please check input data.")
        st.stop()

    # 3. Display Results
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Prediction Result")
        risk_percentage = prob * 100

        # Dynamic Color Logic
        if risk_percentage < 30:
            color = "#28a745"  # Green
            level = "Low Risk"
        elif risk_percentage < 70:
            color = "#ffc107"  # Orange/Yellow
            level = "Intermediate Risk"
        else:
            color = "#dc3545"  # Red
            level = "High Risk"

        st.markdown(f"""
        <div style="text-align: center; border: 3px solid {color}; padding: 25px; border-radius: 15px; background-color: #f8f9fa;">
            <h4 style="color: #6c757d; margin: 0;">1-Year Readmission Probability</h4>
            <h1 style="color: {color}; font-size: 60px; margin: 10px 0;">{risk_percentage:.1f}%</h1>
            <h3 style="color: {color}; margin: 0;">{level}</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Risk Attribution (SHAP Waterfall)")
        with st.spinner("Analyzing feature contributions..."):
            # SHAP Calculation Logic
            estimator = model
            if hasattr(estimator, 'calibrated_classifiers_'):
                estimator = estimator.calibrated_classifiers_[0].estimator

            if hasattr(estimator, 'named_steps'):
                scaler = estimator.named_steps['scaler']
                clf = estimator.named_steps['clf']
                X_bg_scaled = scaler.transform(X_train_bg)
                X_input_scaled = scaler.transform(input_df)
                explainer = shap.LinearExplainer(clf, X_bg_scaled, feature_perturbation="interventional")
                shap_values = explainer(X_input_scaled)
            else:
                explainer = shap.LinearExplainer(estimator, X_train_bg, feature_perturbation="interventional")
                shap_values = explainer(input_df)

            # Assign English names to SHAP values
            shap_values.feature_names = [NAME_MAPPING.get(c, c) for c in model_features]

            # Plotting Waterfall
            # Use bbox_inches='tight' and larger figure size
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)

    # AI Interpretation Text
    st.markdown("---")
    st.subheader("🤖 Interpretation")
    top_feature_idx = np.argmax(np.abs(shap_values.values[0]))
    top_feature_name = shap_values.feature_names[top_feature_idx]
    contribution = shap_values.values[0][top_feature_idx]

    direction = "increased" if contribution > 0 else "decreased"

    st.info(f"Analysis indicates that **{top_feature_name}** is the most influential factor for this patient, "
            f"which **{direction}** the risk probability by approximately **{abs(contribution) * 100:.1f}%**.")

else:
    st.info("👈 Please enter clinical parameters in the sidebar and click 'Calculate Risk'.")