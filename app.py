Here is your finalized, fully updated app.py. I’ve standardized all logic to the 0.5 (50%) threshold, made the UI fully reactive, and cleaned up the "Strategic Recommendations" to match professional industry standards.

Python

import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np

# 1. Page Config
st.set_page_config(page_title="RetentionOps v1", page_icon="🛡️", layout="wide")

# 2. Precise Layout CSS
st.markdown("""
    <style>
    .stApp { background-color: #0F172A; }
    .block-container { padding: 2rem 3rem !important; }
    
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
    }

    .card-label {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }
    .main-metric {
        color: #3B82F6;
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }
    .sub-text {
        color: #64748B;
        font-size: 0.85rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Asset Loading
@st.cache_resource
def load_assets():
    try:
        with open('churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
        return model, mappings, model.get_booster().feature_names
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

model, mappings, model_features = load_assets()

# 4. Sidebar: Control Panel
with st.sidebar:
    st.markdown("### 🛡️ RetentionOps")
    st.caption("v1.0.3 • Production Ready")
    st.divider()
    
    st.markdown("### 👤 User Profile")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (Months)", 1, 72, 24)
    
    st.markdown("### ⚙️ Services")
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    monthly = st.number_input("Monthly Revenue ($)", 18.0, 150.0, 70.0)
    tech_support = st.radio("Tech Support Plan", ["Yes", "No", "No internet service"], horizontal=True)
    
    st.divider()
    st.info("The dashboard updates automatically as you adjust parameters.")

# 5. Main Content Area
if model:
    # --- Prediction Engine (Dynamic Calculation) ---
    input_dict = {feat: [0] for feat in model_features} 
    input_dict.update({
        'tenure': [tenure], 
        'MonthlyCharges': [monthly], 
        'TotalCharges': [tenure * monthly],
        'Contract': [mappings['Contract'][contract]], 
        'InternetService': [mappings['InternetService'][internet]],
        'TechSupport': [mappings['TechSupport'][tech_support]]
    })
    input_df = pd.DataFrame(input_dict)[model_features]
    prob = float(model.predict_proba(input_df)[0][1])

    # --- Header ---
    st.markdown("### Executive Summary")
    
    # --- Grid Layout ---
    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown('<p class="card-label">Churn Risk Probability</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="main-metric">{prob:.1%}</p>', unsafe_allow_html=True)
            
            # --- STANDARDIZED 0.5 THRESHOLD LOGIC ---
            if prob >= 0.5:
                badge_color, badge_text = "#F87171", "High Risk / Action Required"
            elif prob >= 0.2:
                badge_color, badge_text = "#FBBF24", "Elevated Risk / Monitor"
            else:
                badge_color, badge_text = "#34D399", "Stable Profile"
                
            st.markdown(f'<span style="color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase;">{badge_text}</span>', unsafe_allow_html=True)
            st.markdown('<p class="sub-text">Classification threshold: 0.5</p>', unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<p class="card-label">Dynamic Feature Impact</p>', unsafe_allow_html=True)
            
            # Simulated Impact calculation for visual feedback
            attr_data = pd.DataFrame({
                'Feature': ['Tenure', 'Monthly', 'Contract', 'Support'],
                'Impact': [tenure/-72, monthly/150, 0.6 if contract == 'Month-to-month' else -0.4, 0.2 if tech_support == 'No' else -0.2]
            }).set_index('Feature')
            
            st.bar_chart(attr_data, color="#3B82F6", height=215)

    # 6. Strategic Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["💡 Strategic Recommendations", "🔬 Technical Metadata"])
    
    with t1:
        if prob >= 0.5:
            st.warning(f"**High Risk Intervention:** Customer has crossed the {prob:.1%} probability threshold. Recommend an immediate 'Retention Offer' (e.g., 12-month contract lock-in with 15% discount).")
        elif prob >= 0.2:
            st.info(f"**Preventative Monitoring:** Risk is elevated at {prob:.1%}. Recommend a proactive 'Service Health Check' email and evaluation of technical support needs.")
        else:
            st.success(f"**Growth Strategy:** Profile is stable ({prob:.1%}). Customer is a prime candidate for long-term loyalty rewards or premium service cross-selling.")

    with t2:
        st.json({
            "model_architecture": "XGBoost (Extreme Gradient Boosting)",
            "classification_threshold": 0.5,
            "probability_raw": round(prob, 4),
            "feature_vector_input": input_dict,
            "system_status": "Stable"
        })
else:
    st.error("Assets missing. Ensure 'churn_model.pkl' and 'mappings.pkl' are in the root directory.")
    
