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
    
    /* This targets Streamlit's native containers to look like cards */
    [data-testid="stVerticalBlockBorderWrapper"] > div > div {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
    }

    /* Formatting the text inside our 'cards' */
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
    except:
        return None, None, None

model, mappings, model_features = load_assets()

# 4. Sidebar: Control Panel
with st.sidebar:
    st.markdown("### 🛡️ RetentionOps")
    st.caption("v1.0.2 • System Active")
    st.divider()
    
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (Months)", 1, 72, 24)
    internet = st.selectbox("Service Tier", ["Fiber optic", "DSL", "No"])
    monthly = st.number_input("Monthly Revenue ($)", 18.0, 150.0, 70.0)
    tech_support = st.radio("Support Plan", ["Yes", "No", "No internet service"], horizontal=True)
    
    st.divider()
    predict_btn = st.button("RUN INFERENCE", type="primary", use_container_width=True)

# 5. Main Content Area
if model:
    # --- Prediction Logic ---
    input_dict = {feat: [0] for feat in model_features} 
    input_dict.update({
        'tenure': [tenure], 'MonthlyCharges': [monthly], 'TotalCharges': [tenure * monthly],
        'Contract': [mappings['Contract'][contract]], 'InternetService': [mappings['InternetService'][internet]],
        'TechSupport': [mappings['TechSupport'][tech_support]]
    })
    input_df = pd.DataFrame(input_dict)[model_features]
    prob = float(model.predict_proba(input_df)[0][1])

    # --- Header ---
    st.markdown("### Executive Summary")
    
    # --- Grid Layout using st.container for the "Box" look ---
    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        with st.container(border=True): # This 'border=True' works with our CSS
            st.markdown('<p class="card-label">Churn Risk Probability</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="main-metric">{prob:.1%}</p>', unsafe_allow_html=True)
            
            badge_color = "#F87171" if prob > 0.5 else "#34D399"
            badge_text = "Action Required" if prob > 0.5 else "Stable"
            st.markdown(f'<span style="color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase;">{badge_text}</span>', unsafe_allow_html=True)
            st.markdown('<p class="sub-text">Inference result based on session vector.</p>', unsafe_allow_html=True)

    with col_right:
        with st.container(border=True):
            st.markdown('<p class="card-label">Local Feature Importance</p>', unsafe_allow_html=True)
            
            # Prepare chart data
            attr_data = pd.DataFrame({
                'Feature': ['Tenure', 'Monthly', 'Contract', 'Support'],
                'Impact': [tenure/-72, monthly/150, 0.6 if contract == 'Month-to-month' else -0.4, 0.2 if tech_support == 'No' else -0.2]
            }).set_index('Feature')
            
            # This bar chart will now be forced inside the container styling
            st.bar_chart(attr_data, color="#3B82F6", height=215)

    # 6. Bottom Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2 = st.tabs(["💡 Recommendations", "🔬 Technical Metadata"])
    
    with t1:
        if prob > 0.5:
            st.warning("**Retention Strategy:** High-risk customer. Intervention recommended.")
        else:
            st.success("**Retention Strategy:** Stable customer profile.")

    with t2:
        st.json({"Pipeline": "XGBoost-v1", "Features": model_features})

else:
    st.error("Assets missing. Ensure 'churn_model.pkl' and 'mappings.pkl' are present.")
    
