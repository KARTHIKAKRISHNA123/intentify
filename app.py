import streamlit as st
import pandas as pd
import numpy as np
import joblib

# SaaS UI Configuration
st.set_page_config(page_title="Intentify AI", page_icon="🛒", layout="centered")

# Custom CSS for a clean blue/white SaaS theme
st.markdown("""
    <style>
    .stButton>button {
        background-color: #0052CC;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛒 Intentify: Real-Time Conversion Engine")

@st.cache_resource
def load_engine():
    return joblib.load('intentify_model.pkl')

engine = load_engine()

# --- 1. RAW TELEMETRY INPUTS ---
st.sidebar.header("Live Session Telemetry")
page_values = st.sidebar.slider("PageValues Metric", 0.0, 300.0, 15.0)
prod_pages = st.sidebar.slider("Product Pages Visited", 0, 100, 12)
prod_duration = st.sidebar.slider("Product Page Time (s)", 0.0, 3000.0, 350.0)
admin_pages = st.sidebar.slider("Admin Pages", 0, 20, 2)
admin_duration = st.sidebar.slider("Admin Time (s)", 0.0, 1000.0, 50.0)
info_pages = st.sidebar.slider("Info Pages", 0, 10, 0)
info_duration = st.sidebar.slider("Info Time (s)", 0.0, 500.0, 0.0)
bounce_rate = st.sidebar.slider("Bounce Rate", 0.0, 0.2, 0.02)
exit_rate = st.sidebar.slider("Exit Rate", 0.0, 0.2, 0.04)

# Categoricals
month = st.sidebar.selectbox("Month", ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
visitor_type = st.sidebar.selectbox("Visitor", ["Returning_Visitor", "New_Visitor", "Other"])
weekend = st.sidebar.checkbox("Weekend Session")

# --- 2. MIDDLEWARE: DYNAMIC FEATURE ENGINEERING ---
# We must recreate the exact math from your Jupyter Notebook
avg_time_per_product = prod_duration / (prod_pages + 1)
total_engagement = admin_duration + info_duration + prod_duration
bounce_exit_score = bounce_rate * exit_rate
log_page_values = np.log1p(page_values)
has_page_value = int(page_values > 0)
product_page_ratio = prod_pages / (admin_pages + info_pages + prod_pages + 1)

# --- 3. PAYLOAD CONSTRUCTION ---
session_payload = pd.DataFrame([{
    'Administrative': admin_pages, 'Administrative_Duration': admin_duration,
    'Informational': info_pages, 'Informational_Duration': info_duration,
    'ProductRelated': prod_pages, 'ProductRelated_Duration': prod_duration,
    'BounceRates': bounce_rate, 'ExitRates': exit_rate,
    'SpecialDay': 0.0, 'PageValues': page_values, 'Month': month,
    'OperatingSystems': 2, 'Browser': 2, 'Region': 1, 'TrafficType': 2,
    'VisitorType': visitor_type, 'Weekend': bool(weekend),
    # Appending the dynamically engineered features
    'avg_time_per_product': avg_time_per_product,
    'total_engagement': total_engagement,
    'bounce_exit_score': bounce_exit_score,
    'log_page_values': log_page_values,
    'has_page_value': has_page_value,
    'product_page_ratio': product_page_ratio
}])

# --- 4. INFERENCE ---
if st.button("Evaluate Conversion Intent"):
    prob = engine.predict_proba(session_payload)[0][1] * 100
    pred = engine.predict(session_payload)[0]
    
    st.metric(label="Purchase Probability", value=f"{prob:.1f}%")
    
    if pred == 1:
        st.success("✅ High Intent: Trigger fast checkout flow.")
    elif prob > 30.0:
        st.warning("⚠️ Medium Intent: Offer 10% discount to secure conversion.")
    else:
        st.error("📉 Low Intent: Passive browsing.")