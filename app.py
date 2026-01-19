import streamlit as st
import cv2
import tempfile
import os
import supervision as sv
from inference.models.utils import get_roboflow_model

# ==========================================
# 1. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Driven 13 | Professional ROI", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { 
        background-color: #1e2130; 
        padding: 20px; 
        border-radius: 15px; 
        border: 1px solid #00ffcc; 
        box-shadow: 0px 4px 10px rgba(0, 255, 204, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: Professional Sponsorship Auditor")
st.subheader("Advanced Media Valuation Engine")

# ==========================================
# 2. SIDEBAR - VALUATION CONTROLS
# ==========================================
with st.sidebar:
    st.header("1. Authentication")
    api_key = st.text_input("Roboflow API Key", value="fR501eGEeNlUzVcE3uNj", type="password")
    model_id = st.text_input("Model ID", value="driven-13-aramco-roi/9")
    
    st.header("2. Market Valuation")
    valuation_type = st.selectbox("Valuation Benchmark", ["TV Broadcast (30s Slot)", "Social Media (CPM)"])
    
    if valuation_type == "TV Broadcast (30s Slot)":
        base_rate = st.number_input("Ad Rate ($)", value=50000)
    else:
        base_rate = st.number_input("CPM Rate
