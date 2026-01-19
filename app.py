import streamlit as st
import cv2
import tempfile
import os
import supervision as sv
from inference.models.utils import get_roboflow_model

# ==========================================
# 1. PAGE SETUP & DESIGN
# ==========================================
st.set_page_config(page_title="Driven 13 | Sponsorship ROI", page_icon="🏎️", layout="wide")

# Custom CSS for a professional dark racing theme
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
    [data-testid="stMetricValue"] { color: #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: AI Sponsorship Auditor")
st.markdown("### Turning Raw Racing Footage into Earned Media Value (EMV)")
st.divider()

# ==========================================
# 2. SIDEBAR - CONTROL PANEL
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Aramco_Logo.svg/1200px-Aramco_Logo.svg.png", wi