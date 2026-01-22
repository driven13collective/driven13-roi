import streamlit as st
import cv2
import tempfile
import supervision as sv
import pandas as pd
import plotly.express as px
from roboflow import Roboflow
import numpy as np
import os
import time

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | ROI Auditor", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #00ffcc; color: white;}
    </style>
    """, unsafe_allow_html=True)
st.title("🏎️ Driven 13 Collective: ROI Sponsorship Auditor")

# 2. SIDEBAR CONFIG
with st.sidebar:
    st.header("Audit & Goal Settings")
    api_key = st.text_input("API Key", value="3rcTmYwUyM4deHfzdLhy", type="password")
    roi_goal = st.number_input("Target ROI Goal ($)", value=5000.0)
    val_price = st.number_input("Valvoline $/sighting", value=15.0)
    aramco_price = st.number_input("Aramco $/sighting", value=12.0)
    
    if st.button("🔄 Reset Audit Data"):
        st.session_state.audit_data = {"Valvoline": 0.0, "Aramco": 0.0, "Count": {"Valvoline": 0, "Aramco": 0}}
        st.rerun()

# 3. INITIALIZE AI & ANNOTATORS
if api_key:
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("valvoline-roi")
        model = project.version(1).model 
        
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    except Exception as e:
        st.error(f"Connection failed: {e}")

# 4. UPLOAD & ANALYSIS
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(up_file.read())
    tfile.close() # Close to ensure it's written
    
    if 'audit_data' not in st.session_state:
        st.session_state.audit_data = {"Valvoline": 0.0, "Aramco": 0.0, "Count": {"Valvoline": 0, "Aramco": 0}}

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Analysis Feed")
        frame_window = st.empty()
        goal_bar = st.progress(0)
        
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("🚀 START ROI AUDIT"):
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                # ONLY ANALYZE EVERY 6TH FRAME (Approx 5 frames per second)
                # This prevents "Error 429: Too Many Requests"
                if frame_count % 6 != 0:
                    continue

                temp_img_path = "current_frame.jpg"
                cv2.imwrite(temp_img_path, frame)
                
                # Tiny pause to ensure file write is finished
                time.sleep(0.05) 
                
                try:
                    results = model.predict(temp_img_path, confidence=40).json()
                    detections = sv.Detections.from_inference(results)
                    
                    for pred in results['predictions']:
                        raw_class = pred['class'].lower()
                        brand = "Valvoline" if "valvoline" in raw_class else "Aramco"
                        st.session_state.audit_data[brand] += val_price if brand == "Valvoline" else aramco_price
                        st.session_state.audit_data["Count"][brand] += 1

                    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                    frame_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    
                    val_roi = st.session_state.audit_data["Valvoline"]
                    goal_bar.progress(min(val_roi / roi_goal, 1.0))
                    
                except Exception as api_err:
                    st.warning(f"API Busy... skipping frame. ({api_err})")
                    time.sleep(1) # Wait a second if we hit a limit
            
            cap.release()
            if os.path.exists(temp_img_path): os.remove(temp_img_path)

    with col2:
        st.subheader("Share of Voice (SOV)")
        sov_df = pd.DataFrame({
            "Brand": ["Valvoline", "Aramco"],
            "Money": [st.session_state.audit_data["Valvoline"], st.session_state.audit_data["Aramco"]]
        })
        
        if sov_df["Money"].sum() > 0:
            fig = px.pie(sov_df, values='Money', names='Brand', hole=0.5,
                         color='Brand', color_discrete_map={'Valvoline': '#CC0000', 'Aramco': '#007A33'})
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Valvoline Total", f"${st.session_state.audit_data['Valvoline']:,.2f}")
        st.metric("Aramco Total", f"${st.session_state.audit_data['Aramco']:,.2f}")
        st.write(f"Detections: {sum(st.session_state.audit_data['Count'].values())}")

st.caption("Driven 13 Collective | Production-V1")

