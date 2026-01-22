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
    .stProgress > div > div > div > div { background-color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: ROI Sponsorship Auditor")

# 2. SIDEBAR & API CONFIG
with st.sidebar:
    st.header("Audit & Goal Settings")
    # Using your confirmed Private API Key
    api_key = st.text_input("Private API Key", value="3rcTmYwUyM4deHfzdLhy", type="password")
    roi_goal = st.number_input("Target ROI Goal ($)", value=5000.0)
    val_price = st.number_input("Valvoline $/sighting", value=15.0)
    aramco_price = st.number_input("Aramco $/sighting", value=12.0)
    
    if st.button("🔄 Reset Audit Data"):
        st.session_state.audit_data = {"Valvoline": 0.0, "Aramco": 0.0, "Count": {"Valvoline": 0, "Aramco": 0}}
        st.rerun()

# 3. INITIALIZE MODEL (VERSION 5)
model = None
if api_key:
    try:
        rf = Roboflow(api_key=api_key)
        # Verify project name matches your URL slug
        project = rf.workspace().project("valvoline-roi")
        # Explicitly using Version 5 (Production-V1)
        model = project.version(5).model 
        
        # Professional UI Annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        st.sidebar.success("✅ Connected: Production-V1 (v5)")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

# 4. UPLOAD & ANALYSIS
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file and model:
    # Save video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(up_file.read())
    tfile.close() 
    
    # Initialize session state for tracking
    if 'audit_data' not in st.session_state:
        st.session_state.audit_data = {"Valvoline": 0.0, "Aramco": 0.0, "Count": {"Valvoline": 0, "Aramco": 0}}

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Analysis Feed")
        frame_window = st.empty()
        goal_text = st.empty()
        goal_bar = st.progress(0)
        
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button("🚀 START ROI AUDIT"):
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                # ANALYZE 1 EVERY 6 FRAMES (Reduces API load, maintains demo quality)
                if frame_count % 6 != 0:
                    continue

                # Temp image for Roboflow Predict
                temp_img_path = f"frame_{frame_count}.jpg"
                cv2.imwrite(temp_img_path, frame)
                
                # Small wait to ensure disk write is complete
                time.sleep(0.05) 
                
                try:
                    # Run Inference on Version 5
                    results = model.predict(temp_img_path, confidence=40).json()
                    detections = sv.Detections.from_inference(results)
                    
                    # Log ROI logic
                    for pred in results['predictions']:
                        raw_class = pred['class'].lower()
                        brand = "Valvoline" if "valvoline" in raw_class else "Aramco"
                        st.session_state.audit_data[brand] += val_price if brand == "Valvoline" else aramco_price
                        st.session_state.audit_data["Count"][brand] += 1

                    # Draw Boxes and Labels
                    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                    
                    # Update Progress Bar
                    val_roi = st.session_state.audit_data["Valvoline"]
                    goal_bar.progress(min(val_roi / roi_goal, 1.0))
                    goal_text.write(f"**Valvoline Current ROI:** ${val_roi:,.2f} / ${roi_goal:,.2f}")
                    
                    # Convert BGR to RGB for Streamlit display
                    frame_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    
                except Exception as api_err:
                    st.warning(f"API Rate Limit - skipping frame... ({api_err})")
                    time.sleep(1) # Cool down
                
                # Cleanup frame
                if os.path.exists(temp_img_path): os.remove(temp_img_path)
            
            cap.release()
            if st.session_state.audit_data["Valvoline"] >= roi_goal: 
                st.balloons()
            st.success("Audit Complete!")

    with col2:
        st.subheader("Share of Voice (SOV)")
        sov_df = pd.DataFrame({
            "Brand": ["Valvoline", "Aramco"],
            "Value": [st.session_state.audit_data["Valvoline"], st.session_state.audit_data["Aramco"]]
        })
        
        if sov_df["Value"].sum() > 0:
            fig = px.pie(sov_df, values='Value', names='Brand', hole=0.5,
                         color='Brand', color_discrete_map={'Valvoline': '#CC0000', 'Aramco': '#007A33'})
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Valvoline ROI", f"${st.session_state.audit_data['Valvoline']:,.2f}")
        st.metric("Aramco ROI", f"${st.session_state.audit_data['Aramco']:,.2f}")
        st.divider()
        st.write(f"Total Detections: {sum(st.session_state.audit_data['Count'].values())}")

st.caption("Driven 13 Collective | Production-V1 Engine (v5)")


