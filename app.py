import streamlit as st
import cv2
import tempfile
import supervision as sv
import pandas as pd
import plotly.express as px
from roboflow import Roboflow
import numpy as np
import os

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | ROI Auditor", page_icon="🏎️", layout="wide")

# BEAUTIFICATION
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
        st.session_state.audit_data = {
            "Valvoline": 0.0, 
            "Aramco": 0.0, 
            "Count": {"Valvoline": 0, "Aramco": 0}
        }
        st.rerun()

# 3. INITIALIZE AI & ANNOTATORS
if api_key:
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("valvoline-roi")
        # Pulling Production-V1
        model = project.version(1).model 
        
        # Professional UI Annotators
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    except Exception as e:
        st.error(f"Connection failed: {e}")

# 4. UPLOAD & ANALYSIS
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(up_file.read())
    
    if 'audit_data' not in st.session_state:
        st.session_state.audit_data = {
            "Valvoline": 0.0, 
            "Aramco": 0.0, 
            "Count": {"Valvoline": 0, "Aramco": 0}
        }

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Analysis Feed")
        frame_window = st.empty()
        goal_text = st.empty()
        goal_bar = st.progress(0)
        
        cap = cv2.VideoCapture(tfile.name)

        if st.button("🚀 START ROI AUDIT"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # --- THE FIX ---
                # Save frame to disk so Roboflow can read it correctly
                temp_img_path = "current_frame.jpg"
                cv2.imwrite(temp_img_path, frame)
                
                # Inference on the saved file
                results = model.predict(temp_img_path, confidence=40).json()
                detections = sv.Detections.from_inference(results)
                
                # Process Detections
                for pred in results['predictions']:
                    raw_class = pred['class'].lower()
                    if "valvoline" in raw_class:
                        brand = "Valvoline"
                        price = val_price
                    elif "aramco" in raw_class:
                        brand = "Aramco"
                        price = aramco_price
                    else:
                        continue # Skip unidentified logos
                    
                    st.session_state.audit_data[brand] += price
                    st.session_state.audit_data["Count"][brand] += 1

                # DRAW ANNOTATIONS
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

                # UPDATE PROGRESS
                current_val_roi = st.session_state.audit_data["Valvoline"]
                progress_pct = min(current_val_roi / roi_goal, 1.0)
                goal_bar.progress(progress_pct)
                goal_text.write(f"**Valvoline ROI:** ${current_val_roi:,.2f} / ${roi_goal:,.2f}")
                
                # DISPLAY FRAME
                frame_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            
            if st.session_state.audit_data["Valvoline"] >= roi_goal: 
                st.balloons()
            
            cap.release()
            # Cleanup temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    with col2:
        st.subheader("Share of Voice (SOV)")
        
        # Data for Pie Chart
        sov_data = pd.DataFrame({
            "Brand": ["Valvoline", "Aramco"],
            "Money": [st.session_state.audit_data["Valvoline"], st.session_state.audit_data["Aramco"]]
        })
        
        # Donut Chart with Official Brand Colors
        if sov_data["Money"].sum() > 0:
            fig = px.pie(
                sov_data, 
                values='Money', 
                names='Brand', 
                hole=0.5,
                color='Brand',
                color_discrete_map={'Valvoline': '#CC0000', 'Aramco': '#007A33'}
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting detections...")

        # Financial Metrics
        st.metric("Valvoline Total", f"${st.session_state.audit_data['Valvoline']:,.2f}")
        st.metric("Aramco Total", f"${st.session_state.audit_data['Aramco']:,.2f}")
        
        st.divider()
        st.write(f"Total Logos Detected: {sum(st.session_state.audit_data['Count'].values())}")

st.caption("Driven 13 Collective | Production-V1 ROI Engine")

