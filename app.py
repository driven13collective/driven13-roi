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
from datetime import datetime

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | ROI Auditor", page_icon="🏎️", layout="wide")

st.markdown("""
    <style>
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #00ffcc; color: white;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: ROI Sponsorship Auditor")

# 2. SIDEBAR
with st.sidebar:
    st.header("Audit Configuration")
    api_key = st.text_input("Private API Key", value="3rcTmYwUyM4deHfzdLhy", type="password")
    roi_goal = st.number_input("Target ROI Goal ($)", value=5000.0)
    val_price = st.number_input("Valvoline $/sighting", value=15.0)
    aramco_price = st.number_input("Aramco $/sighting", value=12.0)
    
    if st.button("🔄 Reset Audit"):
        st.session_state.audit_log = []
        st.session_state.audit_data = {"Valvoline": 0.0, "Aramco": 0.0, "Count": {"Valvoline": 0, "Aramco": 0}}
        st.rerun()

# 3. AI INITIALIZATION
if api_key:
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("valvoline-roi")
        model = project.version(5).model 
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    except Exception as e:
        st.error(f"AI Connection Error: {e}")

# 4. UPLOAD
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(up_file.read())
    tfile.close()
    
    if 'audit_data' not in st.session_state:
        st.session_state.audit_data = {"Valvoline": 0.0, "Aramco": 0.0, "Count": {"Valvoline": 0, "Aramco": 0}}
        st.session_state.audit_log = [] # For the CSV report

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
                # Increase skip to 10 for "Fast Mode" if 6 is still too slow
                if frame_count % 8 != 0: 
                    continue

                temp_path = "stream_frame.jpg"
                cv2.imwrite(temp_path, frame)
                
                try:
                    results = model.predict(temp_path, confidence=40).json()
                    detections = sv.Detections.from_inference(results)
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    for pred in results['predictions']:
                        brand = "Valvoline" if "valvoline" in pred['class'].lower() else "Aramco"
                        price = val_price if brand == "Valvoline" else aramco_price
                        
                        # Update State
                        st.session_state.audit_data[brand] += price
                        st.session_state.audit_data["Count"][brand] += 1
                        
                        # Log for CSV
                        st.session_state.audit_log.append({
                            "Timestamp": timestamp,
                            "Frame": frame_count,
                            "Brand": brand,
                            "Value_Generated": price
                        })

                    # Annotation & Smooth Display
                    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                    frame_window.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    
                    val_roi = st.session_state.audit_data["Valvoline"]
                    goal_bar.progress(min(val_roi / roi_goal, 1.0))
                    
                except:
                    continue 

            cap.release()
            st.success("Audit Complete!")

    with col2:
        st.subheader("Financial Report")
        
        # DONUT CHART
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

        # --- NEW: DOWNLOAD REPORT SECTION ---
        st.divider()
        if st.session_state.audit_log:
            report_df = pd.DataFrame(st.session_state.audit_log)
            csv = report_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Audit Report (CSV)",
                data=csv,
                file_name=f"driven13_roi_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
            st.write("Report contains per-second brand valuations.")

st.caption("Driven 13 Collective | ROI Engine v5.1")



