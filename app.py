import streamlit as st
import cv2
import tempfile
import os
import supervision as sv
import pandas as pd
import plotly.express as px
from roboflow import Roboflow

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | ROI Auditor", page_icon="🏎️", layout="wide")
st.markdown("<style>.stMetric {background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #00ffcc;}</style>", unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: ROI Sponsorship Auditor")
st.subheader("Valvoline vs. Competitive Share of Voice")

# 2. SIDEBAR CONFIG
with st.sidebar:
    st.header("Settings")
    # Using your provided credentials
    api_key = st.text_input("Roboflow API Key", value="3rcTmYwUyM4deHfzdLhy", type="password")
    model_id = st.text_input("Model ID", value="valvoline-roi/1")
    val_price = st.number_input("Valvoline Value ($/sighting)", value=15.0)
    comp_price = st.number_input("Competitor Value ($/sighting)", value=10.0)

# 3. INITIALIZE MODELS
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("valvoline-roi")
model = project.version(1).model 
tracker = sv.ByteTrack()

# 4. UPLOAD
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(up_file.read())
    
    col1, col2 = st.columns([2, 1])
    
    # Storage for ROI Data
    # Using a dictionary to keep live track
    if 'brand_stats' not in st.session_state:
        st.session_state.brand_stats = {"Valvoline": {"money": 0.0, "sightings": 0}, 
                                        "Competitor": {"money": 0.0, "sightings": 0}}

    with col1:
        st.subheader("AI Analysis Feed")
        frame_window = st.empty()
        cap = cv2.VideoCapture(tfile.name)
        v_info = sv.VideoInfo.from_video_path(tfile.name)

        if st.button("🚀 START ROI AUDIT"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Inference
                results = model.predict(frame, confidence=45).json()
                detections = sv.Detections.from_inference(results)
                detections = tracker.update_with_detections(detections)
                
                # Update Stats
                for pred in results['predictions']:
                    label = "Valvoline" if "valvoline" in pred['class'].lower() else "Competitor"
                    st.session_state.brand_stats[label]["sightings"] += 1
                    price = val_price if label == "Valvoline" else comp_price
                    st.session_state.brand_stats[label]["money"] += price

                # Visuals
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb)
            cap.release()

    with col2:
        st.subheader("Financial Performance")
        
        # Create DataFrame from session stats
        df_roi = pd.DataFrame([
            {"Brand": "Valvoline", "Money": st.session_state.brand_stats["Valvoline"]["money"], "Sightings": st.session_state.brand_stats["Valvoline"]["sightings"]},
            {"Brand": "Competitor", "Money": st.session_state.brand_stats["Competitor"]["money"], "Sightings": st.session_state.brand_stats["Competitor"]["sightings"]}
        ])
        
        # Pie Chart
        fig = px.pie(df_roi, values='Money', names='Brand', hole=0.4,
                     color_discrete_sequence=['#CC0000', '#003366'])
        fig.update_traces(
            textinfo='label+percent',
            texttemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{customdata[0]} Sightings",
            customdata=df_roi[['Sightings']]
        )
        st.plotly_chart(fig, use_container_width=True)

        # 5. DOWNLOAD REPORT
        st.divider()
        st.subheader("Driven 13 ROI Report")
        
        csv = df_roi.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download ROI Report (CSV)",
            data=csv,
            file_name='Driven13_ROI_Report.csv',
            mime='text/csv',
        )

st.caption("Driven 13 Collective | ROI Sponsorship Auditor | V11.0")

