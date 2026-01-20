import streamlit as st
import cv2
import tempfile
import supervision as sv
import pandas as pd
import plotly.express as px
from roboflow import Roboflow

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | ROI Auditor", page_icon="🏎️", layout="wide")
st.markdown("""
    <style>
    .stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #00ffcc;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: ROI Sponsorship Auditor")

# 2. CONFIGURATION (Sidebar)
with st.sidebar:
    st.header("Audit Settings")
    api_key = st.text_input("API Key", value="3rcTmYwUyM4deHfzdLhy", type="password")
    model_id = st.text_input("Model ID", value="valvoline-roi/1")
    val_price = st.number_input("Valvoline $/sighting", value=15.0)
    comp_price = st.number_input("Competitor $/sighting", value=10.0)

# 3. INITIALIZE AI
rf = Roboflow(api_key=api_key)
# Parsing project name from model_id (valvoline-roi/1 -> valvoline-roi)
project_name = model_id.split('/')[0]
project = rf.workspace().project(project_name)
model = project.version(1).model 
tracker = sv.ByteTrack()

up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(up_file.read())
    
    # Session State to hold data
    if 'audit_data' not in st.session_state:
        st.session_state.audit_data = {
            "Valvoline": {"money": 0.0, "sightings": 0, "quality_sum": 0.0},
            "Competitor": {"money": 0.0, "sightings": 0, "quality_sum": 0.0}
        }

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Analysis")
        frame_window = st.empty()
        cap = cv2.VideoCapture(tfile.name)
        v_info = sv.VideoInfo.from_video_path(tfile.name)

        if st.button("🚀 RUN ROI AUDIT"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # AI Inference
                results = model.predict(frame, confidence=40).json()
                
                for pred in results['predictions']:
                    label = "Valvoline" if "valvoline" in pred['class'].lower() else "Competitor"
                    
                    # --- MEDIA QUALITY SCORE LOGIC ---
                    # Quality = (Size of Logo relative to screen) + (AI Confidence)
                    area_score = (pred['width'] * pred['height']) / (v_info.width * v_info.height)
                    quality_score = (area_score * 100) + (pred['confidence'] * 0.5)
                    quality_score = min(quality_score * 10, 100) # Scale to 0-100
                    
                    st.session_state.audit_data[label]["sightings"] += 1
                    st.session_state.audit_data[label]["money"] += val_price if label == "Valvoline" else comp_price
                    st.session_state.audit_data[label]["quality_sum"] += quality_score

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame_rgb)
            cap.release()

    with col2:
        st.subheader("Financial Performance")
        
        # Calculate Averages for Report
        report_list = []
        for brand, stats in st.session_state.audit_data.items():
            avg_q = stats["quality_sum"] / stats["sightings"] if stats["sightings"] > 0 else 0
            report_list.append({
                "Brand": brand,
                "Money": stats["money"],
                "Sightings": stats["sightings"],
                "Media Quality Score": round(avg_q, 1)
            })
        
        df_roi = pd.DataFrame(report_list)

        # Pie Chart
        fig = px.pie(df_roi, values='Money', names='Brand', hole=0.4,
                     color_discrete_sequence=['#CC0000', '#003366'])
        fig.update_traces(texttemplate="$%{value:,.0f}<br>%{percent}")
        st.plotly_chart(fig, use_container_width=True)

        # Show Quality Score Metric
        for _, row in df_roi.iterrows():
            st.metric(f"{row['Brand']} Quality", f"{row['Media Quality Score']}%")

        # 4. DOWNLOAD DRIVEN 13 ROI REPORT
        st.divider()
        csv = df_roi.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Driven 13 ROI Report",
            data=csv,
            file_name='Driven13_ROI_Report.csv',
            mime='text/csv',
        )

st.caption("Driven 13 Collective | Sponsorship Auditor V12.0")
