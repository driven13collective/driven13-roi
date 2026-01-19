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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Aramco_Logo.svg/1200px-Aramco_Logo.svg.png", width=150)
    st.header("Audit Settings")
    api_key = st.text_input("Roboflow API Key", value="fR501eGEeNlUzVcE3uNj", type="password")
    model_id = st.text_input("Model ID", value="driven-13-aramco-roi/9")
    
    st.subheader("Valuation Logic")
    ad_rate = st.number_input("Ad Rate ($ per 30s Slot)", value=50000)
    
    st.info("Version 9 Engine Active | ByteTrack Enabled")

# ==========================================
# 3. UPLOAD AREA
# ==========================================
uploaded_file = st.file_uploader("Drag and drop race footage here...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    st.success("✅ Footage loaded and ready for audit.")
    
    if st.button("🚀 START AI AUDIT"):
        try:
            with st.spinner("Initializing AI Engine..."):
                model = get_roboflow_model(model_id=model_id, api_key=api_key)
                tracker = sv.ByteTrack()
                video_info = sv.VideoInfo.from_video_path(tfile.name)
            
            stats = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            def analyze_frame(frame, index):
                results = model.infer(frame, confidence=0.40)[0]
                detections = sv.Detections.from_inference(results)
                detections = tracker.update_with_detections(detections)
                
                for tid in detections.tracker_id:
                    if tid not in stats:
                        stats[tid] = 0
                    stats[tid] += 1
                
                progress = int((index / video_info.total_frames) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Auditing Frame {index} of {video_info.total_frames}...")
                return frame

            # Process
            sv.process_video(source_path=tfile.name, target_path="temp_out.mp4", callback=analyze_frame)

            # ==========================================
            # 4. RESULTS DASHBOARD
            # ==========================================
            st.divider()
            st.header("📊 Certified Sponsorship Report")
            
            total_frames = sum(stats.values())
            total_secs = total_frames / video_info.fps
            unique_appearances = len(stats)
            emv = (total_secs / 30) * ad_rate

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Airtime", f"{total_secs:.2f}s")
            with col2:
                st.metric("Unique Sightings", f"{unique_appearances}")
            with col3:
                st.metric("Earned Media Value", f"${emv:,.2f}")

            st.subheader("Executive Summary")
            st.write(f"This footage generated ${emv:,.2f} in marketing value based on {total_secs:.2f} seconds of exposure.")
            st.balloons()

        except Exception as e:
            st.error(f"Audit Error: {e}")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Driven 13 Collective | Hawaii HQ Development Build")
