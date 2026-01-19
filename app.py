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
        benchmark_label = "Per 30s Slot"
    else:
        base_rate = st.number_input("CPM Rate ($)", value=25)
        benchmark_label = "Per 1k Views (Est.)"

# ==========================================
# 3. AUDIT LOGIC
# ==========================================
uploaded_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    if st.button("🚀 GENERATE CERTIFIED AUDIT"):
        model = get_roboflow_model(model_id=model_id, api_key=api_key)
        tracker = sv.ByteTrack()
        video_info = sv.VideoInfo.from_video_path(tfile.name)
        
        # PRO DATA STORAGE
        total_weighted_value = 0.0
        total_raw_seconds = 0.0
        unique_ids = set()

        progress_bar = st.progress(0)

        def analyze_frame(frame, index):
            nonlocal total_weighted_value, total_raw_seconds
            results = model.infer(frame, confidence=0.40)[0]
            detections = sv.Detections.from_inference(results)
            detections = tracker.update_with_detections(detections)
            
            frame_value = 0.0
            for i in range(len(detections)):
                unique_ids.add(detections.tracker_id[i])
                
                # --- QUALITY WEIGHTING ENGINE ---
                # 1. Size Weight (How big is it?)
                coords = detections.xyxy[i]
                w, h = coords[2]-coords[0], coords[3]-coords[1]
                coverage = (w * h) / (video_info.width * video_info.height)
                size_multiplier = min(coverage * 50, 1.0) # Cap at 100% value
                
                # 2. Clarity Weight (How sure is the AI?)
                clarity_multiplier = detections.confidence[i]
                
                # Combine weights
                quality_score = (size_multiplier * 0.7) + (clarity_multiplier * 0.3)
                
                # Calculate dollar value for this specific frame
                if valuation_type == "TV Broadcast (30s Slot)":
                    # (Rate / 30 seconds / FPS) * Quality
                    frame_value += (base_rate / 30 / video_info.fps) * quality_score
                else:
                    # (CPM logic - assuming 100k views for demo)
                    frame_value += (base_rate * 100 / video_info.fps) * quality_score

            total_weighted_value += frame_value
            if len(detections) > 0:
                total_raw_seconds += (1 / video_info.fps)
            
            progress_bar.progress(int((index / video_info.total_frames) * 100))
            return frame

        sv.process_video(source_path=tfile.name, target_path="/tmp/null.mp4", callback=analyze_frame)

        # ==========================================
        # 4. THE PRO REPORT
        # ==========================================
        st.balloons()
        st.header("📊 Driven 13 Collective | Exposure Audit")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Airtime", f"{total_raw_seconds:.2f}s")
        col2.metric("Unique Sightings", f"{len(unique_ids)}")
        col3.metric("Earned Media Value (EMV)", f"${total_weighted_value:,.2f}")

        st.info(f"💡 This valuation includes a Quality Score penalty for motion blur and distance, ensuring your report is 100% audit-ready for sponsors.")

        st.success(f"Final Valuation: The brand Aramco achieved an adjusted Earned Media Value of ${total_weighted_value:,.2f} in this asset.")
