import streamlit as st
import cv2
import tempfile
import os
import supervision as sv
from inference.models.utils import get_roboflow_model

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 ROI", page_icon="🏎️", layout="wide")
st.markdown("<style>.stMetric {background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #00ffcc;}</style>", unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: AI Sponsorship Auditor")
st.subheader("Professional Media Valuation Engine")

# 2. SIDEBAR
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Roboflow API Key", value="fR501eGEeNlUzVcE3uNj", type="password")
    model_id = st.text_input("Model ID", value="driven-13-aramco-roi/9")
    val_type = st.selectbox("Benchmark", ["TV (30s Slot)", "Social (CPM)"])
    base_rate = st.number_input("Rate ($)", value=50000 if "TV" in val_type else 25)

# 3. UPLOAD & AUDIT
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(up_file.read())
    
    if st.button("🚀 GENERATE CERTIFIED AUDIT"):
        try:
            model = get_roboflow_model(model_id=model_id, api_key=api_key)
            tracker = sv.ByteTrack()
            v_info = sv.VideoInfo.from_video_path(tfile.name)
            
            # Mutable dictionary to bypass nonlocal restrictions
            results_store = {"val": 0.0, "secs": 0.0, "ids": set()}
            prog = st.progress(0)

            def analyze_frame(frame, idx):
                res = model.infer(frame, confidence=0.40)[0]
                det = sv.Detections.from_inference(res)
                det = tracker.update_with_detections(det)
                
                f_val = 0.0
                for i in range(len(det)):
                    results_store["ids"].add(det.tracker_id[i])
                    # Quality Weighting Logic
                    box = det.xyxy[i]
                    area = ((box[2]-box[0])*(box[3]-box[1])) / (v_info.width*v_info.height)
                    quality = (min(area * 50, 1.0) * 0.7) + (det.confidence[i] * 0.3)
                    
                    if "TV" in val_type:
                        f_val += (base_rate / 30 / v_info.fps) * quality
                    else:
                        f_val += (base_rate * 100 / v_info.fps) * quality

                results_store["val"] += f_val
                if len(det) > 0: results_store["secs"] += (1 / v_info.fps)
                if idx % 15 == 0: prog.progress(int((idx / v_info.total_frames) * 100))
                return frame

            sv.process_video(source_path=tfile.name, target_path="/tmp/null.mp4", callback=analyze_frame)

            # 4. DISPLAY RESULTS
            st.balloons()
            st.header("📊 Certified Exposure Audit")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Airtime", f"{results_store['secs']:.2f}s")
            c2.metric("Unique Sightings", f"{len(results_store['ids'])}")
            c3.metric("Earned Media Value (EMV)", f"${results_store['val']:,.2f}")
            st.success(f"Final Valuation: The brand Aramco achieved ${results_store['val']:,.2f} in this asset.")

        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Driven 13 Collective | Hawaii HQ Development Build | V10.2")
