import streamlit as st
import cv2
import tempfile
import os
import supervision as sv
import pandas as pd
import plotly.express as px
from inference.models.utils import get_roboflow_model

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | Competitive ROI", page_icon="🏎️", layout="wide")
st.markdown("<style>.stMetric {background-color: #1e2130; padding: 20px; border-radius: 15px; border: 1px solid #00ffcc;}</style>", unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: Competitive Sponsorship Auditor")
st.subheader("Multi-Brand Share of Voice & Media Valuation")

# 2. SIDEBAR
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Roboflow API Key", value="fR501eGEeNlUzVcE3uNj", type="password")
    model_id = st.text_input("Model ID", value="driven-13-aramco-roi/9")
    val_type = st.selectbox("Benchmark", ["TV (30s Slot)", "Social (CPM)"])
    base_rate = st.number_input("Market Rate ($)", value=50000 if "TV" in val_type else 25)

# 3. UPLOAD & AUDIT
up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(up_file.read())
    
    if st.button("🚀 GENERATE COMPETITIVE AUDIT"):
        try:
            model = get_roboflow_model(model_id=model_id, api_key=api_key)
            tracker = sv.ByteTrack()
            v_info = sv.VideoInfo.from_video_path(tfile.name)
            
            # --- MULTI-CLASS DATA STORAGE ---
            # This tracks every brand the AI sees separately
            brand_stats = {} 
            prog = st.progress(0)

            def analyze_frame(frame, idx):
                res = model.infer(frame, confidence=0.40)[0]
                det = sv.Detections.from_inference(res)
                det = tracker.update_with_detections(det)
                
                for i in range(len(det)):
                    # Get brand name from the model detection
                    brand = det.data['class_name'][i]
                    
                    if brand not in brand_stats:
                        brand_stats[brand] = {"val": 0.0, "frames": 0, "ids": set()}
                    
                    brand_stats[brand]["ids"].add(det.tracker_id[i])
                    brand_stats[brand]["frames"] += 1
                    
                    # Valuation Logic
                    box = det.xyxy[i]
                    area = ((box[2]-box[0])*(box[3]-box[1])) / (v_info.width*v_info.height)
                    quality = (min(area * 50, 1.0) * 0.7) + (det.confidence[i] * 0.3)
                    
                    if "TV" in val_type:
                        brand_stats[brand]["val"] += (base_rate / 30 / v_info.fps) * quality
                    else:
                        brand_stats[brand]["val"] += (base_rate * 100 / v_info.fps) * quality

                if idx % 20 == 0: prog.progress(int((idx / v_info.total_frames) * 100))
                return frame

            sv.process_video(source_path=tfile.name, target_path="/tmp/null.mp4", callback=analyze_frame)

            # --- 4. DISPLAY COMPETITIVE RESULTS ---
            st.balloons()
            st.header("📊 Market Share of Voice (SoV) Report")
            
            # Convert stats to a readable Table for Charting
            report_data = []
            for brand, data in brand_stats.items():
                secs = data["frames"] / v_info.fps
                report_data.append({
                    "Brand": brand,
                    "Total Exposure (s)": round(secs, 2),
                    "Earned Media Value ($)": round(data["val"], 2),
                    "Unique Sightings": len(data["ids"])
                })
            
            df = pd.DataFrame(report_data)

            # Create visual charts
            col_chart, col_data = st.columns([2, 1])
            
            with col_chart:
                fig = px.pie(df, values='Total Exposure (s)', names='Brand', 
                             title='Share of Voice (By Airtime)',
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)

            with col_data:
                st.write("### Total Value by Brand")
                for index, row in df.iterrows():
                    st.metric(f"{row['Brand']}", f"${row['Earned Media Value ($)']:,.2f}")

            st.divider()
            st.write("### Technical Breakdown")
            st.table(df)

        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Driven 13 Collective | Competitive Audit Build | V10.3")

