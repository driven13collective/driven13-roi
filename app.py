import streamlit as st
import cv2
import tempfile
import supervision as sv
import pandas as pd
import plotly.express as px
from roboflow import Roboflow

# 1. PAGE SETUP
st.set_page_config(page_title="Driven 13 | ROI Auditor", page_icon="🏎️", layout="wide")
st.markdown("<style>.stMetric {background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #00ffcc;}</style>", unsafe_allow_html=True)

st.title("🏎️ Driven 13 Collective: ROI Sponsorship Auditor")

# 2. SIDEBAR CONFIG
with st.sidebar:
    st.header("Audit & Goal Settings")
    api_key = st.text_input("API Key", value="3rcTmYwUyM4deHfzdLhy", type="password")
    roi_goal = st.number_input("Target ROI Goal ($)", value=5000.0)
    tv_slot_cost = st.number_input("Standard TV 30s Slot Cost ($)", value=25000.0)
    val_price = st.number_input("Valvoline $/sighting", value=15.0)
    comp_price = st.number_input("Competitor $/sighting", value=10.0)

# 3. INITIALIZE AI
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("valvoline-roi")
model = project.version(1).model 

up_file = st.file_uploader("Upload Race Footage", type=["mp4", "mov"])

if up_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(up_file.read())
    
    if 'audit_data' not in st.session_state:
        st.session_state.audit_data = {
            "Valvoline": {"money": 0.0, "sightings": 0, "quality_sum": 0.0},
            "Competitor": {"money": 0.0, "sightings": 0, "quality_sum": 0.0}
        }

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Analysis")
        frame_window = st.empty()
        
        # --- GOAL PROGRESS BAR ---
        goal_text = st.empty()
        goal_bar = st.progress(0)
        
        cap = cv2.VideoCapture(tfile.name)
        v_info = sv.VideoInfo.from_video_path(tfile.name)

        if st.button("🚀 RUN ROI AUDIT"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                results = model.predict(frame, confidence=40).json()
                
                for pred in results['predictions']:
                    label = "Valvoline" if "valvoline" in pred['class'].lower() else "Competitor"
                    
                    # --- MEDIA QUALITY SCORE LOGIC ---
                    area = (pred['width'] * pred['height']) / (v_info.width * v_info.height)
                    q_score = min(((area * 1000) + (pred['confidence'] * 100)) / 2, 100)
                    
                    st.session_state.audit_data[label]["sightings"] += 1
                    st.session_state.audit_data[label]["money"] += val_price if label == "Valvoline" else comp_price
                    st.session_state.audit_data[label]["quality_sum"] += q_score

                # Update Progress Bar Live
                current_roi = st.session_state.audit_data["Valvoline"]["money"]
                progress_pct = min(current_roi / roi_goal, 1.0)
                goal_bar.progress(progress_pct)
                goal_text.write(f"**Valvoline ROI Goal Progress:** ${current_roi:,.2f} / ${roi_goal:,.2f} ({progress_pct*100:.1f}%)")

                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if current_roi >= roi_goal:
                st.balloons()
                st.success("🎯 Target ROI Goal Reached!")
            cap.release()

    with col2:
        st.subheader("Financial Share of Voice")
        
        report_list = []
        for brand, stats in st.session_state.audit_data.items():
            avg_q = stats["quality_sum"] / stats["sightings"] if stats["sightings"] > 0 else 0
            report_list.append({
                "Brand": brand, "Money": stats["money"], 
                "Sightings": stats["sightings"], "Quality": round(avg_q, 1)
            })
        
        df_roi = pd.DataFrame(report_list)
        
        fig = px.pie(df_roi, values='Money', names='Brand', hole=0.4, 
                     color_discrete_sequence=['#CC0000', '#003366'])
        fig.update_traces(texttemplate="$%{value:,.0f}<br>%{percent}")
        st.plotly_chart(fig, use_container_width=True)

        for _, row in df_roi.iterrows():
            st.metric(f"{row['Brand']} Quality", f"{row['Quality']}%")
        
        # --- COST AVOIDANCE ---
        st.divider()
        st.subheader("Cost Avoidance")
        avoidance = tv_slot_cost - st.session_state.audit_data["Valvoline"]["money"]
        st.metric("Savings vs TV Spend", f"${avoidance:,.2f}", delta="Cost Saved")

        # --- DOWNLOAD REPORT ---
        csv = df_roi.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Driven 13 ROI Report", 
            data=csv, 
            file_name='Driven13_ROI_Report.csv', 
            mime='text/csv'
        )

st.caption("Driven 13 Collective | Sponsorship Auditor V14.0")


