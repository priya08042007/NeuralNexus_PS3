import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import plotly.express as px

# ==============================
# LOAD MODEL
# ==============================

model = YOLO("runs/detect/train/weights/best.pt")
model.fuse()

class_names = ['fiber', 'fragment', 'film', 'pellet']

# ==============================
# FUNCTIONS
# ==============================

def feret_diameter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_d = 0
    for cnt in contours:
        for i in cnt:
            for j in cnt:
                dist = np.linalg.norm(i - j)
                max_d = max(max_d, dist)

    return max_d


def risk_score(label, size):
    morph_score = {
        "fiber": 40,
        "fragment": 30,
        "film": 25,
        "pellet": 20
    }
    size_score = max(0, 60 - size)
    return min(100, morph_score[label] + size_score)


def risk_label(score):
    if score > 75:
        return "🔴 High"
    elif score > 60:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# ==============================
# UI
# ==============================

st.set_page_config(layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #00E5FF;
}
.metric-box {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🌊 Microplastic AI Dashboard")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)

# ==============================
# PAGE STATE
# ==============================

if "page" not in st.session_state:
    st.session_state.page = "main"

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# ==============================
# PROCESS
# ==============================

if uploaded_files:

    summary = {"fiber": 0, "fragment": 0, "film": 0, "pellet": 0}
    image_summary = {}
    all_sizes = []
    all_scores = []
    image_detections = {}

    # ==============================
    # DETECTION LOOP
    # ==============================

    for file in uploaded_files:

        image_summary[file.name] = {"fiber": 0, "fragment": 0, "film": 0, "pellet": 0}
        image_detections[file.name] = []

        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                label = class_names[int(cls)]

                size = feret_diameter(crop)
                score = risk_score(label, size)

                summary[label] += 1
                image_summary[file.name][label] += 1

                all_sizes.append(size)
                all_scores.append(score)

                image_detections[file.name].append({
                    "crop": crop,
                    "label": label,
                    "size": size,
                    "score": score
                })

    # ==============================
    # MAIN PAGE
    # ==============================

    if st.session_state.page == "main":

        col1, col2 = st.columns([2.5, 1])

        # ==============================
        # LEFT PANEL (UNCHANGED LOGIC)
        # ==============================

        with col1:
            for file in uploaded_files:

                st.markdown(f"## 📸 {file.name}")
                detections = image_detections[file.name]

                cols = st.columns(4)

                for i, det in enumerate(detections[:8]):

                    color = "red" if det["score"] > 75 else "orange" if det["score"] > 60 else "lime"

                    with cols[i % 4]:
                        st.image(det["crop"])

                        st.markdown(f"""
                        <div style="
                            background-color:#2a2a2a;
                            padding:12px;
                            border-radius:12px;
                            border:1px solid #444;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                            color:#e0e0e0;
                        ">
                            <b style="color:#00E5FF;">{det['label'].upper()}</b><br>
                            📏 Size: {det['size']:.2f}<br>
                            ⚠️ Score: {det['score']:.1f}<br>
                            <span style="color:{color}; font-weight:bold;">
                                {risk_label(det['score'])}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)

                if len(detections) > 8:
                    if st.button(f"🔎 View More ({file.name})"):
                        st.session_state.page = "details"
                        st.session_state.selected_image = file.name
                        st.rerun()

        # ==============================
        # RIGHT PANEL (UNCHANGED)
        # ==============================

        with col2:
            st.markdown("## 📊 Insights Panel")

            total = sum(summary.values())

            st.markdown(f"<div class='metric-box'>Total: {total}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'>Avg Size: {np.mean(all_sizes):.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-box'>Avg Risk: {np.mean(all_scores):.2f}</div>", unsafe_allow_html=True)

            eco_index = np.mean(all_scores)

            st.markdown("### 🌍 Ecological Threat Index")

            if eco_index > 75:
                st.error(f"{eco_index:.2f} 🔴 Severe Threat")
            elif eco_index > 60:
                st.warning(f"{eco_index:.2f} 🟡 Moderate Threat")
            else:
                st.success(f"{eco_index:.2f} 🟢 Low Threat")

            df = pd.DataFrame(list(summary.items()), columns=["Class", "Count"])
            st.plotly_chart(px.bar(df, x="Class", y="Count", color="Class"), use_container_width=True)
            st.plotly_chart(px.pie(df, names="Class", values="Count"), use_container_width=True)

    # ==============================
    # DETAILS PAGE
    # ==============================

    elif st.session_state.page == "details":

        selected = st.session_state.selected_image

        st.title(f"📄 Detailed View: {selected}")

        if st.button("⬅ Back"):
            st.session_state.page = "main"
            st.rerun()

        st.markdown("## 🖼 Image-wise Detection")

        counts = image_summary[selected]

        df_img = pd.DataFrame({
            "Class": list(counts.keys()),
            "Count": list(counts.values())
        })

        st.table(df_img)

        st.markdown("## 🔬 All Detections")

        detections = image_detections[selected]

        cols = st.columns(4)

        for i, det in enumerate(detections):

            color = "red" if det["score"] > 75 else "orange" if det["score"] > 60 else "lime"

            with cols[i % 4]:
                st.image(det["crop"])

                st.markdown(f"""
                <div style="
                    background-color:#2a2a2a;
                    padding:12px;
                    border-radius:12px;
                    border:1px solid #444;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                    color:#e0e0e0;
                ">
                    <b style="color:#00E5FF;">{det['label'].upper()}</b><br>
                    📏 Size: {det['size']:.2f}<br>
                    ⚠️ Score: {det['score']:.1f}<br>
                    <span style="color:{color}; font-weight:bold;">
                        {risk_label(det['score'])}
                    </span>
                </div>
                """, unsafe_allow_html=True)