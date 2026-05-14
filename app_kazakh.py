"""
🚦 Traffic Sign Detection System - Kazakh UI with English Labels (No question marks!)
Run: streamlit run app_kazakh_fixed.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Jol Belgilerin Anyktau | Traffic Sign Detection",
    page_icon="🚦",
    layout="wide"
)

# ========== KAZAKH UI TRANSLATIONS ==========
UI_TEXT = {
    "title": "🚦 Жол белгілерін анықтау жүйесі",
    "subtitle": "YOLOv8 AI технологиясымен | Нақты уақыт режимінде анықтау",
    "image_detection": "📸 Суреттен анықтау",
    "video_detection": "🎥 Бейнеден анықтау",
    "webcam": "📹 Веб-камера",
    "analytics": "📊 Статистика",
    "settings": "⚙️ Баптаулар",
    "confidence": "Сенімділік шегі",
    "detect_button": "🔍 Анықтау",
    "process_video": "🎬 Бейнені өңдеу",
    "take_photo": "📸 Суретке түсіру",
    "upload_image": "Сурет жүктеу",
    "upload_video": "Бейне жүктеу",
    "results": "Нәтижелер",
    "detections": "Анықталған белгілер",
    "total_signs": "Барлық белгілер",
    "sessions": "Сессиялар",
    "no_detections": "Ешқандай белгі анықталмады",
    "success": "Анықталды!",
    "loading_model": "Модель жүктелуде...",
    "footer": "🚦 Жол белгілерін анықтау жүйесі | YOLOv8 негізінде | Streamlit платформасында",
    "statistics": "Статистика",
}

# ========== ENGLISH LABELS (For OpenCV rendering - NO QUESTION MARKS!) ==========
# These will appear on the image bounding boxes
ENGLISH_LABELS = {
    0: "Info",
    7: "20 km/h",
    8: "30 km/h",
    9: "40 km/h",
    10: "60 km/h",
    11: "50 km/h",
    12: "60 km/h",
    16: "No Entry",
    22: "Pedestrian",
    23: "Sign",
    24: "Ahead",
    25: "Parking",
    26: "Children",
    31: "Two-way",
    33: "Sign",
    34: "Roundabout",
    35: "Straight/Right",
    36: "Straight",
    37: "Straight",
    38: "Yield",
    39: "Priority",
    40: "STOP",
    41: "Children",
    42: "Road Work",
}

# Kazakh labels for UI display (in Streamlit, not on OpenCV)
KAZAKH_LABELS = {
    0: "Ақпараттық белгі",
    7: "20 км/сағ",
    8: "30 км/сағ",
    9: "40 км/сағ",
    10: "60 км/сағ",
    11: "50 км/сағ",
    12: "60 км/сағ",
    16: "Көлік жүруге тыйым",
    22: "Жаяу жүргіншілер",
    23: "Жол белгісі",
    24: "Тіке жүру",
    25: "Тұрақ",
    26: "Балалар",
    31: "Қос бағытты",
    33: "Жол белгісі",
    34: "Дөңгелек қиылыс",
    35: "Тіке/Оңға",
    36: "Тіке жүру",
    37: "Тіке жүру",
    38: "Бағыт беру",
    39: "Бас жол",
    40: "ТОҚТА",
    41: "Балалар",
    42: "Жол жөндеу",
}

# Emojis
SIGN_EMOJIS = {
    40: "🛑", 38: "⚠️", 34: "🔄", 22: "🚶", 7: "🔵", 8: "🔵",
    9: "🔵", 11: "🔵", 12: "🔵", 26: "🧒", 42: "🚧", 25: "🅿️"
}


def get_english_label(class_id):
    """Get English label for OpenCV rendering (ASCII only, no question marks)"""
    return ENGLISH_LABELS.get(class_id, f"Sign_{class_id}")


def get_kazakh_ui_label(class_id):
    """Get Kazakh label for UI display"""
    if class_id in KAZAKH_LABELS:
        emoji = SIGN_EMOJIS.get(class_id, "🏷️")
        return f"{emoji} {KAZAKH_LABELS[class_id]}"
    return f"🏷️ Белгі_{class_id}"


# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: rgba(255,255,255,0.7);
        margin-bottom: 30px;
    }
    .detection-card {
        background: linear-gradient(135deg, rgba(0,255,135,0.1), rgba(96,239,255,0.05));
        border-left: 4px solid #00ff87;
        border-radius: 10px;
        padding: 12px;
        margin: 8px 0;
    }
    .success-message {
        background: linear-gradient(90deg, rgba(0,255,135,0.15), rgba(96,239,255,0.1));
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin: 15px 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #00ff87, #60efff);
        color: #0f2027;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: rgba(255,255,255,0.4);
        margin-top: 40px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #00ff87;
    }
</style>
""", unsafe_allow_html=True)


# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load YOLO model"""
    with st.spinner(UI_TEXT["loading_model"]):
        try:
            model_path = Path("model/best.pt")
            if model_path.exists():
                model = YOLO(str(model_path))
                return model
            else:
                return YOLO("yolov8n.pt")
        except Exception as e:
            st.error(f"Модель жүктелмеді: {e}")
            return None


# ========== SESSION STATE ==========
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0

# ========== HEADER ==========
st.markdown(f'<div class="main-title">{UI_TEXT["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{UI_TEXT["subtitle"]}</div>', unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown(f"### {UI_TEXT['settings']}")
    confidence = st.slider(UI_TEXT["confidence"], 0.1, 0.9, 0.25, 0.05)

    st.markdown("---")
    st.markdown(f"### 📊 {UI_TEXT['statistics']}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<p class="stat-number">{st.session_state.total_detections}</p>', unsafe_allow_html=True)
        st.caption(UI_TEXT["total_signs"])
    with col2:
        st.markdown(f'<p class="stat-number">{len(st.session_state.history)}</p>', unsafe_allow_html=True)
        st.caption(UI_TEXT["sessions"])

    st.markdown("---")
    st.markdown("### 🎯 Қолдау көрсетілетін белгілер")
    for class_id, label in list(KAZAKH_LABELS.items())[:10]:
        emoji = SIGN_EMOJIS.get(class_id, "🏷️")
        st.markdown(f"{emoji} {label}")

# ========== LOAD MODEL ==========
model = load_model()
if model is None:
    st.stop()

# ========== MAIN TABS ==========
tab1, tab2, tab3, tab4 = st.tabs([
    UI_TEXT["image_detection"],
    UI_TEXT["video_detection"],
    UI_TEXT["webcam"],
    UI_TEXT["analytics"]
])

# ========== TAB 1: IMAGE DETECTION ==========
with tab1:
    st.markdown(f"### {UI_TEXT['upload_image']}")

    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Түпнұсқа сурет", use_container_width=True)

        with col2:
            if st.button(UI_TEXT["detect_button"], use_container_width=True):
                with st.spinner("Анықтау жүріп жатыр..."):
                    img_np = np.array(image)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    results = model(img_cv, conf=confidence)

                    if results[0].boxes:
                        detections = []
                        for box in results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])

                            # Use ENGLISH label for OpenCV rendering (NO question marks!)
                            label_en = get_english_label(cls)
                            # Use KAZAKH label for UI display
                            label_kz = get_kazakh_ui_label(cls)

                            detections.append({'label': label_kz, 'confidence': conf})

                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_cv, f"{label_en} {conf:.0%}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, caption="Анықталған нәтиже", use_container_width=True)

                        st.markdown(f'<div class="success-message">✅ {len(detections)} {UI_TEXT["detections"]}</div>',
                                    unsafe_allow_html=True)

                        for det in detections:
                            st.markdown(f"""
                            <div class="detection-card">
                                <div style="display: flex; justify-content: space-between;">
                                    <span><strong>{det['label']}</strong></span>
                                    <span>{det['confidence']:.1%}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.session_state.total_detections += len(detections)
                        st.session_state.history.append({
                            'timestamp': datetime.now(),
                            'type': 'image',
                            'detections': len(detections)
                        })
                    else:
                        st.warning(UI_TEXT["no_detections"])

# ========== TAB 2: VIDEO DETECTION ==========
with tab2:
    st.markdown(f"### {UI_TEXT['upload_video']}")

    video_file = st.file_uploader("", type=['mp4', 'avi', 'mov', 'mkv'])

    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button(UI_TEXT["process_video"], use_container_width=True):
            with st.spinner("Бейне өңделуде..."):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                all_detections = []
                progress_bar = st.progress(0)

                # Create output video path
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_output.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = None

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % 30 == 0:
                        results = model(frame, conf=confidence)
                        if results[0].boxes:
                            all_detections.append(len(results[0].boxes))

                            # Draw boxes on frame
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                label_en = get_english_label(cls)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label_en} {conf:.0%}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Initialize video writer
                            if out is None:
                                h, w = frame.shape[:2]
                                out = cv2.VideoWriter(output_path, fourcc, max(1, 30), (w, h))

                            out.write(frame)

                    frame_count += 1
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

                cap.release()
                if out:
                    out.release()

                st.markdown(f'<div class="success-message">✅ {UI_TEXT["success"]}</div>', unsafe_allow_html=True)

                if all_detections:
                    st.metric(UI_TEXT["total_signs"], sum(all_detections))
                    st.session_state.total_detections += sum(all_detections)

                    # Show output video
                    if Path(output_path).exists():
                        st.video(output_path)
                else:
                    st.info(UI_TEXT["no_detections"])

# ========== TAB 3: WEBCAM ==========
with tab3:
    st.markdown(f"### {UI_TEXT['webcam']}")

    st.info("📸 Веб-камера арқылы суретке түсіріңіз")

    camera_image = st.camera_input(UI_TEXT["take_photo"])

    if camera_image:
        image = Image.open(camera_image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Түсірілген сурет", use_container_width=True)

        with col2:
            with st.spinner("Анықтау жүріп жатыр..."):
                img_np = np.array(image)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                results = model(img_cv, conf=confidence)

                if results[0].boxes:
                    detections = []
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        label_en = get_english_label(cls)
                        label_kz = get_kazakh_ui_label(cls)

                        detections.append({'label': label_kz, 'confidence': conf})

                        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_cv, f"{label_en} {conf:.0%}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    result_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption="Анықталған нәтиже", use_container_width=True)

                    st.markdown(f'<div class="success-message">✅ {len(detections)} {UI_TEXT["detections"]}</div>',
                                unsafe_allow_html=True)

                    for det in detections:
                        st.markdown(f"""
                        <div class="detection-card">
                            <div style="display: flex; justify-content: space-between;">
                                <span><strong>{det['label']}</strong></span>
                                <span>{det['confidence']:.1%}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.total_detections += len(detections)
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'type': 'webcam',
                        'detections': len(detections)
                    })
                else:
                    st.warning(UI_TEXT["no_detections"])

# ========== TAB 4: ANALYTICS ==========
with tab4:
    st.markdown(f"### {UI_TEXT['analytics']}")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(UI_TEXT["total_signs"], st.session_state.total_detections)
        with col2:
            avg = st.session_state.total_detections / len(df) if len(df) > 0 else 0
            st.metric("Орташа көрсеткіш", f"{avg:.1f}")
        with col3:
            st.metric(UI_TEXT["sessions"], len(df))

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['detections'],
            mode='lines+markers',
            name='Анықтаулар',
            line=dict(color='#00ff87', width=3),
            marker=dict(size=10, color='#60efff')
        ))
        fig.update_layout(
            title="Анықтау тарихы",
            xaxis_title="Уақыт",
            yaxis_title="Белгілер саны",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Әлі дерек жоқ. Сурет немесе бейне жүктеңіз!")

# ========== FOOTER ==========
st.markdown(f"""
<div class="footer">
    <p>{UI_TEXT["footer"]}</p>
    <p style="font-size: 0.7rem;">© 2024 | Нақты уақыт режимінде анықтау</p>
</div>
""", unsafe_allow_html=True)