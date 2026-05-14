# frontend/streamlit_app.py - Optimized for Render
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide"
)

# Get API URL from environment
API_URL = os.environ.get("API_URL", "https://traffic-sign-api.onrender.com")

# Custom labels
LABELS = {
    40: "STOP",
    38: "YIELD",
    34: "ROUNDABOUT",
    22: "PEDESTRIAN",
    7: "20 km/h",
    8: "30 km/h",
    9: "40 km/h",
    11: "50 km/h",
    12: "60 km/h",
}


def get_label(class_id):
    return LABELS.get(class_id, f"Sign_{class_id}")


# Load local model (for direct detection)
@st.cache_resource
def load_local_model():
    try:
        model = YOLO('model/best.pt')
        return model
    except:
        return None


local_model = load_local_model()

# Title
st.title("🚦 Traffic Sign Detection System")
st.markdown("Real-time detection of traffic signs using YOLOv8 AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    use_api = st.checkbox("Use API (if available)", value=False)

    st.markdown("---")
    st.markdown("### 📊 Stats")
    if local_model:
        st.success("✅ Local model loaded")
    else:
        st.warning("⚠️ Using API mode")

# Main interface
tab1, tab2 = st.tabs(["📷 Image Detection", "ℹ️ About"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original Image", use_column_width=True)

        with col2:
            if st.button("🔍 Detect Signs", type="primary"):
                with st.spinner("Detecting traffic signs..."):

                    # Convert image
                    img_np = np.array(image)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # Use local model or API
                    if use_api and not local_model:
                        # API mode
                        try:
                            img_bytes = uploaded_file.getvalue()
                            files = {"file": img_bytes}
                            response = requests.post(
                                f"{API_URL}/detect",
                                files=files,
                                params={"confidence": confidence},
                                timeout=30
                            )

                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ Found {result['count']} traffic signs!")

                                # Display detections
                                for det in result['detections']:
                                    st.markdown(f"""
                                    <div style="background: rgba(0,255,0,0.1); padding: 10px; border-radius: 5px; margin: 5px 0;">
                                        <strong>{det['label']}</strong><br>
                                        Confidence: {det['confidence']:.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.error("API error. Using local model...")
                                # Fallback to local
                                results = local_model(img_cv, conf=confidence)
                                if results[0].boxes:
                                    for box in results[0].boxes:
                                        conf = float(box.conf[0])
                                        cls = int(box.cls[0])
                                        st.markdown(f"✅ {get_label(cls)} ({conf:.1%})")
                        except Exception as e:
                            st.error(f"API error: {e}")
                    else:
                        # Local mode
                        results = local_model(img_cv, conf=confidence)

                        if results[0].boxes:
                            st.success(f"✅ Found {len(results[0].boxes)} traffic signs!")

                            # Show annotated image
                            annotated = results[0].plot()
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, caption="Detection Results", use_column_width=True)

                            # List detections
                            for box in results[0].boxes:
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                label = get_label(cls)
                                st.markdown(f"""
                                <div style="background: rgba(0,255,0,0.1); padding: 8px; border-radius: 5px; margin: 3px 0;">
                                    <strong>{label}</strong> - {conf:.1%} confidence
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.warning("No traffic signs detected in this image.")

with tab2:
    st.markdown("""
    ### 🚦 About This System

    **Model:** YOLOv8 (You Only Look Once)

    **Accuracy:** 87.8% mAP (mean Average Precision)

    **Classes:** 43 different traffic signs including:
    - STOP signs
    - Speed limits (20, 30, 40, 50, 60, 80, 100, 120 km/h)
    - Warning signs (yield, pedestrian, children, road work)
    - Direction signs (roundabout, straight, turn)

    **Architecture:**
    - Backend: FastAPI
    - Frontend: Streamlit
    - Model: YOLOv8
    - Deployment: Render.com

    **Features:**
    - ✅ Real-time detection
    - ✅ Image upload support
    - ✅ High accuracy
    - ✅ Fast inference (20-30 FPS)
    """
                )

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>🚦 Built with YOLOv8 | Real-time Traffic Sign Detection</div>",
            unsafe_allow_html=True)