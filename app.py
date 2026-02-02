import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from utils.predict import predict_pil_image
from utils.recommendations import RECOMMENDATIONS
from utils.severity import estimate_severity
from utils.gradcam import generate_gradcam
from utils.heatmap_score import compute_heatmap_strength
from utils.database import create_table, insert_record, fetch_history
from utils.leaf_check import is_leaf_present
from utils.crop_filter import filter_predictions_by_crop



# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crop Disease Detection", layout="centered")
create_table()

model = tf.keras.models.load_model("model/trained_model.keras")


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {background-color:#0f172a;}
.main {background-color:#0f172a;}

.title-text {
    font-size:36px;
    font-weight:bold;
    color:#22c55e;
}

.subtitle-text {
    font-size:16px;
    color:#cbd5e1;
}

.card {
    background:#020617;
    padding:20px;
    border-radius:12px;
    box-shadow:0 0 12px rgba(34,197,94,0.25);
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- WEBCAM PROCESSOR ----------------
class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        if not is_leaf_present(pil_img):
            cv2.putText(img, "No leaf detected", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            return img

        predictions = predict_pil_image(pil_img)
        filtered = filter_predictions_by_crop(predictions, selected_crop)
        label, confidence = filtered[0]


        cv2.putText(
            img,
            f"{label} ({confidence*100:.1f}%)",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (
                0,255,0),
            2
        )
        return img


# ---------------- HEADER ----------------
st.markdown("<div class='title-text'>🌱 Crop Disease Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>AI-powered leaf disease identification & severity analysis</div>", unsafe_allow_html=True)
selected_crop = st.selectbox(
    "Select Crop Type",
    ["Rice", "Maize", "Cotton", "Wheat", "Sugarcane"]
)


# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])


# ---------------- IMAGE PIPELINE ----------------
if uploaded_file:

    image = Image.open(uploaded_file)

    if not is_leaf_present(image):
        st.error("No leaf detected. Please upload a clear leaf image.")
        st.stop()

    with st.spinner("Analyzing image..."):
        predictions = predict_pil_image(image)

        filtered_predictions = filter_predictions_by_crop(
            predictions,
            selected_crop
        )

        label, confidence = filtered_predictions[0]


        severity_percent, severity_level = estimate_severity(image)
        heatmap_img, raw_heatmap = generate_gradcam(model, image)
        heatmap_strength = compute_heatmap_strength(heatmap_img)

    info = RECOMMENDATIONS.get(label)
    reliability = (confidence * 0.6) + (heatmap_strength * 0.4)

    if info:
        insert_record(
            info["crop"],
            info["disease"],
            round(confidence*100,2),
            severity_level,
            round(reliability*100,2)
        )

    col1, col2 = st.columns(2)

    # ---------- LEFT ----------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Uploaded Leaf")
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- RIGHT ----------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if info:
            st.write(f"**Crop:** {info['crop']}")
            st.write(f"**Disease:** {info['disease']}")
        else:
            st.write(f"**Prediction:** {label}")

        st.write(f"**Confidence:** {confidence*100:.2f}%")
        st.write(f"**Severity:** {severity_level} ({severity_percent}%)")
        st.write(f"**Reliability Score:** {reliability*100:.1f}%")

        st.markdown("### Top-3 Predictions (Filtered by Crop)")
        for name, prob in filtered_predictions[:3]:
            st.write(f"{name} → {prob*100:.2f}%")


        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- WARNING ----------
    if heatmap_strength < 0.12:
        st.warning("⚠ Low visual certainty: Please upload clearer image or multiple angles.")

    # ---------- HEATMAP ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Explainable AI Heatmap")
    st.image(heatmap_img, use_column_width=True)

    st.markdown("""
    **Heatmap Color Guide**  
    🔴 Red → Highly infected  
    🟡 Yellow → Moderate infection  
    🟢 Green → Slight infection  
    🔵 Blue → Healthy  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- TREATMENT ----------
    if info:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Recommended Actions")
        for step in info["treatment"]:
            st.write("✔ " + step)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------- SIDEBAR HISTORY ----------------
st.sidebar.title("Detection History")
rows = fetch_history()

if len(rows) == 0:
    st.sidebar.info("No history yet.")
else:
    for r in rows[:10]:
        st.sidebar.write(
            f"{r[1]} | {r[2]} | {r[3]}\n"
            f"Conf:{r[4]}% | Severity:{r[5]} | Rel:{r[6]}%"
        )


# ---------------- LIVE CAMERA ----------------
st.subheader("Live Camera Detection")

webrtc_streamer(
    key="camera",
    video_processor_factory=VideoProcessor
)
