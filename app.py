# app.py — Streamlit UI using ONNX Runtime (CPU-only)
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import onnxruntime as ort

BASE = Path(__file__).parent
ONNX_MODEL = BASE / "pneumonia_densenet_model.onnx"  # committed file

st.set_page_config(page_title="🫁 Pediatric Pneumonia Detector", layout="centered")
st.title("🫁 Pediatric Pneumonia Detector (ONNX)")

with st.sidebar:
    st.subheader("Settings")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    st.caption("Prediction ≥ threshold → **PNEUMONIA**; otherwise **NORMAL**.")

@st.cache_resource(show_spinner=False)
def load_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found:\n{model_path}")
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name

def _resize_center_crop(img: Image.Image, size=224):
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left+size, top+size))

def preprocess(pil: Image.Image) -> np.ndarray:
    pil = pil.convert("RGB")
    pil = _resize_center_crop(pil, 224)
    x = np.asarray(pil).astype("float32") / 255.0  # HWC
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
    return x

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

try:
    sess, in_name, out_name = load_session(ONNX_MODEL)
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

st.write("Upload a chest X-ray (PNG/JPG).")
file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

if file:
    try:
        pil = Image.open(file)
    except UnidentifiedImageError:
        st.error("Could not read image. Please upload a valid PNG/JPG.")
        st.stop()

    st.image(pil, caption="Uploaded image", use_container_width=True)
    with st.spinner("Running inference..."):
        x = preprocess(pil)
        logits = sess.run([out_name], {in_name: x})[0]
        logit = float(np.array(logits).reshape(-1)[0])
        prob = float(sigmoid(logit))

    pred = "PNEUMONIA" if prob >= thr else "NORMAL"
    st.success(f"**Prediction:** {pred}")
    st.metric("Pneumonia probability", f"{prob*100:.1f}%")
    st.caption(f"Threshold = {thr:.2f}")
else:
    st.info("Choose an image to get a prediction.")
