# app.py — Streamlit UI using ONNX Runtime (no torch/torchvision on the server)
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
import onnxruntime as ort

# -------------------- PATHS --------------------
BASE = Path(__file__).parent
ONNX_MODEL = BASE / "pneumonia_densenet_model.onnx"   # commit this file to GitHub

# -------------------- UI -----------------------
st.set_page_config(page_title="🫁 Pediatric Pneumonia Detector", layout="centered")
st.title("🫁 Pediatric Pneumonia Detector (ONNX)")

with st.sidebar:
    st.subheader("Settings")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    st.caption("Prediction ≥ threshold → **PNEUMONIA**, otherwise **NORMAL**.")

@st.cache_resource(show_spinner=False)
def load_session(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at:\n{model_path}")
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name

def _resize_center_crop(img: Image.Image, size=224):
    # keep aspect, then center-crop to size x size
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    img = img.resize((nw, nh), Image.BICUBIC)
    left = (nw - size) // 2
    top  = (nh - size) // 2
    img = img.crop((left, top, left+size, top+size))
    return img

def preprocess(pil: Image.Image) -> np.ndarray:
    """Match typical DenseNet/ImageNet preprocessing."""
    pil = pil.convert("RGB")
    pil = _resize_center_crop(pil, 224)
    arr = np.asarray(pil).astype("float32") / 255.0  # HWC, 0..1
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std  = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 1, 2))  # HWC (no-op but explicit)
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = arr[None, ...]                # NCHW
    return arr

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# -------------------- Load Model ----------------
try:
    sess, in_name, out_name = load_session(ONNX_MODEL)
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

# -------------------- Upload & Predict ----------
st.write("Upload a single chest X-ray (JPG/PNG).")
file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

if file:
    try:
        pil = Image.open(file)
    except UnidentifiedImageError:
        st.error("Could not read image file. Please upload a valid PNG/JPG.")
        st.stop()

    st.image(pil, caption="Uploaded image", use_container_width=True)
    with st.spinner("Running inference..."):
        x = preprocess(pil)
        logits = sess.run([out_name], {in_name: x})[0]  # shape (N,1) or (N,)
        logit = float(np.array(logits).reshape(-1)[0])
        prob = float(sigmoid(np.array([logit]))[0])

    pred = "PNEUMONIA" if prob >= thr else "NORMAL"
    st.success(f"**Prediction:** {pred}")
    st.metric("Pneumonia probability", f"{prob*100:.1f}%")
    st.caption(f"Threshold = {thr:.2f}")
else:
    st.info("Choose an image to get a prediction.")
