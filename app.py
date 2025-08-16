# app.py — Streamlit ONNX pneumonia detector (CPU-only, no torch/torchvision needed)

from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# ========= Project paths (edit only if your files live elsewhere) =========
BASE = Path(__file__).parent

# We try these paths in order to find your ONNX and metrics JSON
CANDIDATE_ONNX = [
    BASE / "pneumonia_densenet_model.onnx",
    BASE / "pediatric_pneumonia_xray" / "pneumonia_densenet_model.onnx",
]
CANDIDATE_METRICS = [
    BASE / "test_metrics.json",
    BASE / "outputs" / "test_metrics.json",
    BASE / "pediatric_pneumonia_xray" / "outputs" / "test_metrics.json",
]

# ========= Model/data constants (keep consistent with training) =========
# If your training used other stats/size, change here:
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMG_SIZE = 224         # your export likely used 224
DEFAULT_THRESHOLD = 0.50       # will be overridden by metrics.json if present
POSITIVE_LABEL = "PNEUMONIA"   # positive class name
NEGATIVE_LABEL = "NORMAL"

# ========= Small helpers =========
def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def safe_open_image(uploaded_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(uploaded_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def preprocess(pil: Image.Image, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """PIL RGB -> normalized NCHW float32 (1,3,H,W) for ONNX runtime."""
    if pil.size != (size, size):
        pil = pil.resize((size, size))
    x = np.asarray(pil).astype("float32") / 255.0           # HWC, [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD                  # normalize
    x = np.transpose(x, (2, 0, 1))                          # HWC->CHW
    x = x[np.newaxis, ...]                                  # add batch
    return x

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

# ========= Load model + meta (cached) =========
@st.cache_resource(show_spinner=True)
def load_session_and_meta():
    import onnxruntime as ort

    onnx_path = first_existing(CANDIDATE_ONNX)
    if onnx_path is None:
        raise FileNotFoundError(
            "ONNX model not found. Expected at one of:\n" +
            "\n".join([str(p) for p in CANDIDATE_ONNX])
        )

    # ONNXRuntime CPU session
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    # Determine input spatial size from the model if given (fallback to 224)
    try:
        model_in = session.get_inputs()[0]
        shape = model_in.shape  # e.g., ['N', 3, 224, 224] or [None, 3, None, None]
        h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else DEFAULT_IMG_SIZE
        w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else DEFAULT_IMG_SIZE
        img_size = int(h) if h == w else DEFAULT_IMG_SIZE
    except Exception:
        img_size = DEFAULT_IMG_SIZE

    # Best threshold from metrics (optional)
    metrics_path = first_existing(CANDIDATE_METRICS)
    best_thr = DEFAULT_THRESHOLD
    if metrics_path is not None:
        try:
            data = json.loads(metrics_path.read_text())
            if "best_threshold" in data:
                best_thr = float(data["best_threshold"])
        except Exception:
            pass

    # IO names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    meta = {
        "onnx_path": onnx_path,
        "img_size": img_size,
        "best_threshold": best_thr,
        "input_name": input_name,
        "output_name": output_name,
        "metrics_path": metrics_path,
    }
    return session, meta

def predict_one(session, meta: dict, pil: Image.Image, threshold: float | None = None):
    """Returns (prob_pneumonia, label_str)"""
    x = preprocess(pil, size=meta["img_size"])
    out = session.run([meta["output_name"]], {meta["input_name"]: x})[0]
    logit = float(np.ravel(out)[0])
    prob = sigmoid(logit)
    thr = float(meta["best_threshold"] if threshold is None else threshold)
    label = POSITIVE_LABEL if prob >= thr else NEGATIVE_LABEL
    return prob, label, thr

# ============================== UI ==============================
st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁", layout="centered")

st.title("🫁 Pediatric Pneumonia Detector")
st.caption("DenseNet121 (exported to ONNX). CPU-only inference.")

# Load model + meta once
try:
    session, meta = load_session_and_meta()
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

with st.expander("Settings", expanded=False):
    st.write("**Model file:**", meta["onnx_path"].name)
    st.write("**Input size:**", f"{meta['img_size']}×{meta['img_size']}")
    if meta["metrics_path"] is not None:
        st.write("**Metrics file:**", meta["metrics_path"].name)
    else:
        st.write("**Metrics file:** not found (using threshold 0.50)")

    # Let user override threshold
    thr_user = st.slider(
        "Decision threshold (probability for PNEUMONIA)",
        min_value=0.00, max_value=1.00, value=float(meta["best_threshold"]), step=0.01
    )

st.subheader("Upload chest X-ray")
uploaded_files = st.file_uploader(
    "Upload one or more images (PNG/JPG/TIFF/BMP)…",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload a PNG/JPG to get a prediction.")
    st.stop()

# Process each file
for i, uf in enumerate(uploaded_files, start=1):
    st.markdown(f"---\n**Image {i}:** `{uf.name}`")
    data = uf.read()
    if not data:
        st.error("Empty file. Please try another image.")
        continue

    # Validate and preview
    try:
        pil = safe_open_image(data)
    except Exception as e:
        st.error(f"Could not read image: {e}")
        continue

    # Preview (convert to uint8 numpy to avoid mode issues)
    preview = np.asarray(pil).astype("uint8")
    st.image(preview, caption="Uploaded image", use_container_width=True)

    # Predict
    prob, label, used_thr = predict_one(session, meta, pil, threshold=thr_user)

    # Display result
    cols = st.columns([1, 2])
    with cols[0]:
        st.metric("Prediction", label)
    with cols[1]:
        st.metric("Pneumonia probability", f"{prob:.3f}")
    st.caption(f"Threshold = {used_thr:.2f}  •  Output = {label}")

st.success("Done.")
