# app.py — Streamlit ONNX pneumonia detector (robust preview)

from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFile
import streamlit as st

# ========= Project paths =========
BASE = Path(__file__).parent
CANDIDATE_ONNX = [
    BASE / "pneumonia_densenet_model.onnx",
    BASE / "pediatric_pneumonia_xray" / "pneumonia_densenet_model.onnx",
]
CANDIDATE_METRICS = [
    BASE / "test_metrics.json",
    BASE / "outputs" / "test_metrics.json",
    BASE / "pediatric_pneumonia_xray" / "outputs" / "test_metrics.json",
]

# ========= Model/data constants =========
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.50
POSITIVE_LABEL = "PNEUMONIA"
NEGATIVE_LABEL = "NORMAL"

# ========= Helpers =========
def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def safe_open_image(uploaded_bytes: bytes) -> Image.Image:
    """
    Robustly open user image and return RGB 8-bit PIL Image with EXIF applied.
    Handles truncated files, 16-bit inputs, and non-RGB modes.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(io.BytesIO(uploaded_bytes))
    # Apply EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Normalize mode → RGB (8-bit)
    if img.mode in ("I;16", "I"):  # 16-bit grayscale/integer
        arr = np.array(img, dtype=np.uint16)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        # Map 0..65535 → 0..255 safely
        arr = (arr / 257.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    elif img.mode == "L":  # 8-bit grayscale
        img = Image.merge("RGB", (img, img, img))
    elif img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    return img

def preprocess(pil: Image.Image, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    """PIL RGB -> normalized NCHW float32 (1,3,H,W) for ONNX runtime."""
    if pil.size != (size, size):
        pil = pil.resize((size, size))
    arr = np.asarray(pil)
    if arr.dtype != np.uint8:
        # Ensure consistent 0..255 before normalization
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    x = arr.astype("float32") / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))      # HWC->CHW
    x = x[np.newaxis, ...]              # add batch
    return x

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

# ========= Load model + meta =========
@st.cache_resource(show_spinner=True)
def load_session_and_meta():
    import onnxruntime as ort

    onnx_path = first_existing(CANDIDATE_ONNX)
    if onnx_path is None:
        raise FileNotFoundError(
            "ONNX model not found. Looked in:\n" + "\n".join(map(str, CANDIDATE_ONNX))
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Input size (fallback to 224)
    try:
        shape = session.get_inputs()[0].shape
        h = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else DEFAULT_IMG_SIZE
        w = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else DEFAULT_IMG_SIZE
        img_size = h if h == w else DEFAULT_IMG_SIZE
    except Exception:
        img_size = DEFAULT_IMG_SIZE

    # Best threshold from metrics (optional)
    best_thr = DEFAULT_THRESHOLD
    metrics_path = first_existing(CANDIDATE_METRICS)
    if metrics_path is not None:
        try:
            data = json.loads(metrics_path.read_text())
            if "best_threshold" in data:
                best_thr = float(data["best_threshold"])
        except Exception:
            pass

    meta = {
        "onnx_path": onnx_path,
        "metrics_path": metrics_path,
        "img_size": img_size,
        "best_threshold": best_thr,
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
    }
    return session, meta

def predict_one(session, meta: dict, pil: Image.Image, threshold: float | None = None):
    x = preprocess(pil, size=meta["img_size"])
    out = session.run([meta["output_name"]], {meta["input_name"]: x})[0]
    logit = float(np.ravel(out)[0])
    prob = float(sigmoid(logit))
    thr = float(meta["best_threshold"] if threshold is None else threshold)
    label = POSITIVE_LABEL if prob >= thr else NEGATIVE_LABEL
    return prob, label, thr

# ============================== UI ==============================
st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁", layout="centered")
st.title("🫁 Pediatric Pneumonia Detector")
st.caption("DenseNet121 (exported to ONNX). CPU-only inference.")

# Load model once
try:
    session, meta = load_session_and_meta()
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

with st.expander("Settings", expanded=False):
    st.write("**Model file:**", meta["onnx_path"].name)
    st.write("**Input size:**", f"{meta['img_size']}×{meta['img_size']}")
    st.write("**Metrics file:**", meta["metrics_path"].name if meta["metrics_path"] else "not found (using 0.50)")
    thr_user = st.slider("Decision threshold (prob. for PNEUMONIA)", 0.00, 1.00, float(meta["best_threshold"]), 0.01)

st.subheader("Upload chest X-ray")
uploaded_files = st.file_uploader(
    "Upload one or more images (PNG/JPG/TIFF/BMP)…",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload a PNG/JPG to get a prediction.")
    st.stop()

for i, uf in enumerate(uploaded_files, start=1):
    st.markdown(f"---\n**Image {i}:** `{uf.name}`")
    data = uf.read()
    if not data:
        st.error("Empty file. Please try another image.")
        continue

    # Try to decode for both preview + inference
    try:
        pil = safe_open_image(data)
        # Basic debug info
        st.caption(f"Decoded image → mode: **{pil.mode}**, size: **{pil.size[0]}×{pil.size[1]}**")
        # Primary preview path (safe numpy conversion)
        preview = np.asarray(pil)
        if preview.dtype != np.uint8:
            preview = np.clip(preview, 0, 255).astype(np.uint8)
        st.image(preview, caption="Uploaded image", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not preview image; still attempting to run inference. ({e})")
        # Fallback: preview raw bytes
        try:
            st.image(io.BytesIO(data), caption="Uploaded image (raw preview)", use_container_width=True)
            # Best effort: open again for inference (may still succeed even if preview failed)
            pil = safe_open_image(data)
        except Exception as e2:
            st.error(f"Could not read image at all: {e2}")
            continue

    # Predict
    prob, label, used_thr = predict_one(session, meta, pil, threshold=thr_user)

    cols = st.columns([1, 2])
    with cols[0]:
        st.metric("Prediction", label)
    with cols[1]:
        st.metric("Pneumonia probability", f"{prob:.3f}")
    st.caption(f"Threshold = {used_thr:.2f} • Output = **{label}**")

st.success("Done.")
