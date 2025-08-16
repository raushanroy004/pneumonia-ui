# app.py — Streamlit UI for Pediatric Pneumonia (ONNX + CPU-only)
from __future__ import annotations

import io
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

import streamlit as st
import onnxruntime as ort

# ---------------- PATHS ----------------
BASE = Path(__file__).parent
ONNX_PATH = BASE / "pneumonia_densenet_model.onnx"   # <-- put your ONNX file here

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(
    page_title="Pediatric Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
)

def _pill(label: str, value: str, emoji: str = "🧠") -> None:
    st.markdown(
        f"""
        <div style="
            background: #1f2937; padding: 12px 14px; border-radius: 12px;
            border: 1px solid #374151; color: #e5e7eb; font-size: 14px;">
            <span style="opacity:.85">{emoji} {label}:</span>
            <b style="margin-left: 6px">{value}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner=True)
def load_session() -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ONNX_PATH), sess_options=so, providers=providers)

# ---------------- PRE/POST PROCESS ----------------
IM_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_pil_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB")

def preprocess(pil: Image.Image, size: int = IM_SIZE) -> Tuple[np.ndarray, Tuple[int, int]]:
    w, h = pil.size
    # Resize keeping aspect, then center-crop to size x size
    scale = max(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = pil.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    arr = np.asarray(cropped).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # NCHW
    return arr.astype(np.float32), (w, h)

def prob_from_output(y: np.ndarray) -> float:
    """
    Handles either:
      - shape (1,1) logits
      - shape (1,2) logits or probs with class order [normal, pneumonia]
    """
    y = np.array(y)
    if y.ndim == 2 and y.shape[1] == 1:
        logit = float(y[0, 0])
        return 1.0 / (1.0 + math.exp(-logit))
    if y.ndim == 2 and y.shape[1] == 2:
        row = y[0]
        # If not a probability distribution, softmax it.
        if not (0.0 <= row.min() and row.max() <= 1.0 and abs(row.sum() - 1.0) < 1e-4):
            m = row.max()
            e = np.exp(row - m)
            row = e / e.sum()
        return float(row[1])  # pneumonia class
    # Fallback: squeeze -> sigmoid
    logit = float(np.squeeze(y))
    return 1.0 / (1.0 + math.exp(-logit))

def run_inference(sess: ort.InferenceSession, pil_img: Image.Image) -> float:
    x, _ = preprocess(pil_img, IM_SIZE)
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x})[0]
    return prob_from_output(out)

# ---------------- HEATMAP (black-box Grad-CAM style) ----------------
def occlusion_heatmap(
    sess: ort.InferenceSession,
    pil_img: Image.Image,
    base_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    occ_size: int = 32,
    stride: int = 16,
) -> np.ndarray:
    """
    Returns a [H,W] heatmap scaled 0..1 using occlusion sensitivity
    (how much probability drops when a patch is covered).
    Works with any black-box classifier (no gradients).
    """
    # Work on model input resolution
    w0, h0 = pil_img.size
    pil = pil_img.resize((IM_SIZE, IM_SIZE), Image.BICUBIC)
    base_prob = run_inference(sess, pil)

    # Normalized image 0..1 in HWC
    img = np.asarray(pil).astype(np.float32) / 255.0
    H, W, _ = img.shape

    heat = np.zeros((H, W), dtype=np.float32)

    # Precompute mask patch
    patch = np.ones((occ_size, occ_size, 3), dtype=np.float32)
    patch[:] = np.array(base_color, dtype=np.float32)

    # Iterate windows
    for y in range(0, H - occ_size + 1, stride):
        for x in range(0, W - occ_size + 1, stride):
            tmp = img.copy()
            tmp[y:y + occ_size, x:x + occ_size, :] = patch
            # Normalize + NCHW
            arr = (tmp - IMAGENET_MEAN) / IMAGENET_STD
            arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
            prob = prob_from_output(
                sess.run(None, {sess.get_inputs()[0].name: arr})[0]
            )
            drop = max(0.0, base_prob - prob)  # only positive importance
            heat[y:y + occ_size, x:x + occ_size] += drop

    # Normalize heat
    hmin, hmax = float(heat.min()), float(heat.max())
    if hmax > hmin:
        heat = (heat - hmin) / (hmax - hmin)
    else:
        heat[:] = 0.0

    # Upscale back to original image size
    heat_img = Image.fromarray(np.uint8(heat * 255), mode="L").resize((w0, h0), Image.BICUBIC)
    return np.asarray(heat_img).astype(np.float32) / 255.0

def overlay_heatmap(pil_img: Image.Image, heat: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Colorize heatmap (JET) and blend with the original image.
    """
    # Simple JET-like colormap without importing matplotlib
    def jet_colorize(h: np.ndarray) -> np.ndarray:
        # h in [0,1]
        r = np.clip(1.5 - np.abs(4*h - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4*h - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4*h - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)

    img = np.asarray(pil_img.convert("RGB")).astype(np.float32) / 255.0
    color = jet_colorize(heat)
    overlay = (1 - alpha) * img + alpha * color
    overlay = np.clip(overlay, 0, 1)
    return Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB")

# ---------------- UI ----------------
st.markdown("<h1>🫁 Pediatric Pneumonia Detector</h1>", unsafe_allow_html=True)
st.caption("DenseNet121 (exported to ONNX). CPU-only inference.")

with st.sidebar:
    _pill("Backbone", "DenseNet121 (ONNX)")
    use_tuned = st.toggle("Use tuned threshold", value=True)
    thr = 0.50 if use_tuned else st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01)
    show_cam = st.toggle("Show Grad-CAM", value=True)

st.subheader("Upload a chest X-ray")
files = st.file_uploader(
    "Upload one or more images (PNG/JPG/TIFF/BMP)…",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

# Load model session once
if not ONNX_PATH.exists():
    st.error(f"ONNX model not found: {ONNX_PATH}")
    st.stop()

sess = load_session()

if files:
    for idx, uf in enumerate(files, 1):
        st.markdown(f"### Image {idx}: {uf.name}")
        data = uf.getvalue()  # read bytes once

        # --- Preview (robust) ---
        preview_col, result_col = st.columns([1.1, 1.3])
        preview = None
        try:
            preview = load_pil_from_bytes(data)
            with preview_col:
                st.image(np.asarray(preview), caption="Uploaded image", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            with preview_col:
                st.info("Could not preview image; still attempting to run inference.")

        # --- Inference ---
        try:
            pil = preview if preview is not None else load_pil_from_bytes(data)
        except Exception as e:
            st.error(f"Failed to read image for inference: {e}")
            continue

        with st.spinner("Running inference…"):
            t0 = time.time()
            prob = run_inference(sess, pil)
            elapsed = time.time() - t0

        label = "PNEUMONIA" if prob >= thr else "NORMAL"
        box = st.success if label == "PNEUMONIA" else st.info
        box(f"**Prediction:** {label}  |  **Probability (Pneumonia):** {prob:.3f}  |  **Threshold:** {thr:.2f}  ·  _{elapsed*1000:.0f} ms_")

        # --- Heatmap / Grad-CAM style ---
        if show_cam:
            with st.spinner("Computing Grad-CAM style heatmap…"):
                heat = occlusion_heatmap(sess, pil, occ_size=32, stride=16)
                overlay = overlay_heatmap(pil, heat, alpha=0.45)
            with result_col:
                st.image(np.asarray(overlay), caption="Grad-CAM overlay", use_container_width=True)
        st.divider()
else:
    st.info("Drag & drop chest X-ray images above to get predictions and heatmaps.")
