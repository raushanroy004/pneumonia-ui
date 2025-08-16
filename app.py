# app.py — Streamlit ONNX pneumonia detector with occlusion heatmap (CPU-only)
from __future__ import annotations
import io, json
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageOps, ImageFile
import streamlit as st

ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly corrupted JPEGs

# ------- Paths -------
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

# ------- Constants (match training) -------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.50
POSITIVE_LABEL = "PNEUMONIA"
NEGATIVE_LABEL = "NORMAL"

# ------- Helpers -------
def first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def safe_open_image(uploaded_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(uploaded_bytes))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def preprocess(pil: Image.Image, size: int = DEFAULT_IMG_SIZE) -> np.ndarray:
    if pil.size != (size, size):
        pil = pil.resize((size, size))
    x = np.asarray(pil).astype("float32") / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # HWC->CHW
    return x[np.newaxis, ...]       # NCHW

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

@st.cache_resource(show_spinner=True)
def load_session_and_meta():
    import onnxruntime as ort
    onnx_path = first_existing(CANDIDATE_ONNX)
    if onnx_path is None:
        raise FileNotFoundError(
            "ONNX model not found. Looked in:\n" + "\n".join(map(str, CANDIDATE_ONNX))
        )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # infer input size if present
    try:
        shp = session.get_inputs()[0].shape  # [N,3,H,W]
        h = int(shp[2]) if isinstance(shp[2], (int, np.integer)) else DEFAULT_IMG_SIZE
        w = int(shp[3]) if isinstance(shp[3], (int, np.integer)) else DEFAULT_IMG_SIZE
        img_size = h if h == w else DEFAULT_IMG_SIZE
    except Exception:
        img_size = DEFAULT_IMG_SIZE

    # load best threshold if available
    metrics_path = first_existing(CANDIDATE_METRICS)
    best_thr = DEFAULT_THRESHOLD
    if metrics_path is not None:
        try:
            data = json.loads(metrics_path.read_text())
            if "best_threshold" in data:
                best_thr = float(data["best_threshold"])
        except Exception:
            pass

    meta = {
        "onnx_path": onnx_path,
        "img_size": img_size,
        "best_threshold": best_thr,
        "metrics_path": metrics_path,
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
    }
    return session, meta

def predict_one(session, meta: dict, pil: Image.Image, threshold: float | None = None) -> Tuple[float, str, float]:
    x = preprocess(pil, size=meta["img_size"])
    out = session.run([meta["output_name"]], {meta["input_name"]: x})[0]
    logit = float(np.ravel(out)[0])
    prob = sigmoid(logit)
    thr = float(meta["best_threshold"] if threshold is None else threshold)
    label = POSITIVE_LABEL if prob >= thr else NEGATIVE_LABEL
    return prob, label, thr

def occlusion_heatmap(session, meta: dict, pil: Image.Image,
                      patch: int = 32, stride: int = 16, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient-free saliency: slide a gray patch; measure drop in P(pneumonia).
    Returns heat (H,W) in [0,1] and RGB overlay (H,W,3) uint8.
    """
    size = meta["img_size"]
    pil_resized = pil.resize((size, size))
    x0 = preprocess(pil_resized, size=size)
    baseline_norm = ((0.5 - IMAGENET_MEAN) / IMAGENET_STD).astype("float32").reshape(1,3,1,1)

    out0 = session.run([meta["output_name"]], {meta["input_name"]: x0})[0]
    prob0 = sigmoid(float(np.ravel(out0)[0]))

    H = W = size
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    samples, coords = [], []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + patch, H)
            x2 = min(x + patch, W)
            x_occ = x0.copy()
            x_occ[:, :, y:y2, x:x2] = baseline_norm
            samples.append(x_occ)
            coords.append((y, y2, x, x2))

    def run_batch(arr_list):
        b = np.concatenate(arr_list, axis=0)  # (B,3,H,W)
        out = session.run([meta["output_name"]], {meta["input_name"]: b})[0]
        logits = np.ravel(out).astype("float32")
        return 1.0 / (1.0 + np.exp(-logits))

    start = 0
    while start < len(samples):
        end = min(start + batch_size, len(samples))
        probs = run_batch(samples[start:end])
        for k, p in enumerate(probs):
            y, y2, x, x2 = coords[start + k]
            drop = max(0.0, prob0 - float(p))
            acc[y:y2, x:x2] += drop
            cnt[y:y2, x:x2] += 1.0
        start = end

    cnt[cnt == 0] = 1.0
    heat = acc / cnt
    m = float(heat.max())
    if m > 1e-8:
        heat /= m
    heat = np.clip(heat, 0.0, 1.0)

    base = np.asarray(pil_resized).astype("float32")
    red = np.zeros_like(base)
    red[..., 0] = 255.0 * heat
    overlay = (0.6 * base + 0.4 * red).clip(0, 255).astype("uint8")
    return heat, overlay

# ---------------- UI ----------------
st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁")
st.title("🫁 Pediatric Pneumonia Detector")
st.caption("DenseNet121 exported to ONNX • CPU-only inference • Occlusion-based heatmap (Grad-CAM-style).")

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
    thr_user = st.slider("Decision threshold (probability for PNEUMONIA)",
                         0.00, 1.00, float(meta["best_threshold"]), 0.01)
    st.markdown("---")
    st.markdown("**Heatmap options** (occlusion method)")
    show_heat = st.checkbox("Show occlusion heatmap", value=True)
    patch = st.select_slider("Patch size (px)", options=[16, 24, 32, 40, 48, 56, 64], value=32)
    stride = st.select_slider("Stride (px)", options=[8, 12, 16, 20, 24, 28, 32], value=16)

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
    raw = uf.getvalue()
    if not raw:
        st.error("Empty file. Please try another image.")
        continue

    # Robust preview (PIL -> numpy)
    try:
        pil = safe_open_image(raw)
        st.image(np.asarray(pil), caption="Uploaded image")
    except Exception as e:
        st.warning(f"Could not preview image ({e}); still running inference.")
        try:
            pil = safe_open_image(raw)
        except Exception as e2:
            st.error(f"Could not read image for inference: {e2}")
            continue

    # Predict
    prob, label, used_thr = predict_one(session, meta, pil, threshold=thr_user)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Prediction", label)
    with c2:
        st.metric("Pneumonia probability", f"{prob:.3f}")
    st.caption(f"Threshold = {used_thr:.2f}  •  Output = {label}")

    # Heatmap (optional)
    if show_heat:
        with st.spinner("Computing heatmap…"):
            heat, overlay = occlusion_heatmap(session, meta, pil, patch=patch, stride=stride, batch_size=64)

        # convert heat (H,W) to RGB for older Streamlit versions
        heat_rgb = (heat * 255).astype("uint8")
        heat_rgb = np.stack([heat_rgb]*3, axis=-1)  # grayscale -> 3-channel

        h1, h2 = st.columns(2)
        with h1:
            st.image(heat_rgb, caption="Occlusion heatmap")
        with h2:
            st.image(overlay, caption="Overlay")

st.success("Done.")
