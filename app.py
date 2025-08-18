# app.py — Fast, explainable ONNX X-ray pneumonia detector (CPU, no torch)
from __future__ import annotations
import io, json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw, UnidentifiedImageError
import streamlit as st

# -------------------- Paths --------------------
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

# -------------------- Constants (match training) --------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.50
POS_LABEL, NEG_LABEL = "PNEUMONIA", "NORMAL"

# -------------------- Utils --------------------
def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def load_pil_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b))
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def pil_to_png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))

# Preprocess NCHW without extra PIL hops (faster)
def to_nchw_normalized(rgb_uint8: np.ndarray) -> np.ndarray:
    x = rgb_uint8.astype("float32") / 255.0           # H,W,3 in [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD            # normalize
    x = np.transpose(x, (2, 0, 1))                    # 3,H,W
    return x

# -------------------- ONNX runtime --------------------
@st.cache_resource(show_spinner=True)
def load_session_and_meta():
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime not installed. Add it to requirements.txt") from e

    onnx_path = first_existing(CANDIDATE_ONNX)
    if onnx_path is None:
        raise FileNotFoundError("ONNX model not found at:\n" + "\n".join(map(str, CANDIDATE_ONNX)))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # infer input size
    try:
        shp = sess.get_inputs()[0].shape  # [None,3,H,W]
        H = int(shp[2]) if isinstance(shp[2], (int, np.integer)) else DEFAULT_IMG_SIZE
        W = int(shp[3]) if isinstance(shp[3], (int, np.integer)) else DEFAULT_IMG_SIZE
        img_size = H if H == W else DEFAULT_IMG_SIZE
    except Exception:
        img_size = DEFAULT_IMG_SIZE

    # threshold from metrics, if present
    thr = DEFAULT_THRESHOLD
    metrics_path = first_existing(CANDIDATE_METRICS)
    if metrics_path is not None:
        try:
            meta_json = json.loads(metrics_path.read_text())
            thr = float(meta_json.get("best_threshold", thr))
        except Exception:
            pass

    meta = dict(
        onnx_path=onnx_path,
        img_size=img_size,
        best_threshold=thr,
        input_name=sess.get_inputs()[0].name,
        output_name=sess.get_outputs()[0].name,
        metrics_path=metrics_path,
    )
    return sess, meta

def predict_batch(sess, meta, batch_nchw: np.ndarray) -> np.ndarray:
    """
    batch_nchw: [N,3,H,W] float32
    returns probs: [N]
    """
    y = sess.run([meta["output_name"]], {meta["input_name"]: batch_nchw})[0].reshape(-1)
    return sigmoid(y).astype("float32")

def predict_one(sess, meta, pil: Image.Image, thr: Optional[float]=None) -> Tuple[float, str, float]:
    size = meta["img_size"]
    img = pil.resize((size, size))
    prob = float(predict_batch(sess, meta, to_nchw_normalized(np.asarray(img, dtype=np.uint8))[None, ...])[0])
    used_thr = float(meta["best_threshold"] if thr is None else thr)
    label = POS_LABEL if prob >= used_thr else NEG_LABEL
    return prob, label, used_thr

# -------------------- FAST EXPLAINER (coarse-to-fine, batched) --------------------
def fast_explain(sess, meta, pil: Image.Image,
                 coarse_grid:int=8, top_k:int=8, refine_factor:int=2,
                 mask_mode:str="mean") -> np.ndarray:
    """
    Model-agnostic, fast:
      1) Split image into coarse_grid x coarse_grid tiles.
      2) Mask each tile (in ONE batched ONNX run), measure prob drop.
      3) Take top_k tiles; refine each into (refine_factor x refine_factor) subtiles,
         mask each (again batched), distribute drops.
    Returns heatmap in [0,1] with shape (H,W) == (size,size).
    """
    size = meta["img_size"]
    img = pil.resize((size, size))
    arr = np.asarray(img, dtype=np.uint8)  # H,W,3
    H, W = arr.shape[:2]

    # base prob (single pass)
    base_prob = float(predict_batch(sess, meta, to_nchw_normalized(arr)[None, ...])[0])

    # ----- coarse pass (batch all tiles) -----
    h_step = H // coarse_grid
    w_step = W // coarse_grid

    tiles: List[Tuple[int,int,int,int]] = []
    batch = []
    # fill color once (mean) if needed
    fill_color = arr.mean(axis=(0,1), keepdims=True).astype(np.uint8)

    for gy in range(coarse_grid):
        for gx in range(coarse_grid):
            y0 = gy * h_step
            x0 = gx * w_step
            y1 = H if gy == coarse_grid - 1 else (y0 + h_step)
            x1 = W if gx == coarse_grid - 1 else (x0 + w_step)

            masked = arr.copy()
            if mask_mode == "black":
                masked[y0:y1, x0:x1, :] = 0
            else:
                masked[y0:y1, x0:x1, :] = fill_color
            batch.append(to_nchw_normalized(masked))
            tiles.append((y0, y1, x0, x1))

    batch_arr = np.stack(batch, axis=0).astype("float32")
    probs = predict_batch(sess, meta, batch_arr)  # [Ntiles]
    drops = np.maximum(0.0, base_prob - probs)    # how much removing tile hurt pneumonia score

    # build coarse heat
    heat = np.zeros((H, W), dtype=np.float32)
    for drop, (y0, y1, x0, x1) in zip(drops, tiles):
        heat[y0:y1, x0:x1] += float(drop)

    # ----- refine top-k tiles -----
    if top_k > 0 and refine_factor > 1:
        top_idx = np.argsort(-drops)[:top_k]
        sub_batch = []
        sub_meta = []
        for idx in top_idx:
            y0, y1, x0, x1 = tiles[idx]
            hh = max(1, (y1 - y0) // refine_factor)
            ww = max(1, (x1 - x0) // refine_factor)
            for sy in range(refine_factor):
                for sx in range(refine_factor):
                    yy0 = y0 + sy * hh
                    xx0 = x0 + sx * ww
                    yy1 = y1 if sy == refine_factor - 1 else (yy0 + hh)
                    xx1 = x1 if sx == refine_factor - 1 else (xx0 + ww)
                    masked = arr.copy()
                    if mask_mode == "black":
                        masked[yy0:yy1, xx0:xx1, :] = 0
                    else:
                        masked[yy0:yy1, xx0:xx1, :] = fill_color
                    sub_batch.append(to_nchw_normalized(masked))
                    sub_meta.append((yy0, yy1, xx0, xx1))

        if sub_batch:
            sub_probs = predict_batch(sess, meta, np.stack(sub_batch, axis=0))
            sub_drops = np.maximum(0.0, base_prob - sub_probs)
            for drop, (yy0, yy1, xx0, xx1) in zip(sub_drops, sub_meta):
                heat[yy0:yy1, xx0:xx1] += float(drop)

    # normalize to [0,1]
    m = heat.max()
    if m > 0:
        heat /= m
    return heat

# -------------------- Post-processing: region text + box --------------------
def region_summary_text(heat: np.ndarray) -> str:
    """Summarize top lung zones by heat mass."""
    H, W = heat.shape
    thirds = [0, H//3, (2*H)//3, H]
    halves = [0, W//2, W]
    names = []
    masses = []

    zones = [
        ("upper-left",   (thirds[0], thirds[1], halves[0], halves[1])),
        ("middle-left",  (thirds[1], thirds[2], halves[0], halves[1])),
        ("lower-left",   (thirds[2], thirds[3], halves[0], halves[1])),
        ("upper-right",  (thirds[0], thirds[1], halves[1], halves[2])),
        ("middle-right", (thirds[1], thirds[2], halves[1], halves[2])),
        ("lower-right",  (thirds[2], thirds[3], halves[1], halves[2])),
    ]
    for name, (y0,y1,x0,x1) in zones:
        m = float(heat[y0:y1, x0:x1].sum())
        names.append(name); masses.append(m)

    order = np.argsort(-np.array(masses))
    top = [names[i] for i in order[:2]]
    if len(top) == 0 or (np.array(masses).sum() == 0):
        return "No concentrated evidence zone detected."
    return "Most influential regions: " + ", ".join(top) + "."

def draw_bbox_on_overlay(base: Image.Image, heat_up: np.ndarray, frac: float=0.6) -> Image.Image:
    """Draw a rectangle around hottest cluster (pixels >= frac * max)."""
    mask = heat_up >= (heat_up.max() * frac if heat_up.max() > 0 else 1.0)
    if not mask.any():
        return base

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    out = base.copy()
    dr = ImageDraw.Draw(out)
    # thick rectangle
    for t in range(3):
        dr.rectangle([x0-t, y0-t, x1+t, y1+t], outline=(255, 0, 0))
    return out

def overlay_heatmap(pil: Image.Image, heat: np.ndarray, alpha: float=0.45) -> Image.Image:
    """Simple red overlay for hot areas; heat in [0,1], resized to pil.size."""
    H, W = pil.size[1], pil.size[0]
    heat_u8 = (np.clip(heat,0,1) * 255).astype("uint8")
    hm = Image.fromarray(heat_u8).resize((pil.size[0], pil.size[1]), Image.BILINEAR)
    hm = hm.convert("L")
    rgba = pil.convert("RGBA")
    red = Image.new("RGBA", rgba.size, (255, 0, 0, 0))
    red.putalpha(hm)  # alpha from heat
    out = Image.alpha_composite(rgba, red).convert("RGB")
    return out

# -------------------- UI --------------------
st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁")

st.title("🫁 Pediatric Pneumonia Detector")
st.caption("DenseNet121 exported to ONNX • CPU-only • Fast, interpretable explanations.")

# Load model/session
try:
    session, meta = load_session_and_meta()
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

with st.expander("Settings", expanded=False):
    st.write("**Model file:**", meta["onnx_path"].name)
    st.write("**Input size:**", f"{meta['img_size']}×{meta['img_size']}")
    st.write("**Metrics file:**", meta["metrics_path"].name if meta["metrics_path"] else "not found (using 0.50)")
    thr_user = st.slider("Decision threshold (probability for PNEUMONIA)",
                         0.00, 1.00, float(meta["best_threshold"]), 0.01)
    st.write("**Explainer:** Coarse-to-fine perturbation (tiles are briefly hidden and the drop in pneumonia probability is measured).")
    with st.popover("Explainer speed / quality"):
        coarse = st.slider("Coarse grid (N×N)", 6, 12, 8, 1)
        top_k = st.slider("Refine top-K tiles", 0, 12, 8, 1)
        refine = st.slider("Refine factor (per tile)", 1, 3, 2, 1)
        mask_mode = st.selectbox("Mask mode", ["mean", "black"], index=0)
        st.caption("Lower grids & K are faster; higher show more detail.")

st.subheader("Upload chest X-ray")
files = st.file_uploader("Upload one or more images (PNG/JPG/TIFF/BMP)…",
                         type=["png","jpg","jpeg","bmp","tif","tiff"], accept_multiple_files=True)

# Helpful explanation (once)
with st.expander("How to read the explanation", expanded=False):
    st.markdown("""
**What’s happening:** we run the model once to get a pneumonia probability.
Then we **briefly hide small tiles** of the image (in **one batched pass** for speed) 
and see how much the probability **drops**. Tiles that cause a **bigger drop** are **more important**.

**What the heatmap means:** red = **important for detecting pneumonia**.  
**Where is the pneumonia?** The hottest cluster (red box) shows **where the model found the most evidence**.

**What the result means:** if **probability ≥ threshold**, the output is **PNEUMONIA**; otherwise **NORMAL**.
""")

if not files:
    st.info("Upload a PNG/JPG to get a prediction and a full explanation.")
    st.stop()

for i, uf in enumerate(files, start=1):
    st.markdown(f"---\n**Image {i}:** `{uf.name}`")
    data = uf.getvalue()
    if not data:
        st.error("Empty file.")
        continue

    # Preview
    try:
        preview = load_pil_from_bytes(data)
        small = preview.copy(); small.thumbnail((1600,1600))
        st.image(pil_to_png_bytes(small), caption="Uploaded image")
    except (UnidentifiedImageError, OSError):
        st.info("Could not preview image; still attempting to run inference.")
        try:
            preview = load_pil_from_bytes(data)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            continue

    # Prediction
    prob, label, used_T = predict_one(session, meta, preview, thr_user)
    c1, c2 = st.columns([1,2])
    with c1: st.metric("Prediction", label)
    with c2: st.metric("Pneumonia probability", f"{prob:.3f}")
    st.caption(f"Threshold = {used_T:.2f} • Output = {label}")

    # Fast explanation (batched, coarse->fine)
    with st.spinner("Explaining the decision…"):
        heat = fast_explain(session, meta, preview,
                            coarse_grid=int(coarse), top_k=int(top_k),
                            refine_factor=int(refine), mask_mode=mask_mode)
        # upscale heatmap to image size
        heat_img = Image.fromarray((heat*255).astype("uint8")).resize(preview.size, Image.BILINEAR)
        heat_up = np.asarray(heat_img).astype("float32") / 255.0
        overlay = overlay_heatmap(preview, heat_up, alpha=0.45)
        boxed = draw_bbox_on_overlay(overlay, heat_up, frac=0.6)

    cc1, cc2 = st.columns(2)
    with cc1: st.image(pil_to_png_bytes(Image.fromarray((heat_up*255).astype("uint8"))), caption="Importance heatmap")
    with cc2: st.image(pil_to_png_bytes(boxed), caption="Overlay with region box")

    # Natural-language explanation
    st.markdown("#### What this means")
    if label == POS_LABEL:
        st.markdown(
            f"- The model sees **pneumonia-like patterns** with probability **{prob:.2f}** (≥ {used_T:.2f}).\n"
            f"- {region_summary_text(heat_up)}\n"
            f"- Red areas show **where** the image changed the model’s score the most when hidden (i.e., likely consolidation/infiltrates)."
        )
    else:
        st.markdown(
            f"- The model predicts **NORMAL** with probability **{1.0 - prob:.2f}** (pneumonia prob {prob:.2f} < {used_T:.2f}).\n"
            f"- Any faint red areas indicate **weak** evidence for pneumonia; overall impact was low.\n"
            f"- {region_summary_text(heat_up)}"
        )

st.success("Done.")
