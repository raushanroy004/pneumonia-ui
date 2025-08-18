# app.py — Streamlit ONNX pneumonia detector with Explainable AI (Occlusion + RISE), CPU-only
from __future__ import annotations
import io, json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

# ========== Paths (change only if needed) ==========
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

# ========== Preprocess constants ==========
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.50
POS_LABEL = "PNEUMONIA"; NEG_LABEL = "NORMAL"

# ========== Small helpers ==========
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
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()

def preprocess(pil: Image.Image, size: int) -> np.ndarray:
    if pil.size != (size, size):
        pil = pil.resize((size, size))
    x = np.asarray(pil).astype("float32") / 255.0     # HWC [0,1]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD            # normalize
    x = np.transpose(x, (2, 0, 1))[None, ...]         # (1,3,H,W)
    return x

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

# ========== Try matplotlib for plots/colormaps (optional) ==========
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    plt = None
    mpl_cm = None

def to_colormap(heat: np.ndarray) -> np.ndarray:
    """heat in [0,1] -> uint8 RGB colormap image."""
    heat = np.clip(heat, 0.0, 1.0)
    if mpl_cm is not None:
        rgb = (mpl_cm.jet(heat)[..., :3] * 255).astype("uint8")
    else:
        # minimal fallback colormap: blue->green->red
        r = np.clip(2*heat - 0.0, 0, 1)
        g = np.clip(2*heat, 0, 1) * (1 - np.abs(heat - 0.5)*2)
        b = np.clip(1 - 2*heat, 0, 1)
        rgb = np.stack([r,g,b], axis=-1)
        rgb = (rgb*255).astype("uint8")
    return rgb

def overlay_heatmap(pil: Image.Image, heat: np.ndarray, alpha: float=0.45) -> Image.Image:
    """Return PIL with heatmap overlay; heat in [0,1] shape HxW."""
    base = pil.copy()
    base = base.resize((heat.shape[1], heat.shape[0]))
    hm_rgb = to_colormap(heat)
    hm_img = Image.fromarray(hm_rgb).convert("RGBA")
    base_rgba = base.convert("RGBA")
    # apply alpha
    hm_img.putalpha(int(alpha * 255))
    out = Image.alpha_composite(base_rgba, hm_img).convert("RGB")
    return out

# ========== ONNX runtime ==========
@st.cache_resource(show_spinner=True)
def load_session_and_meta():
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("onnxruntime is not installed. Add `onnxruntime` to requirements.txt.") from e

    onnx_path = first_existing(CANDIDATE_ONNX)
    if onnx_path is None:
        raise FileNotFoundError("ONNX model not found. Expected at:\n" + "\n".join(map(str, CANDIDATE_ONNX)))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Infer input size from model shape
    try:
        inp = sess.get_inputs()[0]
        shape = inp.shape  # [None, 3, H, W]
        H = int(shape[2]) if isinstance(shape[2], (int, np.integer)) else DEFAULT_IMG_SIZE
        W = int(shape[3]) if isinstance(shape[3], (int, np.integer)) else DEFAULT_IMG_SIZE
        img_size = H if H == W else DEFAULT_IMG_SIZE
    except Exception:
        img_size = DEFAULT_IMG_SIZE

    # Threshold from metrics if available
    metrics_path = first_existing(CANDIDATE_METRICS)
    thr = DEFAULT_THRESHOLD
    if metrics_path is not None:
        try:
            meta = json.loads(metrics_path.read_text())
            thr = float(meta.get("best_threshold", thr))
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

def predict_one(sess, meta: dict, pil: Image.Image, thr: Optional[float]=None) -> Tuple[float, str, float]:
    x = preprocess(pil, meta["img_size"])
    y = sess.run([meta["output_name"]], {meta["input_name"]: x})[0]
    logit = float(np.ravel(y)[0])
    p = sigmoid(logit)
    T = float(meta["best_threshold"] if thr is None else thr)
    label = POS_LABEL if p >= T else NEG_LABEL
    return p, label, T

# ========== XAI: Occlusion ==========
def occlusion_heatmap(sess, meta, pil: Image.Image, patch:int=24, stride:int=12, blur:bool=False) -> np.ndarray:
    """Returns heatmap HxW in [0,1] (importance = drop in prob when occluded)."""
    size = meta["img_size"]
    img = pil.resize((size, size))
    base_prob, _, _ = predict_one(sess, meta, img)

    H, W = size, size
    heat = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    img_arr = np.asarray(img).astype(np.uint8)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y0, y1 = y, min(y + patch, H)
            x0, x1 = x, min(x + patch, W)
            masked = img_arr.copy()
            if blur:
                # simple box blur in patch
                patch_mean = masked[y0:y1, x0:x1].mean(axis=(0,1), keepdims=True).astype(np.uint8)
                masked[y0:y1, x0:x1] = patch_mean
            else:
                masked[y0:y1, x0:x1] = 0
            masked_pil = Image.fromarray(masked)
            prob, _, _ = predict_one(sess, meta, masked_pil)
            drop = float(max(0.0, base_prob - prob))
            heat[y0:y1, x0:x1] += drop
            counts[y0:y1, x0:x1] += 1.0

    heat = np.divide(heat, np.maximum(1.0, counts), out=heat, where=counts>0)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat

# ========== XAI: RISE (Randomized Input Sampling for Explanation) ==========
def rise_heatmap(sess, meta, pil: Image.Image, N:int=400, s:int=7, p_keep:float=0.5) -> np.ndarray:
    """
    RISE: sample N random binary masks on an sxs grid, upsample to input size, average contributions.
    """
    size = meta["img_size"]
    img = pil.resize((size, size))
    x0 = preprocess(img, size)

    H, W = size, size
    heat = np.zeros((H, W), dtype=np.float32)

    for _ in range(N):
        # grid mask s x s, 1 with prob p_keep
        grid = (np.random.rand(s, s) < p_keep).astype("float32")
        # upsample to HxWx1
        mask = np.kron(grid, np.ones((H//s + 1, W//s + 1), dtype="float32"))[:H, :W]
        mask = mask[..., None]  # H W 1
        # apply mask to image
        arr = (np.asarray(img).astype("float32") * mask).astype("uint8")
        prob, _, _ = predict_one(sess, meta, Image.fromarray(arr))
        heat += (mask[...,0] * float(prob)).astype("float32")

    heat /= max(1, N)
    # normalize
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat

# ========== Faithfulness check (Insertion/Deletion) ==========
def insertion_deletion_auc(sess, meta, pil: Image.Image, heat: np.ndarray, steps:int=20) -> Tuple[float, float, Optional[bytes]]:
    """
    Compute insertion and deletion curves & AUCs. Returns (AUC_insert, AUC_delete, plot_png_bytes or None)
    """
    size = meta["img_size"]
    img = pil.resize((size, size))
    arr = np.asarray(img).astype("float32")
    heat = (heat - heat.min()) / max(1e-8, (heat.max() - heat.min()))
    idx = np.dstack(np.unravel_index(np.argsort(-heat, axis=None), heat.shape))[0]  # sorted pixel indices by importance

    # Build step masks
    total = heat.size
    step_px = max(1, total // steps)

    # Start images
    black = np.zeros_like(arr)
    ins_scores, del_scores = [], []

    for k in range(0, total, step_px):
        # Insertion: start black, add top-k pixels
        ins = black.copy()
        inds = idx[:k]
        ins[inds[:,0], inds[:,1]] = arr[inds[:,0], inds[:,1]]
        p_ins, _, _ = predict_one(sess, meta, Image.fromarray(ins.astype("uint8")))

        # Deletion: start original, remove top-k
        dele = arr.copy()
        inds = idx[:k]
        dele[inds[:,0], inds[:,1]] = 0
        p_del, _, _ = predict_one(sess, meta, Image.fromarray(dele.astype("uint8")))

        ins_scores.append(p_ins)
        del_scores.append(p_del)

    # AUC via simple Riemann sum
    x = np.linspace(0, 1, len(ins_scores))
    auc_ins = float(np.trapz(ins_scores, x))
    auc_del = float(np.trapz(1.0 - np.array(del_scores), x))  # higher is better

    plot_bytes = None
    if HAVE_MPL:
        fig = plt.figure(figsize=(4,3), dpi=150)
        plt.plot(x, ins_scores, label=f'Insertion (AUC={auc_ins:.2f})')
        plt.plot(x, del_scores, label=f'Deletion (1-AUC={1-auc_del:.2f})')
        plt.xlabel('Fraction of most-important pixels used/removed')
        plt.ylabel('Pneumonia probability')
        plt.legend(); plt.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        plot_bytes = buf.getvalue()

    return auc_ins, auc_del, plot_bytes

# ========== UI ==========
st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁")

st.title("🫁 Pediatric Pneumonia Detector")
st.caption("DenseNet121 exported to ONNX • CPU-only inference • Occlusion/RISE explanations.")

# Load model
try:
    session, meta = load_session_and_meta()
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

with st.expander("Settings", expanded=False):
    st.write("**Model file:**", meta["onnx_path"].name)
    st.write("**Input size:**", f"{meta['img_size']}×{meta['img_size']}")
    st.write("**Metrics file:**", meta["metrics_path"].name if meta["metrics_path"] else "not found (using 0.50)")
    thr_user = st.slider("Decision threshold", 0.00, 1.00, float(meta["best_threshold"]), 0.01)
    xai_method = st.selectbox("Explanation method", ["Occlusion", "RISE"])
    with st.popover("Advanced XAI parameters"):
        if xai_method == "Occlusion":
            patch = st.slider("Patch size", 8, 48, 24, 2)
            stride = st.slider("Stride", 4, 32, 12, 2)
            blur = st.checkbox("Use blur instead of black masking", value=False)
        else:
            N = st.slider("RISE: number of masks (N)", 100, 1000, 400, 50)
            s = st.slider("RISE: grid size (s×s)", 5, 15, 7, 1)
            p_keep = st.slider("RISE: keep probability", 0.10, 0.90, 0.50, 0.05)

st.subheader("Upload chest X-ray")
files = st.file_uploader("Upload one or more images (PNG/JPG/TIFF/BMP)…",
                         type=["png","jpg","jpeg","bmp","tif","tiff"], accept_multiple_files=True)

if not files:
    st.info("Upload a PNG/JPG to get a prediction and explanation.")
    st.stop()

for i, uf in enumerate(files, start=1):
    st.markdown(f"---\n**Image {i}:** `{uf.name}`")
    data = uf.getvalue()
    if not data:
        st.error("Empty file.")
        continue

    # Preview safely (avoid use_container_width arg for max compatibility)
    try:
        preview = load_pil_from_bytes(data)
        show_pil = preview.copy()
        show_pil.thumbnail((1600, 1600))
        st.image(pil_to_png_bytes(show_pil), caption="Uploaded image")
    except (UnidentifiedImageError, OSError):
        st.info("Could not preview image; still attempting to run inference.")
        try:
            preview = load_pil_from_bytes(data)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            continue

    # Prediction
    prob, label, used_T = predict_one(session, meta, preview)
    c1, c2 = st.columns([1,2])
    with c1: st.metric("Prediction", label)
    with c2: st.metric("Pneumonia probability", f"{prob:.3f}")
    st.caption(f"Threshold = {used_T:.2f} • Output = {label}")

    # XAI heatmap
    with st.spinner(f"Generating {xai_method} explanation…"):
        if xai_method == "Occlusion":
            heat = occlusion_heatmap(session, meta, preview, patch=patch, stride=stride, blur=blur)
        else:
            heat = rise_heatmap(session, meta, preview, N=N, s=s, p_keep=p_keep)

        # upscale heatmap to original preview size
        heat_img = Image.fromarray((heat*255).astype("uint8")).resize(preview.size, Image.BILINEAR)
        heat_up = np.asarray(heat_img).astype("float32") / 255.0
        overlay = overlay_heatmap(preview, heat_up, alpha=0.45)

    cc1, cc2 = st.columns(2)
    with cc1: st.image(pil_to_png_bytes(Image.fromarray((heat_up*255).astype("uint8"))), caption=f"{xai_method} heatmap")
    with cc2: st.image(pil_to_png_bytes(overlay), caption="Overlay")

    # Faithfulness (Insertion/Deletion)
    try:
        auc_ins, auc_del, plot_png = insertion_deletion_auc(session, meta, preview, heat_up, steps=20)
        st.markdown(f"**Faithfulness (higher is better):**  Insertion AUC = `{auc_ins:.2f}` • Deletion AUC = `{auc_del:.2f}`")
        if plot_png is not None:
            st.image(plot_png, caption="Insertion/Deletion curves")
    except Exception as e:
        st.info(f"Faithfulness curves unavailable: {e}")

st.success("Done.")
