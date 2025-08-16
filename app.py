# app.py — Streamlit UI for Pediatric Pneumonia (auto-detects DenseNet121 or EfficientNet-B0)

from pathlib import Path
import json
import numpy as np
from PIL import Image, UnidentifiedImageError

import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import streamlit as st, torch, torchvision
st.sidebar.write("Torch:", torch.__version__, "| TV:", torchvision.__version__)


# ---------- PATHS (match your screenshots) ----------
# Your model file sits directly inside: C:\Users\itsra\PneumoniaUI\pediatric_pneumonia_xray\
BASE = Path(__file__).parent / "pediatric_pneumonia_xray"
CKPT_PATH    = BASE / "pneumonia_densenet_model.pth"   # your .pth file
METRICS_JSON = BASE / "test_metrics.json"              # optional

# If you later move files elsewhere, you can instead set absolute paths like:
# CKPT_PATH    = Path(r"C:\full\path\to\pneumonia_densenet_model.pth")
# METRICS_JSON = Path(r"C:\full\path\to\test_metrics.json")

# ---------- HELPERS ----------
def _get_module(model, dotted: str):
    m = model
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m

# ---------- MODEL LOADER (decorator must be above the function) ----------
@st.cache_resource(show_spinner=False)
def load_model_and_meta():
    import json
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision import models

    device = torch.device("cpu")
    if not CKPT_PATH.exists():
        return None, None, 0.50, "features", "Unknown"

    obj = torch.load(CKPT_PATH, map_location=device)

    # defaults if metadata missing
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    dropout = 0.5

    def build_densenet(drop=0.5):
        m = models.densenet121(weights=None)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(nn.Dropout(drop, inplace=True), nn.Linear(in_f, 1))
        return m, "features.denseblock4", "DenseNet121"

    def build_efficientnet(drop=0.5):
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(drop, inplace=True), nn.Linear(in_f, 1))
        return m, "features.6", "EfficientNet-B0"

    def first_present(sd, keys):
        for k in keys:
            if k in sd:
                return sd[k]
        return None

    def maybe_strip_prefix(sd, prefix):
        keys = list(sd.keys())
        if keys and all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    # ---- A) checkpoint dict with 'state_dict' or 'model_state' ----
    if isinstance(obj, dict) and ("state_dict" in obj or "model_state" in obj):
        sd = obj.get("state_dict", obj.get("model_state"))
        sd = maybe_strip_prefix(sd, "module.")
        sd = maybe_strip_prefix(sd, "model.")

        img_size = obj.get("img_size", img_size)
        mean     = obj.get("mean", mean)
        std      = obj.get("std", std)
        dropout  = obj.get("dropout", dropout)
        model_name = (obj.get("model_name") or "").lower()

        # Infer backbone if not stored
        if not model_name:
            w = first_present(sd, ["classifier.1.weight", "classifier.weight"])
            in_feats = int(w.shape[1]) if w is not None else None
            if in_feats == 1280 or any(k.startswith("features.0.0") for k in sd):
                model_name = "efficientnet_b0"
            elif in_feats == 1024 or any("denseblock" in k for k in sd):
                model_name = "densenet121"
            else:
                model_name = "densenet121"

        if model_name == "efficientnet_b0":
            m, target_layer, backbone = build_efficientnet(dropout)
        else:
            m, target_layer, backbone = build_densenet(dropout)

        m.load_state_dict(sd, strict=False)

    # ---- B) raw state_dict only ----
    elif isinstance(obj, dict):
        sd = maybe_strip_prefix(obj, "module.")
        sd = maybe_strip_prefix(sd, "model.")
        w = first_present(sd, ["classifier.1.weight", "classifier.weight"])
        in_feats = int(w.shape[1]) if w is not None else None
        if in_feats == 1280 or any(k.startswith("features.0.0") for k in sd):
            m, target_layer, backbone = build_efficientnet(dropout)
        else:
            m, target_layer, backbone = build_densenet(dropout)
        m.load_state_dict(sd, strict=False)

    # ---- C) whole nn.Module saved ----
    else:
        m = obj
        backbone = "EfficientNet-B0" if "efficientnet" in str(type(m)).lower() else "DenseNet121"
        target_layer = "features.denseblock4" if hasattr(m, "classifier") else "features"

    m.eval()

    tfm = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    tuned_thr = 0.50
    if METRICS_JSON.exists():
        try:
            with open(METRICS_JSON) as f:
                tuned_thr = float(json.load(f).get("best_threshold", 0.50))
        except Exception:
            pass

    return m, tfm, tuned_thr, target_layer, backbone


def predict_prob(model, tfm, pil_img: Image.Image) -> float:
    with torch.inference_mode():
        x = tfm(pil_img.convert("RGB")).unsqueeze(0)
        return float(torch.sigmoid(model(x)).item())

def gradcam_overlay(model, tfm, target_layer_name: str, pil_img: Image.Image):
    import cv2
    x = tfm(pil_img.convert("RGB")).unsqueeze(0)
    acts, grads = {}, {}
    layer = _get_module(model, target_layer_name)

    def fwd_hook(m, i, o): acts["v"] = o.detach()
    def bwd_hook(m, gi, go): grads["v"] = go[0].detach()
    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    prob = torch.sigmoid(logits).item()
    logits[:, 0].backward(retain_graph=True)

    A, G = acts["v"], grads["v"]
    w = G.mean(dim=(2,3), keepdim=True)
    cam = torch.relu((w*A).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-6)

    base = np.array(pil_img.convert("RGB"))
    H, W = base.shape[:2]
    cam_res = cv2.resize(cam, (W, H))
    heat = cv2.applyColorMap(np.uint8(255 * cam_res), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base, 0.5, heat, 0.5, 0)
    from PIL import Image as _Image
    overlay = _Image.fromarray(overlay[:, :, ::-1])  # BGR→RGB
    h1.remove(); h2.remove()
    return overlay, float(prob)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Pediatric Pneumonia Detector", page_icon="🫁", layout="centered")
st.title("🫁 Pediatric Pneumonia Detector")

with st.sidebar:
    st.header("Settings")
    model, tfm, tuned_thr, target_layer, backbone = load_model_and_meta()
    if model is None:
        st.error(f"Model file not found.\nPlease check CKPT_PATH in app.py:\n{CKPT_PATH}")
        st.stop()
    st.info(f"Backbone: **{backbone}**", icon="🧠")
    use_tuned = st.toggle("Use tuned threshold", value=True)
    thr = tuned_thr if use_tuned else 0.50
    thr = st.slider("Classification threshold", 0.05, 0.95, float(thr), 0.01)
    show_cam = st.toggle("Show Grad-CAM", value=True)
    if METRICS_JSON.exists():
        st.caption(f"Tuned threshold from metrics: {tuned_thr:.2f}")

st.subheader("Upload a chest X-ray")
file = st.file_uploader("Drop a JPG/PNG/TIFF here", type=["jpg","jpeg","png","tif","tiff"], accept_multiple_files=False)

if file:
    try:
        img = Image.open(file)
    except UnidentifiedImageError:
        st.error("This file doesn't look like a valid image.")
        st.stop()

    st.image(img, caption=file.name, use_container_width=True)
    with st.spinner("Running inference…"):
        prob = predict_prob(model, tfm, img)
        pred = "PNEUMONIA" if prob >= thr else "NORMAL"
    st.success(f"Prediction: **{pred}**  |  Probability (Pneumonia): **{prob:.3f}**  |  Threshold: {thr:.2f}")

    if show_cam:
        with st.spinner("Computing Grad-CAM…"):
            cam, _ = gradcam_overlay(model, tfm, target_layer, img)
        st.image(cam, caption="Grad-CAM overlay", use_container_width=True)

st.markdown("---")
st.caption("Educational demo only — not a medical device.")
