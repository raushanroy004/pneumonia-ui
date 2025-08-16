# export_to_onnx.py  — one-time converter
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict
from pathlib import Path

CKPT = Path("pneumonia_densenet_model.pth")
OUT  = Path("pneumonia_densenet_model.onnx")

def build_model():
    m = models.densenet121(weights=None)
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 1))  # binary head like training
    return m

def load_state_dict_robust(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):  # strip DataParallel
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("Loaded. Missing:", missing, "| Unexpected:", unexpected)

if __name__ == "__main__":
    model = build_model()
    load_state_dict_robust(model, CKPT)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, OUT.as_posix(),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13
    )
    print("Wrote:", OUT.resolve())
