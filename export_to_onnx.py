# export_to_onnx.py  — one-time local converter
import torch, torch.nn as nn
from torchvision import models
from collections import OrderedDict
from pathlib import Path

CKPT = Path("pneumonia_densenet_model.pth")   # <-- your .pth filename here
OUT  = Path("pneumonia_densenet_model.onnx")  # this will be created

def build_model():
    m = models.densenet121(weights=None)
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, 1))
    return m

def load_sd(model, ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict):
        for k in ("state_dict","model_state_dict","model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]; break
    new = OrderedDict((k[7:], v) if k.startswith("module.") else (k, v) for k, v in sd.items())
    model.load_state_dict(new, strict=False)

if __name__ == "__main__":
    m = build_model(); load_sd(m, CKPT); m.eval()
    dummy = torch.randn(1,3,224,224)
    torch.onnx.export(m, dummy, OUT.as_posix(),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input":{0:"batch"},"logits":{0:"batch"}},
                      opset_version=13)
    print("Wrote:", OUT.resolve())
