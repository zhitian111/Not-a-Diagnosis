import torch
from model.ResNet3D18 import ResNet3D18

device = "cpu"
model = ResNet3D18().to(device)

ckpt = torch.load("./train_results/20251215-233951/models/ResNet/checkpoints/best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dummy_input = torch.randn(1, 1, 64, 64, 64)

torch.onnx.export(
    model,
    dummy_input,
    "ResNet3D18.onnx",
    input_names=["input"],
    output_names=["malignancy_prob"],
    opset_version=11,
    do_constant_folding=True
)
