import torch
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import Qt
from .OrthogonalView import OrthogonalView

from model.Simple3DCNN import Simple3DCNN
from model.ResNet3D18 import ResNet3D18


class InferenceTab(QWidget):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = None
        self.ckpt_path = None
        self.patch_path = None

        self._init_ui()

        self.patch_volume = None
        self.current_z = 32

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(16)

        # ================= 模型配置 =================
        model_box = QGroupBox("Model Configuration")
        model_layout = QHBoxLayout(model_box)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["Simple3DCNN", "ResNet3D18"])

        self.ckpt_btn = QPushButton("Load Checkpoint")
        self.ckpt_btn.clicked.connect(self._load_checkpoint)

        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()
        model_layout.addWidget(self.ckpt_btn)

        # ================= Patch 输入 =================
        patch_box = QGroupBox("Patch Input")
        patch_layout = QHBoxLayout(patch_box)

        self.patch_label = QLabel("No patch selected")
        self.patch_btn = QPushButton("Load Patch (.npy)")
        self.patch_btn.clicked.connect(self._load_patch)

        patch_layout.addWidget(self.patch_label)
        patch_layout.addStretch()
        patch_layout.addWidget(self.patch_btn)
        # ================= Patch Visualization =================
        vis_box = QGroupBox("Patch Visualization")
        vis_layout = QVBoxLayout(vis_box)
        vis_box.setStyleSheet("""
        QGroupBox {
            background-color: #ECEFF4;
        }
        """)

        self.ortho_view = OrthogonalView()

        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(63)
        self.z_slider.setValue(32)
        self.z_slider.valueChanged.connect(self._on_z_changed)

        vis_layout.addWidget(self.ortho_view)
        vis_layout.addWidget(QLabel("Z Slice"))
        vis_layout.addWidget(self.z_slider)

        main_layout.addWidget(vis_box)

        # ================= 推理 =================
        infer_box = QGroupBox("Inference")
        infer_layout = QVBoxLayout(infer_box)

        self.run_btn = QPushButton("▶ Run Inference")
        self.run_btn.setFixedHeight(40)
        self.run_btn.clicked.connect(self._run_inference)

        infer_layout.addWidget(self.run_btn)

        # ================= 结果 =================
        result_box = QGroupBox("Result")
        result_layout = QVBoxLayout(result_box)

        self.prob_label = QLabel("Malignancy Probability: --")
        self.diag_label = QLabel("Diagnosis: --")
        self.prob_label.setObjectName("ResultLabel")
        self.diag_label.setObjectName("ResultLabel")
        self.prob_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prob_label.setStyleSheet("font-size: 16px;")
        self.diag_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        result_layout.addWidget(self.prob_label)
        result_layout.addWidget(self.diag_label)

        # ================= Assemble =================
        main_layout.addWidget(model_box)
        main_layout.addWidget(patch_box)
        main_layout.addWidget(infer_box)
        main_layout.addWidget(result_box)
        main_layout.addStretch()

    # ------------------------------------------------
    # Logic
    # ------------------------------------------------
    def _load_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "", "Checkpoint (*.pt *.pth)"
        )
        if not path:
            return

        self.ckpt_path = path
        self._load_model()
        self.ckpt_btn.setText("Checkpoint Loaded ✔")

    def _load_model(self):
        model_name = self.model_selector.currentText()

        if model_name == "Simple3DCNN":
            self.model = Simple3DCNN()
        else:
            self.model = ResNet3D18()

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _load_patch(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select patch", "", "NumPy file (*.npy)"
        )
        if not path:
            return

        self.patch_path = path
        self.patch_label.setText(path.split("/")[-1])

        self.patch_volume = np.load(path)
        self.current_z = self.patch_volume.shape[2] // 2

        self.z_slider.setMaximum(self.patch_volume.shape[2] - 1)
        self.z_slider.setValue(self.current_z)

        self.ortho_view.update(self.patch_volume, self.current_z)

    def _run_inference(self):
        if self.model is None or self.patch_path is None:
            self.diag_label.setText("⚠ Please load model and patch first")
            return

        patch = np.load(self.patch_path)  # (64,64,64)
        patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        patch = patch.to(self.device)

        with torch.no_grad():
            logit = self.model(patch)
            prob = torch.sigmoid(logit).item()

        self.prob_label.setText(f"Malignancy Probability: {prob:.4f}")

        if prob >= 0.5:
            self.diag_label.setText("🔴 Diagnosis: Malignant")
            self.diag_label.setStyleSheet("color: red; font-size: 20px; font-weight: bold;")
        else:
            self.diag_label.setText("🟢 Diagnosis: Benign")
            self.diag_label.setStyleSheet("color: green; font-size: 20px; font-weight: bold;")
    def _on_z_changed(self, value):
        if self.patch_volume is None:
            return
        self.current_z = value
        self.ortho_view.update(self.patch_volume, value)
