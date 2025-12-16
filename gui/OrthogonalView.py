import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class OrthogonalView(QWidget):
    def __init__(self):
        super().__init__()

        self.fig = Figure(figsize=(6, 2))
        self.canvas = FigureCanvas(self.fig)

        self.ax_axial = self.fig.add_subplot(131)
        self.ax_coronal = self.fig.add_subplot(132)
        self.ax_sagittal = self.fig.add_subplot(133)

        for ax, title in zip(
            [self.ax_axial, self.ax_coronal, self.ax_sagittal],
            ["Axial (XY)", "Coronal (XZ)", "Sagittal (YZ)"]
        ):
            ax.set_title(title)
            ax.axis("off")
        self.canvas.setStyleSheet("""
        background-color: black;
        border: 2px solid #4C566A;
        border-radius: 4px;
        """)

        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)

    def update(self, volume: np.ndarray, z: int):
        self.ax_axial.imshow(volume[:, :, z], cmap="gray", aspect="equal")
        self.ax_coronal.imshow(volume[:, z, :], cmap="gray", aspect="equal")
        self.ax_sagittal.imshow(volume[z, :, :], cmap="gray", aspect="equal")

        for ax in [self.ax_axial, self.ax_coronal, self.ax_sagittal]:
            ax.axis("off")

        self.canvas.draw_idle()
