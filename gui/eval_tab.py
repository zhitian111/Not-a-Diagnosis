# ui/eval_tab.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QFrame
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

from .eval_io import load_eval_folder
from .eval_plots import plot_roc, plot_pr, plot_confusion_matrix


class EvalTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self._apply_style()

    # ================= UI =================
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ---------- Title ----------
        title = QLabel("📊 Model Performance Evaluation")
        title.setObjectName("TitleLabel")
        root.addWidget(title)

        # ---------- Control Bar ----------
        control_card = self._card()
        ctrl_layout = QHBoxLayout(control_card)

        self.load_btn = QPushButton("导入评估记录")
        self.load_btn.clicked.connect(self.on_load_clicked)

        self.info_label = QLabel("未加载评估记录")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        ctrl_layout.addWidget(self.load_btn)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.info_label)

        root.addWidget(control_card)

        # ---------- Plot Area ----------
        plot_card = self._card()
        plot_layout = QVBoxLayout(plot_card)

        self.fig = Figure(constrained_layout=True)
        self.canvas = Canvas(self.fig)

        plot_layout.addWidget(self.canvas)
        root.addWidget(plot_card, stretch=1)

        self.canvas.setStyleSheet("""
        background-color: black;
        border: 2px solid #4C566A;
        border-radius: 4px;
        """)

    def _card(self):
        card = QFrame()
        card.setObjectName("Card")
        card.setFrameShape(QFrame.Shape.StyledPanel)
        return card

    # ================= Style =================
    def _apply_style(self):
        self.setStyleSheet("""
        QWidget {
            background-color: #F7F9FC;
            font-family: "Segoe UI";
            font-size: 14px;
        }

        QLabel {
            color: #2E3440;
        }

        QLabel#TitleLabel {
            font-size: 22px;
            font-weight: 600;
            padding: 6px 0;
        }

        QFrame#Card {
            background-color: #FFFFFF;
            border: 1px solid #E1E5EE;
            border-radius: 10px;
        }

        QPushButton {
            background-color: #4C72B0;
            color: white;
            padding: 6px 14px;
            border-radius: 6px;
        }

        QPushButton:hover {
            background-color: #5A83C7;
        }

        QPushButton:pressed {
            background-color: #3E5F99;
        }
        """)

    # ================= Logic =================
    def on_load_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择评估记录文件夹"
        )
        if not folder:
            return

        try:
            probs, labels, cm, metrics = load_eval_folder(folder)
        except Exception as e:
            self.info_label.setText(f"加载失败: {e}")
            return

        acc = metrics.get("acc", "N/A")
        auc = metrics.get("auc", "N/A")

        self.info_label.setText(
            f"Samples: {len(labels)} | Acc: {acc} | AUC: {auc}"
        )

        self._draw_plots(probs, labels, cm)


    def _draw_metrics(self, ax, labels, probs):
        preds = (probs > 0.5).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        text = (
            f"Accuracy : {acc:.4f}\n\n"
            f"Precision: {prec:.4f}\n"
            f"Recall   : {recall:.4f}\n"
            f"F1-score : {f1:.4f}"
        )

        ax.text(
            0.05, 0.95,
            text,
            va="top",
            ha="left",
            fontsize=12,
            family="monospace"
        )

    def _draw_plots(self, probs, labels, cm):
        self.fig.clf()

        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(222)
        ax3 = self.fig.add_subplot(223)
        ax4 = self.fig.add_subplot(224)

        plot_roc(ax1, labels, probs)
        plot_pr(ax2, labels, probs)
        plot_confusion_matrix(ax3, cm)

        # 第四个：指标文本
        ax4.axis("off")
        self._draw_metrics(ax4, labels, probs)

        self.canvas.draw_idle()

