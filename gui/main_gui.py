from PyQt6.QtWidgets import QTabWidget, QApplication
from PyQt6.QtGui import QIcon
from .eval_tab import EvalTab
from .inference_tab import InferenceTab
import sys

def load_stylesheet(app):
    with open("gui/style.qss", "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())


def gui_main():
    app = QApplication(sys.argv)
    tabs = QTabWidget()
    tabs.addTab(EvalTab(), "模型性能评估")
    tabs.addTab(InferenceTab(), "模型使用预测")
    tabs.setWindowTitle("肺结节切片恶性概率判别系统")
    tabs.setWindowIcon(QIcon("gui/icon.ico"))
    tabs.show()
    sys.exit(app.exec())