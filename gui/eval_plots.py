# utils/eval_plots.py
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_roc(ax, labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.5)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

def plot_pr(ax, labels, probs):
    precision, recall, _ = precision_recall_curve(labels, probs)

    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)

def plot_confusion_matrix(ax, cm):
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
