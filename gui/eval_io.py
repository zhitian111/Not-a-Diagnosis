# utils/eval_io.py
import os
import json
import numpy as np

def safe_load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    arr = np.load(path)
    if arr.size == 0:
        raise ValueError(f"{path} is empty")
    return arr

def load_eval_folder(folder: str):
    """
    返回:
        probs: np.ndarray (N,)
        labels: np.ndarray (N,)
        cm: np.ndarray (2,2)
        metrics: dict
    """
    probs = safe_load_npy(os.path.join(folder, "probs.npy"))
    labels = safe_load_npy(os.path.join(folder, "labels.npy"))
    cm = safe_load_npy(os.path.join(folder, "confusion_matrix.npy"))

    metrics_path = os.path.join(folder, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return probs, labels, cm, metrics
