import os
import numpy as np
import pylidc as pl
from tqdm import tqdm
import configparser

if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser

np.int = int
OUTPUT_ROOT = "./dataset"
PATCH_SIZE = 64

os.makedirs(os.path.join(OUTPUT_ROOT, "benign"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "malignant"), exist_ok=True)
DICOM_ROOT = r"F:\DATASET\LIDC-IDRI\LIDC-IDRI\raw\manifest-1600709154662\LIDC-IDRI"

available_patients = set(os.listdir(DICOM_ROOT))

def crop_patch(volume, center, size=64):
    z, y, x = center
    half = size // 2

    patch = np.zeros((size, size, size), dtype=volume.dtype)

    z1, z2 = z - half, z + half
    y1, y2 = y - half, y + half
    x1, x2 = x - half, x + half

    z1v, z2v = max(0, z1), min(volume.shape[0], z2)
    y1v, y2v = max(0, y1), min(volume.shape[1], y2)
    x1v, x2v = max(0, x1), min(volume.shape[2], x2)

    patch[
        (z1v - z1):(z2v - z1),
        (y1v - y1):(y2v - y1),
        (x1v - x1):(x2v - x1)
    ] = volume[z1v:z2v, y1v:y2v, x1v:x2v]

    return patch


def get_label(nodule):
    scores = [ann.malignancy for ann in nodule]
    avg_score = np.mean(scores)

    if avg_score <= 2:
        return 0
    elif avg_score >= 4:
        return 1
    else:
        return None


def normalize_hu(volume):
    volume = np.clip(volume, -1000, 400)
    volume = (volume + 1000) / 1400
    return volume.astype(np.float32)

error_log = []
discarded_log = []
for scan in tqdm(pl.query(pl.Scan).all()):
    try:
        volume = scan.to_volume()
        nodules = scan.cluster_annotations()

        for i, nodule in enumerate(nodules):
            label = get_label(nodule)
            if label is None:

                discarded_log.append(scan)
                print(f"[WARN] Skip {scan.patient_id}")
                continue

            center = np.mean([ann.centroid for ann in nodule], axis=0).astype(int)
            patch = crop_patch(volume, center, 64)
            patch = normalize_hu(patch)

            subdir = "benign" if label == 0 else "malignant"
            fname = f"{scan.patient_id}_nodule_{i}.npy"

            np.save(os.path.join(OUTPUT_ROOT, subdir, fname), patch)

    except Exception as e:
        error_log.append({
            "patient_id": scan.patient_id,
            "error": str(e)
        })
        print(f"[WARN] Skip {scan.patient_id}: {e}")
        continue

# 保存日志记录
with open(os.path.join(OUTPUT_ROOT, "error.log"), "w") as f:
    f.write("\n".join(error_log))
with open(os.path.join(OUTPUT_ROOT, "discarded.log"), "w") as f:
    f.write("\n".join(discarded_log))