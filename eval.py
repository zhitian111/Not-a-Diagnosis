import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score

from model.Simple3DCNN import Simple3DCNN
from model.ResNet3D18 import ResNet3D18
from dataset import LIDCNoduleDataset


def eval(args, logger):
    # ---------------- device ----------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.ckpt_path is None:
        logger.error("[ERROR] No checkpoint provided.")
        return

    # ---------------- model ----------------
    if args.method == "3DCNN":
        model = Simple3DCNN()
        model_name = "Simple3DCNN"
    elif args.method == "ResNet":
        model = ResNet3D18()
        model_name = "ResNet3D18"
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ---------------- dataset ----------------
    dataset = LIDCNoduleDataset("dataset")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False)

    all_probs, all_labels = [], []

    # ---------------- inference ----------------
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()

    preds = (all_probs > 0.5).astype(int)

    # ---------------- metrics ----------------
    acc = float((preds == all_labels).mean())
    auc = float(roc_auc_score(all_labels, all_probs))
    cm = confusion_matrix(all_labels, preds)

    logger.info(f"[INFO] Accuracy: {acc:.4f}")
    logger.info(f"[INFO] AUC: {auc:.4f}")
    logger.info(f"[INFO] Confusion Matrix:\n{cm}")

    # ---------------- save results ----------------
    eval_dir = args.eval_dir or "eval_results"
    save_dir = os.path.join(
        eval_dir,
        model_name,
        os.path.basename(args.ckpt_path).replace(".pth", "")
    )
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "probs.npy"), all_probs)
    np.save(os.path.join(save_dir, "labels.npy"), all_labels)
    np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)

    metrics = {
        "accuracy": acc,
        "auc": auc,
        "num_samples": int(len(all_labels)),
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"[INFO] Evaluation results saved to {save_dir}")
    logger.info("[INFO] Finished evaluation.")
