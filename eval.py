import torch
from model.Simple3DCNN import Simple3DCNN
from model.ResNet3D18 import ResNet3D18
from dataset import *
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm

logger = None
def eval(args, _logger):
    device_name = args.device
    global logger
    logger = _logger
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    if args.ckpt_path is None:
        logger.error(f"[ERROR] No checkpoint provided.")
        return

    if args.method == "3DCNN":
        model = Simple3DCNN()
        ckpt = torch.load(args.ckpt_path, map_location = device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        dataset = LIDCNoduleDataset("dataset")
        all_probs, all_labels = [], []
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        preds = (all_probs > 0.5).astype(int)

        acc = (preds == all_labels).mean()
        cm = confusion_matrix(all_labels, preds)
        auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"[INFO] Accuracy: {acc}, AUC: {auc}")
        logger.info(f"[INFO] Confusion matrix:\n{cm}")
        logger.info(f"[INFO] Finished evaluation.")
    elif args.method == "ResNet":
        model = ResNet3D18()
        ckpt = torch.load(args.ckpt_path, map_location = device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        dataset = LIDCNoduleDataset("dataset")
        all_probs, all_labels = [], []
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        preds = (all_probs > 0.5).astype(int)

        acc = (preds == all_labels).mean()
        cm = confusion_matrix(all_labels, preds)
        auc = roc_auc_score(all_labels, all_probs)
        logger.info(f"[INFO] Accuracy: {acc}, AUC: {auc}")
        logger.info(f"[INFO] Confusion matrix:\n{cm}")
        logger.info(f"[INFO] Finished evaluation.")