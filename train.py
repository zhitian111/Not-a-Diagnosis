# /*
#  * Copyright (c) 2025 zhitian111
#  * Released under the MIT license. See LICENSE for details.
#  */
import os
from dataset import LIDCNoduleDataset
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()



logger = None

def simple_3dcnn_train(device, dataset, lr, batch_size, epochs, model_dir):
    from model.Simple3DCNN import Simple3DCNN
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = Simple3DCNN().to(device)
    pos_weight = torch.tensor([686 / 235]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    val_acc = 0
    if not os.path.exists(model_dir + "3DCNN/checkpoints/"):
        os.makedirs(model_dir + "3DCNN/checkpoints/")

    best_save_path = os.path.join(model_dir + "3DCNN/checkpoints/", "best.pth")
    last_save_path = os.path.join(model_dir + "3DCNN/checkpoints/", "last.pth")
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).float()

            optimizer.zero_grad()
            logits = model(x)
            y = y.float()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if logger is not None:
            logger.info(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)  # [B]
                probs = torch.sigmoid(logits)  # [B] ∈ (0,1)
                pred = (probs > 0.5).long()  # [B] → 0 / 1

                correct += (pred == y).sum().item()
                total += y.size(0)
        if logger is not None:
            logger.info(f"Val Acc: {correct / total:.3f}")
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, best_save_path)
            if logger is not None:
                logger.info(f"[INFO] Saved best model at epoch {epoch}")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc
    }, last_save_path)
    logger.info(f"[INFO] Saved last model at epoch {epochs}")
    return model

def resnet_train(device, dataset, lr, batch_size, epochs, model_dir):
    from model.ResNet3D18 import ResNet3D18
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNet3D18().to(device)
    pos_weight = torch.tensor([686 / 235]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_val_acc = 0
    val_acc = 0
    if not os.path.exists(model_dir + "ResNet/checkpoints/"):
        os.makedirs(model_dir + "ResNet/checkpoints/")

    best_save_path = os.path.join(model_dir + "ResNet/checkpoints/", "best.pth")
    last_save_path = os.path.join(model_dir + "ResNet/checkpoints/", "last.pth")
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.float().to(device), y.float().to(device)

            optimizer.zero_grad()

            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            total_loss += loss.item()
        if logger is not None:
            logger.info(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)  # [B]
                probs = torch.sigmoid(logits)  # [B] ∈ (0,1)
                pred = (probs > 0.5).long()  # [B] → 0 / 1

                correct += (pred == y).sum().item()
                total += y.size(0)
        if logger is not None:
            logger.info(f"Val Acc: {correct / total:.3f}")
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc
            }, best_save_path)
            if logger is not None:
                logger.info(f"[INFO] Saved best model at epoch {epoch}")
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc
    }, last_save_path)
    logger.info(f"[INFO] Saved last model at epoch {epochs}")
    return model


def train(args, _logger):
    device_name = args.device
    global logger
    logger = _logger
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    dataset = LIDCNoduleDataset("dataset")

    if args.method == "3DCNN":
        model = simple_3dcnn_train(device=device, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, dataset=dataset, model_dir=args.model_dir)

    elif args.method == "ResNet":
        model = resnet_train(device, dataset, args.lr, args.batch_size, args.epochs, args.model_dir)
