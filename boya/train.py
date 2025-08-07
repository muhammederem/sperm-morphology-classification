import os
import random
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score

from model import create_model
from utils import setup_logger, save_model, log_confusion_matrix_tensorboard, log_config_as_text
from config import load_config

from torch.utils.tensorboard import SummaryWriter


# --- Dataset (fold'lara göre veri ayırma) ---
class FoldDataset(Dataset):
    def __init__(self, root_dir, folds, subset="train", transform=None, class_to_idx=None):
        """
        root_dir: Boya1 dizini
        folds: list of fold names, örn: ["fold1"] veya ["fold2","fold3","fold4","fold5"]
        subset: "train" veya "test" klasörü (fold içindeki alt klasör)
        transform: torchvision transformları
        class_to_idx: sınıf isimleri ve indeks eşlemesi (tutarlı olması için)
        """
        self.images = []
        self.labels = []
        self.class_to_idx = class_to_idx
        self.transform = transform

        for fold in folds:
            fold_path = os.path.join(root_dir, fold, subset)
            classes = sorted(os.listdir(fold_path))
            if self.class_to_idx is None:
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

            for cls_name in classes:
                cls_dir = os.path.join(fold_path, cls_name)
                files = glob(os.path.join(cls_dir, "*.jpg"))
                self.images.extend(files)
                self.labels.extend([self.class_to_idx[cls_name]] * len(files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label




def train_one_epoch(model, dataloader, criterion, optimizer, device, logger):
    model.train()
    running_loss = 0
    all_preds = []
    all_labels = []

    # Wrap dataloader with tqdm for progress bar
    loop = tqdm(dataloader, desc="Training", leave=True)

    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Update progress bar postfix with current loss
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    logger.info(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc



def test(model, dataloader, criterion, device, logger, writer, class_names, global_step):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    logger.info(f"Test Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f}")

    writer.add_scalar('Test/Loss', epoch_loss, global_step)
    writer.add_scalar('Test/Accuracy', epoch_acc, global_step)
    log_confusion_matrix_tensorboard(writer, all_labels, all_preds, class_names, global_step)

    return epoch_loss, epoch_acc


def main():
    import logging
    config = load_config("config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = config["paths"]["data_dir"]
    folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    for train_fold in folds:
        test_folds = [f for f in folds if f != train_fold]

        # Tutarlı class_names ve class_to_idx oluştur (train_fold'un train klasöründen)
        class_names = sorted(os.listdir(os.path.join(data_dir, train_fold, "train")))
        num_classes = len(class_names)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        # Logger setup (fold bazlı)
        os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
        log_file = os.path.join(config["paths"]["logs_dir"], f"train_log_{train_fold}.txt")
        setup_logger(log_file)
        logger = logging.getLogger()
        logger.info(f"Starting training for {train_fold}, testing on {test_folds}")

        # TensorBoard writer (fold bazlı)
        writer = SummaryWriter(log_dir=os.path.join(config["paths"]["tensorboard_runs_dir"], train_fold))
        log_config_as_text(writer, config)

        # Dataset ve DataLoader
        train_dataset = FoldDataset(data_dir, folds=[train_fold], subset="train", transform=train_transform, class_to_idx=class_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=8)

        test_datasets = {
            fold: FoldDataset(data_dir, folds=[fold], subset="test", transform=test_transform, class_to_idx=class_to_idx)
            for fold in test_folds
        }
        test_loaders = {
            fold: DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=8)
            for fold, ds in test_datasets.items()
        }

        # Model ve optimizer
        model = create_model(num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                     weight_decay=config["training"]["weight_decay"])

        best_acc = 0
        epochs = config["training"]["epochs"]

        for epoch in range(epochs):
            logger.info(f"Epoch [{epoch + 1}/{epochs}] - Training fold {train_fold}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)

            # Test tüm test foldlarında
            for fold, test_loader in test_loaders.items():
                test_loss, test_acc = test(model, test_loader, criterion, device, logger, writer, class_names, epoch)
                logger.info(f"Fold {fold} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

            # En iyi modeli kaydet (eğitim folduna göre)
            if train_acc > best_acc:
                best_acc = train_acc
                save_path = os.path.join(config["paths"]["models_dir"], f"best_model_{train_fold}.pt")
                save_model(model, save_path)
                logger.info(f"Model saved at {save_path}")

        writer.close()
        logger.info(f"Training finished for {train_fold}\n\n")


if __name__ == "__main__":
    main()
