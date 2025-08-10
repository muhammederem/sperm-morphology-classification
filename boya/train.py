import os
import random
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

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
            classes = sorted([cls for cls in os.listdir(fold_path) if not cls.startswith('.')])  # Skip hidden files
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
    all_probs = []

    # Wrap dataloader with tqdm for progress bar
    loop = tqdm(dataloader, desc="Training", leave=True)

    for imgs, labels in loop:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(labels.cpu().detach().numpy())
        all_probs.extend(probs.cpu().detach().numpy())

        # Update progress bar postfix with current loss
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate confidence metrics
    confidences = [max(prob) for prob in all_probs]
    avg_confidence = sum(confidences) / len(confidences)
    
    logger.info(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Avg Confidence: {avg_confidence:.4f}")
    return epoch_loss, epoch_acc



def test(model, dataloader, criterion, device, logger, writer, class_names, global_step):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate confidence metrics
    confidences = [max(prob) for prob in all_probs]
    avg_confidence = sum(confidences) / len(confidences)
    
    logger.info(f"Test Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Avg Confidence: {avg_confidence:.4f}")

    # Log all metrics to TensorBoard
    writer.add_scalar('Test/Loss', epoch_loss, global_step)
    writer.add_scalar('Test/Accuracy', epoch_acc, global_step)
    writer.add_scalar('Test/Precision', precision, global_step)
    writer.add_scalar('Test/Recall', recall, global_step)
    writer.add_scalar('Test/F1_Score', f1, global_step)
    writer.add_scalar('Test/Average_Confidence', avg_confidence, global_step)
    
    # Log confusion matrix
    log_confusion_matrix_tensorboard(writer, all_labels, all_preds, class_names, global_step)

    return epoch_loss, epoch_acc


def main():
    import logging
    config = load_config("config.yaml")

    # Device selection
    device_config = config["training"]["device"]
    if device_config == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_config)
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

        # Create descriptive run name for TensorBoard
        model_name = config["model"]["base_model"]
        bilinear_suffix = "_bilinear" if config["model"]["use_bilinear_pooling"] else "_standard"
        pretrained_suffix = "_pretrained" if config["model"]["pretrained"] else "_from_scratch"
        run_name = f"{model_name}{bilinear_suffix}{pretrained_suffix}_fold{train_fold}"
        
        # Logger setup (fold bazlı)
        os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
        log_file = os.path.join(config["paths"]["logs_dir"], f"{run_name}_training.log")
        setup_logger(log_file)
        logger = logging.getLogger()
        logger.info(f"Starting training for {run_name}")
        logger.info(f"Model: {model_name}, Bilinear: {config['model']['use_bilinear_pooling']}, Pretrained: {config['model']['pretrained']}")
        logger.info(f"Training on {train_fold}, testing on {test_folds}")
        logger.info(f"Device: {device}, Batch size: {config['training']['batch_size']}, Epochs: {config['training']['epochs']}")

        # TensorBoard writer with descriptive naming
        tensorboard_dir = os.path.join(config["paths"]["tensorboard_runs_dir"], run_name)
        writer = SummaryWriter(log_dir=tensorboard_dir)
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
        model = create_model(
            num_classes=num_classes,
            model_name=config["model"]["base_model"],
            use_bilinear_pooling=config["model"]["use_bilinear_pooling"],
            pretrained=config["model"]["pretrained"]
        )
        model = model.to(device)
        
        # Log model architecture after model is created
        logger.info(f"Model architecture: {model}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        writer.add_text('Model/Architecture', str(model), 0)
        writer.add_scalar('Model/Total_Parameters', total_params, 0)
        writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"],
                                     weight_decay=config["training"]["weight_decay"])

        best_acc = 0
        epochs = config["training"]["epochs"]

        for epoch in range(epochs):
            logger.info(f"Epoch [{epoch + 1}/{epochs}] - Training {run_name}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, logger)
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Accuracy', train_acc, epoch)
            
            # Note: Additional metrics (precision, recall, f1, confidence) are calculated in train_one_epoch
            # but we'll add them here for TensorBoard logging
            from sklearn.metrics import precision_score, recall_score, f1_score
            # We need to recalculate these for TensorBoard logging
            # This is a bit redundant but ensures we have all metrics in TensorBoard
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
            
            # Log epoch progress
            progress = (epoch + 1) / epochs * 100
            writer.add_scalar('Progress/Epoch_Progress', progress, epoch)
            writer.add_scalar('Progress/Best_Accuracy_So_Far', best_acc, epoch)

            # Test tüm test foldlarında
            test_results = {}
            for fold, test_loader in test_loaders.items():
                test_loss, test_acc = test(model, test_loader, criterion, device, logger, writer, class_names, epoch)
                test_results[fold] = {'loss': test_loss, 'accuracy': test_acc}
                logger.info(f"Fold {fold} Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
                
                # Log test metrics with fold-specific tags
                writer.add_scalar(f'Test/{fold}/Loss', test_loss, epoch)
                writer.add_scalar(f'Test/{fold}/Accuracy', test_acc, epoch)
            
            # Log average test metrics
            avg_test_loss = sum(r['loss'] for r in test_results.values()) / len(test_results)
            avg_test_acc = sum(r['accuracy'] for r in test_results.values()) / len(test_results)
            writer.add_scalar('Test/Average/Loss', avg_test_loss, epoch)
            writer.add_scalar('Test/Average/Accuracy', avg_test_acc, epoch)
            
            logger.info(f"Average Test Loss: {avg_test_loss:.4f}, Average Test Acc: {avg_test_acc:.4f}")

            # En iyi modeli kaydet (eğitim folduna göre)
            if train_acc > best_acc:
                best_acc = train_acc
                
                if config["model"]["save_naming"]:
                    # Descriptive naming
                    model_name = config["model"]["base_model"]
                    bilinear_suffix = "_bilinear" if config["model"]["use_bilinear_pooling"] else "_standard"
                    pretrained_suffix = "_pretrained" if config["model"]["pretrained"] else "_from_scratch"
                    save_path = os.path.join(config["paths"]["models_dir"], 
                                           f"{model_name}{bilinear_suffix}{pretrained_suffix}_fold{train_fold}_acc{train_acc:.4f}.pt")
                else:
                    # Simple naming
                    save_path = os.path.join(config["paths"]["models_dir"], f"best_model_{train_fold}.pt")
                
                save_model(model, save_path)
                logger.info(f"Model saved at {save_path}")

        writer.close()
        logger.info(f"Training finished for {train_fold}\n\n")
    
    # Run comprehensive testing after ALL folds are complete (if configured)
    if config["testing"]["run_after_training"]:
        logger.info("All folds training completed. Running comprehensive testing on all folds...")
        try:
            from test_all_folds import main as test_main
            test_main()
            logger.info("Comprehensive testing completed successfully!")
        except Exception as e:
            logger.error(f"Error during comprehensive testing: {e}")
    else:
        logger.info("Skipping comprehensive testing (disabled in config)")


if __name__ == "__main__":
    main()
