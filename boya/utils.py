import os
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import io
from PIL import Image
from torchvision import transforms


def setup_logger(log_path):
    """
    Verilen log_path'e bir logger başlatır.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s | %(levelname)s | %(message)s',
        level=logging.INFO
    )


def save_model(model, path):
    """
    PyTorch modelini verilen path'e kaydeder.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    """
    Confusion matrix'i çizip PNG olarak kaydeder.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix_tensorboard(y_true, y_pred, class_names):
    """
    Confusion matrix görselini bir matplotlib figürü olarak döndürür (TensorBoard için).
    """
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    return fig


def log_confusion_matrix_tensorboard(writer: SummaryWriter, y_true, y_pred, class_names, global_step, tag="ConfusionMatrix"):
    """
    Confusion matrix görselini TensorBoard'a yazar.
    """
    fig = plot_confusion_matrix_tensorboard(y_true, y_pred, class_names)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image(tag, image, global_step)
    plt.close(fig)

def log_config_as_text(writer, config, global_step=0):
    import yaml
    config_str = yaml.dump(config)
    writer.add_text('config/all', config_str, global_step)

