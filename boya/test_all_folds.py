import os
import json
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from model import create_model
from utils import setup_logger, save_confusion_matrix
from config import load_config


class FoldDataset(Dataset):
    def __init__(self, root_dir, folds, subset="test", transform=None, class_to_idx=None):
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


def evaluate_model(model, dataloader, device, class_names, fold_name, model_name):
    """
    Evaluate a model on a specific fold and return detailed metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"Testing {model_name} on {fold_name}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs
    }


def save_results(results, output_dir, model_name):
    """
    Save detailed results to files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall results
    overall_results = {}
    for fold_name, fold_results in results.items():
        overall_results[fold_name] = {
            'accuracy': fold_results['accuracy'],
            'precision': fold_results['classification_report']['weighted avg']['precision'],
            'recall': fold_results['classification_report']['weighted avg']['recall'],
            'f1_score': fold_results['classification_report']['weighted avg']['f1-score']
        }
    
    # Save as JSON
    with open(os.path.join(output_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=2)
    
    # Save detailed classification report
    for fold_name, fold_results in results.items():
        report_path = os.path.join(output_dir, f'{model_name}_{fold_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name} on {fold_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(fold_results['true_labels'], 
                                        fold_results['predictions'], 
                                        target_names=fold_results['class_names']))
    
    # Save confusion matrices
    for fold_name, fold_results in results.items():
        cm_path = os.path.join(output_dir, f'{model_name}_{fold_name}_confusion_matrix.png')
        save_confusion_matrix(fold_results['true_labels'], 
                            fold_results['predictions'], 
                            fold_results['class_names'], 
                            cm_path)


def create_summary_table(all_results, output_dir):
    """
    Create a summary table with all results
    """
    summary_data = []
    
    for model_name, model_results in all_results.items():
        for fold_name, fold_results in model_results.items():
            summary_data.append({
                'Model': model_name,
                'Test Fold': fold_name,
                'Accuracy': f"{fold_results['accuracy']:.4f}",
                'Precision': f"{fold_results['classification_report']['weighted avg']['precision']:.4f}",
                'Recall': f"{fold_results['classification_report']['weighted avg']['recall']:.4f}",
                'F1-Score': f"{fold_results['classification_report']['weighted avg']['f1-score']:.4f}"
            })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'all_results_summary.csv'), index=False)
    
    # Create a pivot table for better visualization
    pivot_df = df.pivot(index='Model', columns='Test Fold', values='Accuracy')
    pivot_df.to_csv(os.path.join(output_dir, 'accuracy_pivot_table.csv'))
    
    return df


def main():
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
    models_dir = config["paths"]["models_dir"]
    results_dir = config["paths"]["results_dir"]
    logs_dir = config["paths"]["logs_dir"]
    
    folds = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    
    # Setup logging
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "test_all_folds.log")
    setup_logger(log_file)
    
    # Transform for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get class names from any fold (they should be consistent)
    class_names = sorted(os.listdir(os.path.join(data_dir, "fold1", "test")))
    num_classes = len(class_names)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create test datasets for all folds
    test_datasets = {}
    test_loaders = {}
    
    for fold in folds:
        test_datasets[fold] = FoldDataset(data_dir, [fold], "test", test_transform, class_to_idx)
        test_loaders[fold] = DataLoader(test_datasets[fold], 
                                       batch_size=config["training"]["batch_size"], 
                                       shuffle=False, 
                                       num_workers=0)  # Set to 0 to avoid multiprocessing issues on macOS
        print(f"Fold {fold}: {len(test_datasets[fold])} test samples")
    
    # Find all trained models
    model_files = glob(os.path.join(models_dir, "*.pt"))
    print(f"Found {len(model_files)} trained models: {[os.path.basename(f) for f in model_files]}")
    
    all_results = {}
    
    # Test each model on all folds
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace('.pt', '')
        # Extract just the base model name for display
        display_name = model_name.split('_')[0] if '_' in model_name else model_name
        print(f"\nTesting {model_name}...")
        
        # Load model
        model = create_model(
            num_classes=num_classes,
            model_name=config["model"]["base_model"],
            use_bilinear_pooling=config["model"]["use_bilinear_pooling"],
            pretrained=config["model"]["pretrained"]
        )
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        
        model_results = {}
        
        # Test on all folds
        for fold in folds:
            print(f"  Testing on {fold}...")
            results = evaluate_model(model, test_loaders[fold], device, class_names, fold, model_name)
            results['class_names'] = class_names
            model_results[fold] = results
            
            print(f"    Accuracy: {results['accuracy']:.4f}")
        
        all_results[model_name] = model_results
        
        # Save individual model results
        save_results(model_results, results_dir, model_name)
    
    # Create summary table
    summary_df = create_summary_table(all_results, results_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL RESULTS")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    # Save summary to file
    with open(os.path.join(results_dir, 'test_summary.txt'), 'w') as f:
        f.write("COMPREHENSIVE TEST RESULTS SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(summary_df.to_string(index=False))
        
        f.write("\n\nDETAILED RESULTS BY MODEL:\n")
        f.write("="*30 + "\n")
        for model_name, model_results in all_results.items():
            f.write(f"\n{model_name}:\n")
            for fold_name, fold_results in model_results.items():
                f.write(f"  {fold_name}: Accuracy={fold_results['accuracy']:.4f}, "
                       f"F1={fold_results['classification_report']['weighted avg']['f1-score']:.4f}\n")
    
    print(f"\nAll results saved to: {results_dir}")
    print(f"Summary saved to: {os.path.join(results_dir, 'test_summary.txt')}")


if __name__ == "__main__":
    main()
