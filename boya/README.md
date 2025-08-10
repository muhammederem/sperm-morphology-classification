# Boya - Configurable Deep Learning Model System

## Quick Start

### 1. Change Models
Simply edit `config.yaml` and change the `base_model` field:

```yaml
model:
  base_model: resnet50        # Change this to switch models
  num_classes: 18
  use_bilinear_pooling: true
  pretrained: true
```

**Available Models:**
- `resnet50` - ResNet50 with bilinear pooling
- `efficientnet_b0` - EfficientNet-B0 with bilinear pooling  
- `densenet121` - DenseNet121 with bilinear pooling

### 2. Train and Test
```bash
# Activate virtual environment
source ../.env/bin/activate

# Train (automatically tests all folds after training)
python train.py

# Or test existing models only
python test_all_folds.py

# Or test existing models only
python test_all_folds.py
```

### 3. Add New Models
To add a new model, create a file in `models/` directory:

```python
# models/your_model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes, use_bilinear_pooling=True, pretrained=True):
    # Your model implementation here
    return model
```

Then just change `base_model: your_model` in config.yaml!

## Configuration Options

### Model Settings
- `base_model`: Model name (resnet50, efficientnet_b0, densenet121)
- `num_classes`: Number of output classes (18 for your dataset)
- `use_bilinear_pooling`: Use bilinear pooling (true/false)
- `pretrained`: Use pretrained weights (true/false)
- `save_naming`: Enable descriptive model naming (true/false)

### Training Settings
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `device`: auto, cuda, mps, or cpu

### Testing Settings
- `run_after_training`: Automatically test after training (true/false)
- `test_all_folds`: Test on all folds (true/false)

## Results & Logging
All results are saved with descriptive names:

### Model Files
- `resnet50_bilinear_pretrained_fold1_acc0.8500.pt`
- `efficientnet_b0_standard_pretrained_fold2_acc0.8200.pt`
- Or simple names: `best_model_fold1.pt` (if save_naming: false)

### Logs & TensorBoard
- `logs/` - Training logs with descriptive names
- `runs/` - TensorBoard logs with descriptive run names
- `results/` - Test results and visualizations

### TensorBoard Features
- **Model Architecture** - Complete model structure and parameters
- **Training Metrics** - Loss, Accuracy, Precision, Recall, F1-Score, Confidence
- **Test Metrics** - Per-fold and average test performance
- **Progress Tracking** - Epoch progress and best accuracy so far
- **Learning Rate** - Current learning rate tracking
- **Confusion Matrices** - Visual confusion matrices for each epoch
