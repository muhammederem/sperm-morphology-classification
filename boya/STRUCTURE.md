# Boya Project Structure

## Core Files
```
boya/
├── config.yaml          # Main configuration (change model here!)
├── train.py            # Training script (runs testing automatically)
├── test_all_folds.py   # Testing script
├── model.py            # Model factory
├── utils.py            # Utility functions
├── config.py           # Config loader
├── requirements.txt    # Dependencies
└── README.md          # Usage instructions
```

## Model Files
```
boya/models/
├── __init__.py         # Model package loader
├── resnet50.py         # ResNet50 implementation
├── efficientnet_b0.py  # EfficientNet-B0 implementation
└── densenet121.py      # DenseNet121 implementation
```

## Generated Files
```
boya/
├── best_model_*.pt     # Trained models
├── results/            # Test results and visualizations
├── logs/               # Training logs
└── runs/               # TensorBoard logs
```

## How to Use
1. Edit `config.yaml` → change `base_model: resnet50` to `base_model: efficientnet_b0`
2. Run `python train.py` → trains and tests automatically
3. Check `results/` for outputs

**That's it!** Simple and clean. 🎯

