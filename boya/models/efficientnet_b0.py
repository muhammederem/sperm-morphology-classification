import torch
import torch.nn as nn
from torchvision import models
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


class BilinearCNN(nn.Module):
    def __init__(self, base_model, num_classes, feature_dim):
        super(BilinearCNN, self).__init__()
        # Remove the last two layers (avgpool and classifier)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.fc_dim = feature_dim
        self.classifier = nn.Linear(self.fc_dim * self.fc_dim, num_classes)

    def bilinear_pooling(self, x):
        # x: (batch, channels, height, width)
        batch_size, ch, h, w = x.size()
        x = x.view(batch_size, ch, h * w)
        x = torch.bmm(x, x.transpose(1, 2)) / (h * w)  # Bilinear outer product
        x = x.view(batch_size, -1)
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)  # Signed sqrt
        x = nn.functional.normalize(x)  # L2 norm
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.bilinear_pooling(x)
        x = self.classifier(x)
        return x


class StandardCNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super(StandardCNN, self).__init__()
        # Remove the last layer (classifier)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        # Get the feature dimension from the base model
        feature_dim = base_model.classifier[1].in_features
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


def create_model(num_classes, use_bilinear_pooling=True, pretrained=True):
    """
    Create EfficientNet-B0 model
    
    Args:
        num_classes: Number of output classes
        use_bilinear_pooling: Whether to use bilinear pooling
        pretrained: Whether to use pretrained weights
    """
    # Create base EfficientNet-B0 model
    if pretrained:
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        base_model = models.efficientnet_b0(weights=None)
    
    feature_dim = 1280  # EfficientNet-B0 feature dimension
    
    if use_bilinear_pooling:
        model = BilinearCNN(base_model, num_classes, feature_dim)
    else:
        model = StandardCNN(base_model, num_classes)
    
    return model

