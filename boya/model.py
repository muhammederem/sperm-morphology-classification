"""
Main model factory for the Boya project.
This module provides a unified interface to create models from the models package.
"""

from models import create_model as create_model_from_package

def create_model(num_classes, model_name='resnet50', use_bilinear_pooling=True, pretrained=True):
    """
    Create a model with specified configuration
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the base model (e.g., 'resnet50', 'efficientnet_b0', 'densenet121')
        use_bilinear_pooling: Whether to use bilinear pooling
        pretrained: Whether to use pretrained weights
    
    Returns:
        The created model
    """
    return create_model_from_package(
        num_classes=num_classes,
        model_name=model_name,
        use_bilinear_pooling=use_bilinear_pooling,
        pretrained=pretrained
    )
