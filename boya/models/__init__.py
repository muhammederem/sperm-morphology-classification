"""
Models package for the Boya project.
This package contains different model implementations that can be loaded dynamically.
"""

import importlib
import os
from typing import Optional

def get_available_models():
    """Get list of available model names"""
    models_dir = os.path.dirname(__file__)
    available_models = []
    
    for file in os.listdir(models_dir):
        if file.endswith('.py') and file != '__init__.py':
            model_name = file[:-3]  # Remove .py extension
            available_models.append(model_name)
    
    return available_models

def load_model(model_name: str, num_classes: int, use_bilinear_pooling: bool = True, pretrained: bool = True):
    """
    Load a model by name
    
    Args:
        model_name: Name of the model (e.g., 'resnet50', 'efficientnet_b0')
        num_classes: Number of output classes
        use_bilinear_pooling: Whether to use bilinear pooling
        pretrained: Whether to use pretrained weights
    
    Returns:
        The loaded model
    """
    try:
        # Import the model module
        module = importlib.import_module(f'models.{model_name}')
        
        # Get the create_model function
        create_model_func = getattr(module, 'create_model')
        
        # Create and return the model
        model = create_model_func(
            num_classes=num_classes,
            use_bilinear_pooling=use_bilinear_pooling,
            pretrained=pretrained
        )
        
        return model
        
    except ImportError:
        raise ValueError(f"Model '{model_name}' not found. Available models: {get_available_models()}")
    except AttributeError:
        raise ValueError(f"Model '{model_name}' does not have a 'create_model' function")
    except Exception as e:
        raise ValueError(f"Error loading model '{model_name}': {str(e)}")

def create_model(num_classes: int, model_name: str = 'resnet50', use_bilinear_pooling: bool = True, pretrained: bool = True):
    """
    Factory function to create models (for backward compatibility)
    
    Args:
        num_classes: Number of output classes
        model_name: Name of the model
        use_bilinear_pooling: Whether to use bilinear pooling
        pretrained: Whether to use pretrained weights
    
    Returns:
        The created model
    """
    return load_model(model_name, num_classes, use_bilinear_pooling, pretrained)

