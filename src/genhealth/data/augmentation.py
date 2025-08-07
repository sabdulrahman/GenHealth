import numpy as np
import torch
from typing import Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalAugmentation:
    """
    Medical-specific data augmentation for training.
    """
    
    def __init__(self, image_size: int = 224, strength: str = "medium"):
        self.image_size = image_size
        self.strength = strength
        
        # Create augmentation pipeline based on strength
        self.transforms = self._create_transforms()
    
    def _create_transforms(self):
        """Create augmentation transforms based on strength."""
        
        if self.strength == "light":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        elif self.strength == "medium":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        else:  # strong
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.15, scale_limit=0.15, rotate_limit=30, p=0.7
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.7
                ),
                A.GaussianBlur(blur_limit=(3, 9), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.GridDistortion(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply augmentations to image."""
        augmented = self.transforms(image=image)
        return augmented['image']