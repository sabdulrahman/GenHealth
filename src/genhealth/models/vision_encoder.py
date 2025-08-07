import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
from typing import Optional, Tuple
import numpy as np


class MedicalVisionEncoder(nn.Module):
    """
    Vision encoder specialized for medical imaging using Vision Transformer.
    Processes X-rays, MRIs, CT scans, and other medical images.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 768,
        image_size: int = 224,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        
        # Load pre-trained Vision Transformer
        self.vision_model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Medical imaging specific layers
        self.medical_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # Projection to target dimension
        self.projection = nn.Sequential(
            nn.Linear(self.vision_model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Medical anomaly detection head
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Region of interest attention
        self.roi_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Image preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode medical images with attention to pathological features.
        
        Args:
            images: Batch of medical images [batch_size, channels, height, width]
            
        Returns:
            Encoded image features
        """
        batch_size = images.size(0)
        
        # Medical-specific preprocessing
        enhanced_images = self.medical_conv(images)
        
        # Vision transformer encoding
        outputs = self.vision_model(pixel_values=enhanced_images)
        
        # Get pooled representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_features = outputs.pooler_output
        else:
            # Use [CLS] token if pooler_output is not available
            pooled_features = outputs.last_hidden_state[:, 0, :]
        
        # Apply ROI attention to patch features
        if len(outputs.last_hidden_state.shape) > 2:
            patch_features = outputs.last_hidden_state
            attended_features, attention_weights = self.roi_attention(
                patch_features, patch_features, patch_features
            )
            # Use attended [CLS] token
            pooled_features = attended_features[:, 0, :]
        
        # Project to target dimension
        encoded_features = self.projection(pooled_features)
        
        return encoded_features
    
    def detect_anomalies(self, images: torch.Tensor) -> torch.Tensor:
        """Detect medical anomalies in images."""
        features = self.forward(images)
        anomaly_scores = self.anomaly_detector(features)
        return anomaly_scores.squeeze(-1)
    
    def extract_roi_attention(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract region of interest attention maps."""
        enhanced_images = self.medical_conv(images)
        outputs = self.vision_model(pixel_values=enhanced_images)
        
        patch_features = outputs.last_hidden_state
        attended_features, attention_weights = self.roi_attention(
            patch_features, patch_features, patch_features
        )
        
        return attended_features, attention_weights
    
    def preprocess_images(self, images: list) -> torch.Tensor:
        """Preprocess medical images for the model."""
        processed_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            
            # Handle grayscale medical images
            if len(img.shape) == 2:
                img = img.unsqueeze(0).repeat(3, 1, 1)
            elif len(img.shape) == 3 and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            
            processed_img = self.transforms(img)
            processed_images.append(processed_img)
        
        return torch.stack(processed_images)