import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from .text_encoder import MedicalTextEncoder
from .vision_encoder import MedicalVisionEncoder
from .fusion_model import MultimodalFusion


class MultimodalMedicalModel(nn.Module):
    """
    Advanced multimodal model for medical report analysis combining
    text, image, and structured data processing.
    """
    
    def __init__(
        self,
        text_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        vision_model_name: str = "google/vit-base-patch16-224",
        hidden_dim: int = 768,
        fusion_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_attention_pooling: bool = True,
    ):
        super().__init__()
        
        self.text_encoder = MedicalTextEncoder(
            model_name=text_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.vision_encoder = MedicalVisionEncoder(
            model_name=vision_model_name,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.fusion_model = MultimodalFusion(
            text_dim=hidden_dim,
            vision_dim=hidden_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_attention=use_attention_pooling
        )
        
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
    def forward(
        self,
        text_input: Dict[str, torch.Tensor],
        image_input: Optional[torch.Tensor] = None,
        structured_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multimodal model.
        
        Args:
            text_input: Tokenized text input with 'input_ids' and 'attention_mask'
            image_input: Processed medical images
            structured_data: Additional structured medical data
            
        Returns:
            Dictionary containing predictions and embeddings
        """
        outputs = {}
        
        # Text encoding
        text_features = self.text_encoder(text_input)
        outputs['text_features'] = text_features
        
        # Vision encoding
        vision_features = None
        if image_input is not None:
            vision_features = self.vision_encoder(image_input)
            outputs['vision_features'] = vision_features
        
        # Multimodal fusion
        fusion_output = self.fusion_model(
            text_features=text_features,
            vision_features=vision_features,
            structured_data=structured_data
        )
        
        outputs.update(fusion_output)
        return outputs
    
    def extract_embeddings(
        self,
        text_input: Dict[str, torch.Tensor],
        image_input: Optional[torch.Tensor] = None,
        structured_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract multimodal embeddings for downstream tasks."""
        with torch.no_grad():
            outputs = self.forward(text_input, image_input, structured_data)
            return outputs['fusion_features']
    
    def predict_diagnosis(
        self,
        text_input: Dict[str, torch.Tensor],
        image_input: Optional[torch.Tensor] = None,
        structured_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate diagnostic predictions with confidence scores."""
        outputs = self.forward(text_input, image_input, structured_data)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0]
        
        return predictions, confidence