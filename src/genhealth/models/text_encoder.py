import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional


class MedicalTextEncoder(nn.Module):
    """
    Specialized text encoder for medical reports using BioBERT/ClinicalBERT.
    Incorporates domain-specific medical knowledge and terminology.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        hidden_dim: int = 768,
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # Load pre-trained medical language model
        self.transformer = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Medical-specific layers
        self.medical_attention = nn.MultiheadAttention(
            embed_dim=self.transformer.config.hidden_size,
            num_heads=12,
            dropout=dropout,
            batch_first=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Medical entity recognition head
        self.entity_classifier = nn.Linear(hidden_dim, 20)  # 20 medical entity types
        
    def forward(self, text_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode medical text with attention to clinical terminology.
        
        Args:
            text_input: Dictionary with 'input_ids' and 'attention_mask'
            
        Returns:
            Encoded text features
        """
        # Base transformer encoding
        outputs = self.transformer(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask'],
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Medical-specific attention
        attended_output, _ = self.medical_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~text_input['attention_mask'].bool()
        )
        
        # Pool attended features (mean pooling with attention mask)
        mask_expanded = text_input['attention_mask'].unsqueeze(-1).expand(attended_output.size()).float()
        sum_embeddings = torch.sum(attended_output * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Project to target dimension
        encoded_features = self.projection(pooled_output)
        
        return encoded_features
    
    def extract_medical_entities(self, text_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract medical entities from text."""
        features = self.forward(text_input)
        entity_logits = self.entity_classifier(features)
        return torch.sigmoid(entity_logits)  # Multi-label classification
    
    def tokenize_text(self, texts: list[str]) -> Dict[str, torch.Tensor]:
        """Tokenize medical text with proper preprocessing."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded