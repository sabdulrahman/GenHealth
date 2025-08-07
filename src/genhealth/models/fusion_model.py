import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class MultimodalFusion(nn.Module):
    """
    Advanced fusion model combining text, vision, and structured data
    using attention mechanisms and cross-modal interactions.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 768,
        structured_dim: int = 64,
        fusion_dim: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_attention: bool = True,
        fusion_strategy: str = "concat_attention"  # "concat", "bilinear", "concat_attention"
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.structured_dim = structured_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.fusion_strategy = fusion_strategy
        
        # Cross-modal attention layers
        if use_attention:
            self.text_vision_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,  # Use fusion_dim instead of text_dim
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.vision_text_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,  # Use fusion_dim instead of vision_dim
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Modality-specific projections
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.structured_projection = nn.Sequential(
            nn.Linear(structured_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layers
        if fusion_strategy == "bilinear":
            self.bilinear_fusion = nn.Bilinear(fusion_dim, fusion_dim, fusion_dim)
        elif fusion_strategy == "concat_attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=fusion_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            fusion_input_dim = fusion_dim  # After attention pooling
        else:  # concat
            fusion_input_dim = fusion_dim * 2  # text + vision (structured is optional)
        
        # Final fusion network - make adaptive to input size
        if fusion_strategy == "bilinear":
            self.fusion_network = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_strategy == "concat_attention":
            self.fusion_network = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2), 
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:  # concat - adaptive size
            # Will be set dynamically based on available modalities
            self.fusion_network = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Auxiliary tasks
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        structured_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multimodal features and generate predictions.
        
        Args:
            text_features: Encoded text features
            vision_features: Encoded vision features (optional)
            structured_data: Additional structured medical data (optional)
            
        Returns:
            Dictionary containing fused features and predictions
        """
        batch_size = text_features.size(0)
        
        # Project modalities to common dimension
        text_proj = self.text_projection(text_features)
        
        # Handle missing modalities
        if vision_features is not None:
            vision_proj = self.vision_projection(vision_features)
        else:
            vision_proj = torch.zeros(batch_size, self.fusion_dim, device=text_features.device)
        
        if structured_data is not None:
            structured_proj = self.structured_projection(structured_data)
        else:
            structured_proj = torch.zeros(batch_size, self.fusion_dim, device=text_features.device)
        
        # Cross-modal attention
        if self.use_attention and vision_features is not None:
            # Text attending to vision
            text_attended, _ = self.text_vision_attention(
                text_proj.unsqueeze(1), 
                vision_proj.unsqueeze(1), 
                vision_proj.unsqueeze(1)
            )
            text_proj = text_attended.squeeze(1)
            
            # Vision attending to text
            vision_attended, _ = self.vision_text_attention(
                vision_proj.unsqueeze(1),
                text_proj.unsqueeze(1),
                text_proj.unsqueeze(1)
            )
            vision_proj = vision_attended.squeeze(1)
        
        # Fusion
        if self.fusion_strategy == "bilinear":
            # Bilinear fusion between text and vision
            fused = self.bilinear_fusion(text_proj, vision_proj)
            # Add structured data if available
            if structured_data is not None:
                fused = fused + structured_proj
        elif self.fusion_strategy == "concat_attention":
            # Concatenate available modalities and apply attention
            modalities_list = [text_proj, vision_proj]
            if structured_data is not None:
                modalities_list.append(structured_proj)
            
            modalities = torch.stack(modalities_list, dim=1)
            attended_fusion, attention_weights = self.fusion_attention(
                modalities, modalities, modalities
            )
            # Mean pool the attended modalities
            fused = attended_fusion.mean(dim=1)
        else:  # concat
            # Concatenate available modalities
            concat_list = [text_proj, vision_proj]
            if structured_data is not None:
                concat_list.append(structured_proj)
            fused = torch.cat(concat_list, dim=-1)
        
        # Final fusion processing
        if self.fusion_strategy == "concat" and self.fusion_network is None:
            # Create fusion network dynamically based on actual input size
            input_size = fused.size(-1)
            self.fusion_network = nn.Sequential(
                nn.Linear(input_size, self.fusion_dim * 2),
                nn.LayerNorm(self.fusion_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.fusion_dim * 2, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            # Move to same device as input
            self.fusion_network = self.fusion_network.to(fused.device)
        
        fusion_features = self.fusion_network(fused)
        
        # Generate predictions
        logits = self.classifier(fusion_features)
        uncertainty = self.uncertainty_head(fusion_features).squeeze(-1)
        
        return {
            'fusion_features': fusion_features,
            'logits': logits,
            'uncertainty': uncertainty,
            'text_projection': text_proj,
            'vision_projection': vision_proj,
            'structured_projection': structured_proj
        }
    
    def compute_modality_importance(
        self,
        text_features: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        structured_data: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute importance weights for each modality."""
        outputs = self.forward(text_features, vision_features, structured_data)
        
        # Compute gradients w.r.t. each modality projection
        logits = outputs['logits']
        predicted_class = torch.argmax(logits, dim=-1)
        
        importance_scores = {}
        
        # Text importance
        if text_features.requires_grad:
            text_grad = torch.autograd.grad(
                logits[range(len(predicted_class)), predicted_class].sum(),
                outputs['text_projection'],
                retain_graph=True
            )[0]
            importance_scores['text'] = text_grad.norm(dim=-1)
        
        # Vision importance
        if vision_features is not None and vision_features.requires_grad:
            vision_grad = torch.autograd.grad(
                logits[range(len(predicted_class)), predicted_class].sum(),
                outputs['vision_projection'],
                retain_graph=True
            )[0]
            importance_scores['vision'] = vision_grad.norm(dim=-1)
        
        # Structured importance
        if structured_data is not None and structured_data.requires_grad:
            struct_grad = torch.autograd.grad(
                logits[range(len(predicted_class)), predicted_class].sum(),
                outputs['structured_projection'],
                retain_graph=True
            )[0]
            importance_scores['structured'] = struct_grad.norm(dim=-1)
        
        return importance_scores