import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import json
import os

from .medical_report_processor import MedicalReportProcessor, MedicalReport
from .image_processor import ImageProcessor


class MedicalDataset(Dataset):
    """
    Dataset for medical reports with optional imaging data.
    """
    
    def __init__(
        self,
        data_path: str,
        text_processor: MedicalReportProcessor,
        image_processor: Optional[ImageProcessor] = None,
        image_dir: Optional[str] = None,
        augment: bool = False
    ):
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.augment = augment
        
        # Load data
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from various formats."""
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
                if 'reports' in data:
                    return data['reports']
                else:
                    return data
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Process text
        text_input = self.text_processor.tokenize_report(item['text'])
        
        # Process image if available
        image_tensor = None
        if self.image_processor and self.image_dir and 'image_path' in item:
            image_path = os.path.join(self.image_dir, item['image_path'])
            if os.path.exists(image_path):
                image_tensor = self.image_processor.preprocess_for_model(
                    image_path, augment=self.augment
                )
        
        # Get label if available
        label = None
        if 'diagnosis' in item or 'label' in item:
            label_text = item.get('diagnosis') or item.get('label')
            # Simple label mapping (expand as needed)
            label_mapping = {
                'normal': 0, 'abnormal': 1, 'pneumonia': 2, 'covid19': 3,
                'tuberculosis': 4, 'lung_cancer': 5, 'heart_disease': 6,
                'fracture': 7, 'inflammation': 8, 'other': 9
            }
            label = label_mapping.get(label_text.lower(), 9)  # Default to 'other'
        
        result = {
            'text_input': text_input,
            'image_input': image_tensor,
            'label': torch.tensor(label) if label is not None else None,
            'metadata': {
                'id': item.get('id', idx),
                'patient_id': item.get('patient_id', 'unknown')
            }
        }
        
        return result


class MultimodalDataLoader:
    """
    Data loader for multimodal medical data with batching support.
    """
    
    def __init__(
        self,
        dataset: MedicalDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function for multimodal data."""
        
        # Separate components
        text_inputs = []
        image_inputs = []
        labels = []
        metadata = []
        
        for item in batch:
            text_inputs.append(item['text_input'])
            if item['image_input'] is not None:
                image_inputs.append(item['image_input'])
            if item['label'] is not None:
                labels.append(item['label'])
            metadata.append(item['metadata'])
        
        # Batch text inputs
        if text_inputs:
            # Stack input_ids and attention_masks
            input_ids = torch.stack([t['input_ids'] for t in text_inputs])
            attention_mask = torch.stack([t['attention_mask'] for t in text_inputs])
            batched_text = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            batched_text = None
        
        # Batch image inputs
        batched_images = torch.stack(image_inputs) if image_inputs else None
        
        # Batch labels
        batched_labels = torch.stack(labels) if labels else None
        
        return {
            'text_input': batched_text,
            'image_input': batched_images,
            'labels': batched_labels,
            'metadata': metadata
        }
    
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=self._collate_fn
        )
    
    def __iter__(self):
        return iter(self.get_dataloader())
    
    def __len__(self):
        return len(self.get_dataloader())