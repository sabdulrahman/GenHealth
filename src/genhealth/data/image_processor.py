import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from typing import List, Dict, Tuple, Optional, Union
import pydicom
from skimage import exposure, morphology, measure
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageProcessor:
    """
    Advanced medical image processor with specialized preprocessing
    for X-rays, MRIs, CT scans, and other medical imaging modalities.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        normalize_method: str = "imagenet",  # "imagenet", "medical", "adaptive"
        enhance_contrast: bool = True,
        denoise: bool = True,
    ):
        self.image_size = image_size
        self.normalize_method = normalize_method
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        
        # Standard ImageNet normalization
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        # Medical image specific normalization (approximate)
        self.medical_mean = [0.449]  # Grayscale
        self.medical_std = [0.226]
        
        # Basic transforms for inference
        self.inference_transforms = self._create_inference_transforms()
        
        # Augmentation transforms for training
        self.train_transforms = self._create_train_transforms()
        
    def _create_inference_transforms(self):
        """Create transforms for model inference."""
        if self.normalize_method == "imagenet":
            normalize = transforms.Normalize(
                mean=self.imagenet_mean,
                std=self.imagenet_std
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize
        ])
    
    def _create_train_transforms(self):
        """Create augmentation transforms for training."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            A.Normalize(
                mean=self.imagenet_mean if self.normalize_method == "imagenet" else [0.5, 0.5, 0.5],
                std=self.imagenet_std if self.normalize_method == "imagenet" else [0.5, 0.5, 0.5]
            ),
            ToTensorV2()
        ])
    
    def load_dicom(self, dicom_path: str) -> np.ndarray:
        """Load and preprocess DICOM medical image."""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)
            
            # Apply DICOM-specific preprocessing
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                image = image * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
            
            # Apply window/level if available
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                window_center = dicom_data.WindowCenter
                window_width = dicom_data.WindowWidth
                
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]
                
                img_min = window_center - window_width // 2
                img_max = window_center + window_width // 2
                image = np.clip(image, img_min, img_max)
            
            return image
        except Exception as e:
            raise ValueError(f"Error loading DICOM file: {e}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from various formats."""
        if image_path.lower().endswith('.dcm'):
            return self.load_dicom(image_path)
        else:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Apply medical-specific image enhancements."""
        if image.dtype != np.uint8:
            # Normalize to 0-255 range
            image_min, image_max = image.min(), image.max()
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
        
        enhanced_image = image.copy()
        
        # Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if self.enhance_contrast:
            if len(enhanced_image.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_image = clahe.apply(enhanced_image)
        
        # Denoising
        if self.denoise:
            if len(enhanced_image.shape) == 3:
                enhanced_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
            else:
                enhanced_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
        
        return enhanced_image
    
    def segment_anatomical_structures(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Basic anatomical structure segmentation (simplified)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        # Extract major structures (simplified approach)
        structures = {}
        for i in range(1, min(num_labels, 6)):  # Limit to 5 major structures
            mask = (labels == i).astype(np.uint8) * 255
            if np.sum(mask) > 1000:  # Filter small components
                structures[f"structure_{i}"] = mask
        
        return structures
    
    def extract_roi(self, image: np.ndarray, roi_method: str = "auto") -> Tuple[np.ndarray, Dict]:
        """Extract region of interest from medical image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        roi_info = {}
        
        if roi_method == "auto":
            # Automatic ROI detection using edge detection and morphology
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                roi = image[y:y+h, x:x+w]
                roi_info = {'x': x, 'y': y, 'width': w, 'height': h, 'method': 'auto'}
            else:
                roi = image
                roi_info = {'method': 'full_image'}
        else:
            roi = image
            roi_info = {'method': 'full_image'}
        
        return roi, roi_info
    
    def preprocess_for_model(
        self,
        image: Union[np.ndarray, str, Image.Image],
        augment: bool = False
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        
        # Load image if path is provided
        if isinstance(image, str):
            image = self.load_image(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array, file path, or PIL Image")
        
        # Enhance medical image
        enhanced_image = self.enhance_medical_image(image)
        
        # Convert to PIL Image for transforms
        if len(enhanced_image.shape) == 2:
            # Convert grayscale to RGB
            enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
        elif enhanced_image.shape[2] == 1:
            enhanced_image = np.repeat(enhanced_image, 3, axis=2)
        
        pil_image = Image.fromarray(enhanced_image.astype(np.uint8))
        
        # Apply transforms
        if augment:
            # Use albumentations for training augmentation
            transformed = self.train_transforms(image=np.array(pil_image))
            return transformed['image']
        else:
            # Use torchvision transforms for inference
            return self.inference_transforms(pil_image)
    
    def process_batch(
        self,
        images: List[Union[np.ndarray, str, Image.Image]],
        augment: bool = False
    ) -> torch.Tensor:
        """Process a batch of images."""
        processed_images = []
        
        for image in images:
            processed_img = self.preprocess_for_model(image, augment=augment)
            processed_images.append(processed_img)
        
        return torch.stack(processed_images)
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from medical image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        features = {}
        
        # Intensity statistics
        features['mean_intensity'] = float(np.mean(gray))
        features['std_intensity'] = float(np.std(gray))
        features['min_intensity'] = float(np.min(gray))
        features['max_intensity'] = float(np.max(gray))
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        
        # Texture features (simplified)
        # Calculate Local Binary Pattern-like features
        features['texture_contrast'] = float(np.std(cv2.Laplacian(gray, cv2.CV_64F)))
        
        # Shape features
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['contour_area'] = float(cv2.contourArea(largest_contour))
            features['contour_perimeter'] = float(cv2.arcLength(largest_contour, True))
            
            if features['contour_perimeter'] > 0:
                features['shape_compactness'] = (4 * np.pi * features['contour_area']) / (features['contour_perimeter'] ** 2)
            else:
                features['shape_compactness'] = 0.0
        else:
            features['contour_area'] = 0.0
            features['contour_perimeter'] = 0.0
            features['shape_compactness'] = 0.0
        
        return features
    
    def create_feature_vector(self, image: np.ndarray) -> torch.Tensor:
        """Create feature vector from image for structured data input."""
        features = self.extract_image_features(image)
        
        # Convert to normalized feature vector
        feature_vector = [
            features['mean_intensity'] / 255.0,
            features['std_intensity'] / 255.0,
            features['intensity_range'] / 255.0,
            features['texture_contrast'] / 10000.0,  # Normalize texture
            min(features['contour_area'] / (self.image_size ** 2), 1.0),  # Normalize area
            min(features['contour_perimeter'] / (self.image_size * 4), 1.0),  # Normalize perimeter
            min(features['shape_compactness'], 1.0),
        ]
        
        return torch.tensor(feature_vector, dtype=torch.float32)