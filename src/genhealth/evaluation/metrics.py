import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, multilabel_confusion_matrix
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns


class MedicalMetrics:
    """
    Comprehensive metrics for medical AI evaluation including
    clinical-specific metrics and interpretability measures.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def compute_classification_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """Compute comprehensive classification metrics."""
        
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Sensitivity and Specificity (for each class)
        if self.num_classes == 2:  # Binary classification
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # ROC-AUC and PR-AUC if probabilities are provided
        if y_prob is not None:
            try:
                if self.num_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob, average=average)
            except ValueError:
                # Handle cases where AUC cannot be computed
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each class individually."""
        
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_prob is not None and isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        
        per_class_metrics = {}
        
        # Compute metrics for each class
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                'f1': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
            }
            
            # Add AUC if probabilities are provided
            if y_prob is not None and i < y_prob.shape[1]:
                binary_true = (y_true == i).astype(int)
                if len(np.unique(binary_true)) > 1:  # Check if both classes are present
                    try:
                        per_class_metrics[class_name]['roc_auc'] = roc_auc_score(binary_true, y_prob[:, i])
                        per_class_metrics[class_name]['pr_auc'] = average_precision_score(binary_true, y_prob[:, i])
                    except ValueError:
                        per_class_metrics[class_name]['roc_auc'] = 0.0
                        per_class_metrics[class_name]['pr_auc'] = 0.0
                else:
                    per_class_metrics[class_name]['roc_auc'] = 0.0
                    per_class_metrics[class_name]['pr_auc'] = 0.0
        
        return per_class_metrics


class DiagnosticMetrics:
    """
    Specialized metrics for diagnostic accuracy evaluation,
    focusing on clinical decision-making support.
    """
    
    def __init__(self, diagnostic_categories: List[str]):
        self.diagnostic_categories = diagnostic_categories
        
    def compute_diagnostic_accuracy(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        confidence_scores: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Compute diagnostic accuracy metrics."""
        
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if confidence_scores is not None and isinstance(confidence_scores, torch.Tensor):
            confidence_scores = confidence_scores.cpu().numpy()
        
        metrics = {}
        
        # Overall diagnostic accuracy
        metrics['diagnostic_accuracy'] = accuracy_score(y_true, y_pred)
        
        # High-confidence predictions accuracy
        if confidence_scores is not None:
            high_conf_mask = confidence_scores > 0.8
            if np.sum(high_conf_mask) > 0:
                metrics['high_confidence_accuracy'] = accuracy_score(
                    y_true[high_conf_mask], 
                    y_pred[high_conf_mask]
                )
                metrics['high_confidence_percentage'] = np.mean(high_conf_mask) * 100
            else:
                metrics['high_confidence_accuracy'] = 0.0
                metrics['high_confidence_percentage'] = 0.0
            
            # Low-confidence predictions (requiring human review)
            low_conf_mask = confidence_scores < 0.6
            metrics['low_confidence_percentage'] = np.mean(low_conf_mask) * 100
            
            # Average confidence
            metrics['average_confidence'] = np.mean(confidence_scores)
        
        return metrics
    
    def compute_clinical_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        severity_weights: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """Compute clinical decision-making metrics."""
        
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        metrics = {}
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # False positive and false negative rates
        if cm.shape[0] == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Weighted error (if severity weights provided)
        if severity_weights:
            weighted_errors = 0
            total_samples = 0
            
            for true_class in range(len(severity_weights)):
                for pred_class in range(len(severity_weights)):
                    if true_class != pred_class:
                        error_weight = severity_weights.get(true_class, 1.0)
                        error_count = cm[true_class, pred_class] if true_class < cm.shape[0] and pred_class < cm.shape[1] else 0
                        weighted_errors += error_count * error_weight
                    total_samples += cm[true_class, pred_class] if true_class < cm.shape[0] and pred_class < cm.shape[1] else 0
            
            metrics['weighted_error_rate'] = weighted_errors / total_samples if total_samples > 0 else 0.0
        
        return metrics
    
    def analyze_prediction_uncertainty(
        self,
        probabilities: Union[np.ndarray, torch.Tensor],
        predictions: Union[np.ndarray, torch.Tensor],
        true_labels: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """Analyze prediction uncertainty for clinical decision support."""
        
        # Convert to numpy if needed
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.cpu().numpy()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()
        
        # Calculate entropy as a measure of uncertainty
        epsilon = 1e-8
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)
        
        # Normalize entropy
        max_entropy = np.log(probabilities.shape[1])
        normalized_entropy = entropy / max_entropy
        
        # Confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        # Prediction correctness
        correct_predictions = (predictions == true_labels).astype(int)
        
        metrics = {
            'mean_entropy': float(np.mean(entropy)),
            'mean_normalized_entropy': float(np.mean(normalized_entropy)),
            'mean_confidence': float(np.mean(confidence)),
            'std_confidence': float(np.std(confidence)),
        }
        
        # Calibration analysis (simplified)
        # Check if high confidence correlates with correct predictions
        high_conf_mask = confidence > 0.8
        low_conf_mask = confidence < 0.5
        
        if np.sum(high_conf_mask) > 0:
            metrics['high_conf_accuracy'] = float(np.mean(correct_predictions[high_conf_mask]))
        else:
            metrics['high_conf_accuracy'] = 0.0
            
        if np.sum(low_conf_mask) > 0:
            metrics['low_conf_accuracy'] = float(np.mean(correct_predictions[low_conf_mask]))
        else:
            metrics['low_conf_accuracy'] = 0.0
        
        # Expected Calibration Error (simplified)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(correct_predictions[in_bin])
                avg_confidence_in_bin = np.mean(confidence[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        metrics['expected_calibration_error'] = float(ece)
        
        return metrics