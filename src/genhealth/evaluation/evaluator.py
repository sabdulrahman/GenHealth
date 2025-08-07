import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import classification_report
import time
import logging

from .metrics import MedicalMetrics, DiagnosticMetrics


class ModelEvaluator:
    """
    Comprehensive model evaluator for medical AI systems.
    Handles evaluation across multiple metrics and provides detailed analysis.
    """
    
    def __init__(
        self,
        model,
        class_names: List[str],
        device: str = "cpu"
    ):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.num_classes = len(class_names)
        
        # Initialize metrics
        self.medical_metrics = MedicalMetrics(
            num_classes=self.num_classes,
            class_names=class_names
        )
        self.diagnostic_metrics = DiagnosticMetrics(
            diagnostic_categories=class_names
        )
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(
        self,
        dataloader,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            dataloader: DataLoader for evaluation data
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing all evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_uncertainties = []
        inference_times = []
        
        if verbose:
            print("🔬 Running Model Evaluation...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if verbose and batch_idx % 10 == 0:
                    print(f"   Processing batch {batch_idx + 1}...")
                
                # Extract batch data
                text_input = batch['text_input']
                image_input = batch.get('image_input', None)
                labels = batch.get('labels', None)
                
                # Move to device
                if text_input:
                    text_input = {k: v.to(self.device) for k, v in text_input.items()}
                if image_input is not None:
                    image_input = image_input.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                # Run inference with timing
                start_time = time.time()
                outputs = self.model(text_input, image_input)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Extract predictions
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                uncertainties = outputs.get('uncertainty', torch.zeros_like(predictions, dtype=torch.float))
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
                
                if labels is not None:
                    all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_uncertainties = np.array(all_uncertainties)
        
        if all_labels:
            all_labels = np.array(all_labels)
        else:
            all_labels = None
        
        # Compute metrics
        results = self._compute_comprehensive_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities,
            uncertainties=all_uncertainties
        )
        
        # Add performance metrics
        results['performance'] = {
            'total_samples': len(all_predictions),
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'total_evaluation_time': sum(inference_times),
            'throughput': len(all_predictions) / sum(inference_times)
        }
        
        if verbose:
            print(f"✅ Evaluation completed!")
            print(f"   Samples: {len(all_predictions)}")
            print(f"   Avg inference time: {np.mean(inference_times):.3f}s")
            print(f"   Throughput: {results['performance']['throughput']:.1f} samples/sec")
        
        return results
    
    def _compute_comprehensive_metrics(
        self,
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        
        results = {}
        
        if y_true is not None:
            # Classification metrics
            results['classification'] = self.medical_metrics.compute_classification_metrics(
                y_true, y_pred, y_prob
            )
            
            # Per-class metrics
            results['per_class'] = self.medical_metrics.compute_per_class_metrics(
                y_true, y_pred, y_prob
            )
            
            # Diagnostic metrics
            confidence_scores = np.max(y_prob, axis=1)
            results['diagnostic'] = self.diagnostic_metrics.compute_diagnostic_accuracy(
                y_true, y_pred, confidence_scores
            )
            
            # Uncertainty analysis
            results['uncertainty'] = self.diagnostic_metrics.analyze_prediction_uncertainty(
                y_prob, y_pred, y_true
            )
            
            # Clinical metrics
            results['clinical'] = self.diagnostic_metrics.compute_clinical_metrics(
                y_true, y_pred
            )
        
        # Prediction distribution
        results['predictions'] = {
            'class_distribution': {
                self.class_names[i]: int(np.sum(y_pred == i))
                for i in range(self.num_classes)
            },
            'confidence_stats': {
                'mean': float(np.mean(np.max(y_prob, axis=1))),
                'std': float(np.std(np.max(y_prob, axis=1))),
                'min': float(np.min(np.max(y_prob, axis=1))),
                'max': float(np.max(np.max(y_prob, axis=1)))
            },
            'uncertainty_stats': {
                'mean': float(np.mean(uncertainties)),
                'std': float(np.std(uncertainties)),
                'min': float(np.min(uncertainties)),
                'max': float(np.max(uncertainties))
            }
        }
        
        return results
    
    def evaluate_single_sample(
        self,
        text_input: Dict[str, torch.Tensor],
        image_input: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single sample with detailed analysis.
        
        Args:
            text_input: Tokenized text input
            image_input: Optional image tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Detailed prediction results
        """
        self.model.eval()
        
        with torch.no_grad():
            # Add batch dimension if needed
            if text_input['input_ids'].dim() == 1:
                text_input = {k: v.unsqueeze(0) for k, v in text_input.items()}
            if image_input is not None and image_input.dim() == 3:
                image_input = image_input.unsqueeze(0)
            
            # Run inference
            start_time = time.time()
            outputs = self.model(text_input, image_input)
            inference_time = time.time() - start_time
            
            # Extract results
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = torch.max(probabilities, dim=-1)[0].item()
            uncertainty = outputs.get('uncertainty', torch.tensor([0.0])).item()
            
            # Get top predictions
            probs_np = probabilities.cpu().numpy()[0]
            top_indices = np.argsort(probs_np)[::-1][:5]
            
            top_predictions = [
                {
                    'class': self.class_names[idx],
                    'probability': float(probs_np[idx]),
                    'class_id': int(idx)
                }
                for idx in top_indices
            ]
        
        result = {
            'predicted_class': self.class_names[predicted_class],
            'predicted_class_id': predicted_class,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'inference_time': inference_time,
            'top_predictions': top_predictions,
            'probability_distribution': {
                self.class_names[i]: float(probs_np[i])
                for i in range(self.num_classes)
            }
        }
        
        if return_attention and 'attention_weights' in outputs:
            result['attention_weights'] = outputs['attention_weights']
        
        return result
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report."""
        
        report_lines = [
            "# GenHealth Model Evaluation Report",
            "=" * 50,
            ""
        ]
        
        # Performance summary
        if 'performance' in evaluation_results:
            perf = evaluation_results['performance']
            report_lines.extend([
                "## Performance Summary",
                f"Total Samples: {perf['total_samples']:,}",
                f"Average Inference Time: {perf['avg_inference_time']:.3f}s",
                f"Throughput: {perf['throughput']:.1f} samples/second",
                ""
            ])
        
        # Classification metrics
        if 'classification' in evaluation_results:
            clf = evaluation_results['classification']
            report_lines.extend([
                "## Classification Metrics",
                f"Accuracy: {clf['accuracy']:.3f}",
                f"Precision: {clf['precision']:.3f}",
                f"Recall: {clf['recall']:.3f}",
                f"F1-Score: {clf['f1']:.3f}",
                f"ROC-AUC: {clf.get('roc_auc', 0):.3f}",
                ""
            ])
        
        # Diagnostic metrics
        if 'diagnostic' in evaluation_results:
            diag = evaluation_results['diagnostic']
            report_lines.extend([
                "## Diagnostic Performance",
                f"Diagnostic Accuracy: {diag['diagnostic_accuracy']:.3f}",
                f"High Confidence Accuracy: {diag.get('high_confidence_accuracy', 0):.3f}",
                f"Average Confidence: {diag.get('average_confidence', 0):.3f}",
                ""
            ])
        
        # Per-class performance
        if 'per_class' in evaluation_results:
            report_lines.extend([
                "## Per-Class Performance",
                f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}",
                "-" * 50
            ])
            
            for class_name, metrics in evaluation_results['per_class'].items():
                report_lines.append(
                    f"{class_name:<20} {metrics['precision']:<10.3f} "
                    f"{metrics['recall']:<10.3f} {metrics['f1']:<10.3f}"
                )
            
            report_lines.append("")
        
        return "\\n".join(report_lines)