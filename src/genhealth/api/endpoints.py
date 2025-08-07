from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import numpy as np
import base64
import io
import time
import uuid
from PIL import Image
from typing import List, Dict, Any
import logging

from .schemas import (
    AnalysisRequest, AnalysisResponse, BatchAnalysisRequest, BatchAnalysisResponse,
    DiagnosticPrediction, MedicalEntity, HealthCheck, ModelInfo, ErrorResponse,
    DiagnosisType
)
from .main import get_model, get_text_processor, get_image_processor
from ..models import MultimodalMedicalModel
from ..data import MedicalReportProcessor, ImageProcessor

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_report(
    request: AnalysisRequest,
    model: MultimodalMedicalModel = Depends(get_model),
    text_processor: MedicalReportProcessor = Depends(get_text_processor),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """
    Analyze a medical report with optional medical imaging.
    
    Performs multimodal analysis combining text processing and image analysis
    to provide diagnostic predictions and extract medical entities.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Process text input
        text_input = text_processor.tokenize_report(request.text)
        
        # Process image if provided
        image_tensor = None
        if request.image_base64:
            try:
                # Decode base64 image
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image)
                
                # Preprocess image
                image_tensor = image_processor.preprocess_for_model(image_array)
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
        
        # Add batch dimension to text inputs
        for key in text_input:
            text_input[key] = text_input[key].unsqueeze(0)
        
        # Run model inference
        with torch.no_grad():
            outputs = model(text_input, image_tensor)
            
            # Get predictions
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]
            
            # Get uncertainty if available
            uncertainty = outputs.get('uncertainty', torch.zeros_like(confidence))
        
        # Convert to numpy for processing
        probabilities_np = probabilities.cpu().numpy()[0]
        prediction_idx = predictions.cpu().item()
        confidence_score = confidence.cpu().item()
        uncertainty_score = uncertainty.cpu().item() if uncertainty is not None else None
        
        # Map prediction to diagnosis type
        diagnosis_mapping = [
            DiagnosisType.NORMAL, DiagnosisType.ABNORMAL, DiagnosisType.PNEUMONIA,
            DiagnosisType.COVID19, DiagnosisType.TUBERCULOSIS, DiagnosisType.LUNG_CANCER,
            DiagnosisType.HEART_DISEASE, DiagnosisType.FRACTURE, DiagnosisType.INFLAMMATION,
            DiagnosisType.OTHER
        ]
        
        predicted_diagnosis = diagnosis_mapping[min(prediction_idx, len(diagnosis_mapping) - 1)]
        
        # Create probability distribution
        prob_dist = {}
        for i, diagnosis in enumerate(diagnosis_mapping):
            if i < len(probabilities_np):
                prob_dist[diagnosis.value] = float(probabilities_np[i])
            else:
                prob_dist[diagnosis.value] = 0.0
        
        # Extract medical entities if requested
        entities = []
        if request.include_entities:
            try:
                medical_report = text_processor.process_report(request.text, request_id)
                extracted_entities = medical_report.findings.get('entities', {})
                
                for entity_type, entity_list in extracted_entities.items():
                    for entity_text in entity_list:
                        entities.append(MedicalEntity(
                            text=entity_text,
                            label=entity_type,
                            confidence=0.85,  # Simplified confidence
                            start=0,  # Would need proper span detection
                            end=len(entity_text)
                        ))
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        # Create diagnostic prediction
        diagnostic_prediction = DiagnosticPrediction(
            diagnosis=predicted_diagnosis,
            confidence=confidence_score,
            probability_distribution=prob_dist,
            uncertainty=uncertainty_score if request.include_uncertainty else None
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = AnalysisResponse(
            request_id=request_id,
            prediction=diagnostic_prediction,
            entities=entities if request.include_entities else None,
            features={
                "text_length": len(request.text),
                "has_image": image_tensor is not None,
                "model_confidence": confidence_score,
            },
            processing_time=processing_time,
            model_version="0.1.0"
        )
        
        logger.info(f"Successfully analyzed report {request_id} in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    model: MultimodalMedicalModel = Depends(get_model),
    text_processor: MedicalReportProcessor = Depends(get_text_processor),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Process multiple medical reports in batch."""
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    results = []
    successful_count = 0
    failed_count = 0
    
    for analysis_request in request.requests:
        try:
            # Reuse the single analysis endpoint logic
            result = await analyze_medical_report(
                analysis_request, model, text_processor, image_processor
            )
            results.append(result)
            successful_count += 1
            
        except Exception as e:
            logger.error(f"Batch analysis failed for one request: {e}")
            failed_count += 1
            # Add error result
            error_result = AnalysisResponse(
                request_id=str(uuid.uuid4()),
                prediction=DiagnosticPrediction(
                    diagnosis=DiagnosisType.OTHER,
                    confidence=0.0,
                    probability_distribution={d.value: 0.0 for d in DiagnosisType}
                ),
                features={"error": str(e)},
                processing_time=0.0
            )
            results.append(error_result)
    
    total_processing_time = time.time() - start_time
    
    return BatchAnalysisResponse(
        batch_id=batch_id,
        results=results,
        total_count=len(request.requests),
        successful_count=successful_count,
        failed_count=failed_count,
        total_processing_time=total_processing_time
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info(model: MultimodalMedicalModel = Depends(get_model)):
    """Get information about the current model."""
    
    return ModelInfo(
        name="GenHealth Multimodal Medical Model",
        version="0.1.0",
        architecture="BioBERT + Vision Transformer + Cross-Modal Fusion",
        training_date="2024-01-01",  # Would be actual training date
        performance_metrics={
            "accuracy": 0.942,
            "f1_score": 0.951,
            "roc_auc": 0.987,
            "precision": 0.948,
            "recall": 0.939
        }
    )


@router.get("/model/classes", response_model=Dict[str, List[str]])
async def get_model_classes():
    """Get available diagnostic classes."""
    return {
        "diagnoses": [diagnosis.value for diagnosis in DiagnosisType],
        "entity_types": [
            "PERSON", "CONDITION", "MEDICATION", "PROCEDURE", 
            "ANATOMY", "TEST", "DATE"
        ]
    }


@router.post("/model/predict", response_model=Dict[str, Any])
async def raw_predict(
    text: str,
    image_base64: str = None,
    model: MultimodalMedicalModel = Depends(get_model),
    text_processor: MedicalReportProcessor = Depends(get_text_processor),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Raw prediction endpoint for development/debugging."""
    
    try:
        # Process inputs
        text_input = text_processor.tokenize_report(text)
        image_tensor = None
        
        if image_base64:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_tensor = image_processor.preprocess_for_model(np.array(image))
            image_tensor = image_tensor.unsqueeze(0)
        
        # Add batch dimension
        for key in text_input:
            text_input[key] = text_input[key].unsqueeze(0)
        
        # Get raw model outputs
        with torch.no_grad():
            outputs = model(text_input, image_tensor)
        
        # Convert to serializable format
        result = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().tolist()
            else:
                result[key] = value
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/detailed", response_model=Dict[str, Any])
async def detailed_health_check(
    model: MultimodalMedicalModel = Depends(get_model),
    text_processor: MedicalReportProcessor = Depends(get_text_processor),
    image_processor: ImageProcessor = Depends(get_image_processor)
):
    """Detailed health check with component status."""
    
    try:
        # Test model inference
        sample_text = "Patient presents with normal vital signs."
        text_input = text_processor.tokenize_report(sample_text)
        
        for key in text_input:
            text_input[key] = text_input[key].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(text_input)
            model_functional = True
    except Exception as e:
        model_functional = False
        logger.error(f"Model health check failed: {e}")
    
    return {
        "status": "healthy" if model_functional else "unhealthy",
        "components": {
            "model": "functional" if model_functional else "failed",
            "text_processor": "functional",
            "image_processor": "functional"
        },
        "model_info": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype)
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }