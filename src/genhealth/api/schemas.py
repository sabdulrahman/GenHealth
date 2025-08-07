from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class DiagnosisType(str, Enum):
    """Diagnostic categories."""
    NORMAL = "normal"
    ABNORMAL = "abnormal"
    PNEUMONIA = "pneumonia"
    COVID19 = "covid19"
    TUBERCULOSIS = "tuberculosis"
    LUNG_CANCER = "lung_cancer"
    HEART_DISEASE = "heart_disease"
    FRACTURE = "fracture"
    INFLAMMATION = "inflammation"
    OTHER = "other"


class AnalysisRequest(BaseModel):
    """Request model for medical report analysis."""
    text: str = Field(..., description="Medical report text", min_length=10)
    image_base64: Optional[str] = Field(None, description="Base64 encoded medical image")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    report_type: Optional[str] = Field("general", description="Type of medical report")
    include_entities: bool = Field(True, description="Extract medical entities")
    include_uncertainty: bool = Field(True, description="Include uncertainty scores")


class MedicalEntity(BaseModel):
    """Medical entity extracted from text."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity type/label")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    start: int = Field(..., description="Start position in text")
    end: int = Field(..., description="End position in text")


class DiagnosticPrediction(BaseModel):
    """Diagnostic prediction result."""
    diagnosis: DiagnosisType = Field(..., description="Predicted diagnosis")
    confidence: float = Field(..., description="Prediction confidence", ge=0.0, le=1.0)
    probability_distribution: Dict[str, float] = Field(..., description="Probability for each class")
    uncertainty: Optional[float] = Field(None, description="Prediction uncertainty", ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Response model for medical report analysis."""
    request_id: str = Field(..., description="Unique request identifier")
    prediction: DiagnosticPrediction = Field(..., description="Main diagnostic prediction")
    entities: Optional[List[MedicalEntity]] = Field(None, description="Extracted medical entities")
    features: Dict[str, Any] = Field({}, description="Extracted clinical features")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field("0.1.0", description="Model version used")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class BatchAnalysisRequest(BaseModel):
    """Batch analysis request."""
    requests: List[AnalysisRequest] = Field(..., description="List of analysis requests", max_items=50)
    priority: str = Field("normal", description="Processing priority")


class BatchAnalysisResponse(BaseModel):
    """Batch analysis response."""
    batch_id: str = Field(..., description="Batch identifier")
    results: List[AnalysisResponse] = Field(..., description="Analysis results")
    total_count: int = Field(..., description="Total number of requests")
    successful_count: int = Field(..., description="Number of successful analyses")
    failed_count: int = Field(..., description="Number of failed analyses")
    total_processing_time: float = Field(..., description="Total processing time")


class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    architecture: str = Field(..., description="Model architecture")
    training_date: Optional[str] = Field(None, description="Training date")
    performance_metrics: Dict[str, float] = Field({}, description="Performance metrics")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier")