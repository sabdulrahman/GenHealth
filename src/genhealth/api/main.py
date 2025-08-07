from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager
import torch
from typing import Dict, Any

from .endpoints import router
from .middleware import LoggingMiddleware, ModelLoadingMiddleware
from ..models import MultimodalMedicalModel
from ..data import MedicalReportProcessor, ImageProcessor


# Global model instance
model_instance = None
text_processor = None
image_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load models on startup."""
    global model_instance, text_processor, image_processor
    
    logging.info("Loading multimodal medical model...")
    
    try:
        # Initialize processors
        text_processor = MedicalReportProcessor()
        image_processor = ImageProcessor()
        
        # Initialize model
        model_instance = MultimodalMedicalModel(
            num_classes=20,  # Adjust based on your classification task
            hidden_dim=768,
            fusion_dim=512
        )
        
        # Load pre-trained weights if available
        try:
            checkpoint_path = "data/models/best_model.pt"
            model_instance.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
            logging.info(f"Loaded model weights from {checkpoint_path}")
        except FileNotFoundError:
            logging.warning("No pre-trained model found. Using randomly initialized weights.")
        
        model_instance.eval()
        logging.info("Model loaded successfully!")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise e
    
    yield
    
    # Cleanup
    logging.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="GenHealth: Multimodal Medical Report Analysis",
        description="Advanced AI pipeline for analyzing medical reports with text and image processing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ModelLoadingMiddleware)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model_loaded": model_instance is not None,
            "version": "0.1.0"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "GenHealth: Multimodal Medical Report Analysis API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logging.error(f"Global exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "type": "internal_error"
            }
        )
    
    return app


# Dependency to get model instance
def get_model():
    """Dependency to get model instance."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance


def get_text_processor():
    """Dependency to get text processor."""
    if text_processor is None:
        raise HTTPException(status_code=503, detail="Text processor not loaded")
    return text_processor


def get_image_processor():
    """Dependency to get image processor.""" 
    if image_processor is None:
        raise HTTPException(status_code=503, detail="Image processor not loaded")
    return image_processor


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    app = create_app()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        access_log=True
    )