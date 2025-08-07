"""
GenHealth: Multimodal Medical Report Analysis Pipeline
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import core components with error handling
try:
    from .models import MultimodalMedicalModel
    from .data import MedicalReportProcessor, ImageProcessor
    __all__ = [
        "MultimodalMedicalModel",
        "MedicalReportProcessor", 
        "ImageProcessor",
    ]
except ImportError as e:
    print(f"Warning: Some components could not be imported: {e}")
    __all__ = []

# Optional API import
try:
    from .api import create_app
    __all__.append("create_app")
except ImportError:
    pass