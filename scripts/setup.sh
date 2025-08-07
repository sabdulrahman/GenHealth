#!/bin/bash

# GenHealth Setup Script
# This script sets up the development environment and downloads necessary models

set -e  # Exit on any error

echo "🚀 Setting up GenHealth: Multimodal Medical Report Analysis"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_success "Python $python_version is compatible"
else
    print_error "Python 3.9+ is required, but you have $python_version"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate || {
    print_error "Failed to activate virtual environment"
    exit 1
}

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install development dependencies
print_status "Installing development dependencies..."
pip install -e ".[dev,viz,tracking]"

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p notebooks
mkdir -p tests
mkdir -p configs

# Download spaCy models (optional)
print_status "Downloading spaCy models..."
python -m spacy download en_core_web_sm || print_warning "Failed to download spaCy model"

# Try to download medical spaCy model
pip install scispacy || print_warning "Failed to install scispacy"
python -m spacy download en_core_sci_sm || print_warning "Failed to download medical spaCy model"

# Create sample data
print_status "Creating sample data..."
cat > data/sample_reports.json << 'EOF'
{
  "reports": [
    {
      "id": "R001",
      "text": "Patient presents with acute chest pain and shortness of breath. Heart rate elevated at 110 bpm. Blood pressure 140/90 mmHg. Chest X-ray shows bilateral infiltrates consistent with pneumonia.",
      "diagnosis": "pneumonia",
      "patient_id": "P001"
    },
    {
      "id": "R002", 
      "text": "72-year-old male with crushing chest pain radiating to left arm. ECG shows ST elevation in leads V1-V4. Troponin levels significantly elevated at 15.2 ng/mL.",
      "diagnosis": "myocardial_infarction",
      "patient_id": "P002"
    },
    {
      "id": "R003",
      "text": "Routine follow-up visit. Patient reports feeling well with no complaints. Vital signs within normal limits. Physical examination unremarkable.",
      "diagnosis": "normal",
      "patient_id": "P003"
    }
  ]
}
EOF

# Create configuration files
print_status "Creating configuration files..."
cat > configs/model_config.json << 'EOF'
{
  "model": {
    "text_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "vision_model": "google/vit-base-patch16-224",
    "hidden_dim": 768,
    "fusion_dim": 512,
    "num_classes": 10,
    "dropout": 0.1,
    "fusion_strategy": "concat_attention"
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "weight_decay": 0.01
  },
  "data": {
    "max_text_length": 512,
    "image_size": 224,
    "augmentation": true
  }
}
EOF

# Set up pre-commit hooks
print_status "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not available, skipping hooks setup"
fi

# Create .env file
print_status "Creating environment configuration..."
cat > .env << 'EOF'
# GenHealth Environment Configuration
PYTHONPATH=./src
MODEL_CACHE_DIR=./data/models
DATA_DIR=./data
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Add your API keys here
# WANDB_API_KEY=your_wandb_key
# MLFLOW_TRACKING_URI=your_mlflow_uri
EOF

# Make scripts executable
print_status "Making scripts executable..."
chmod +x scripts/*.sh

# Run basic tests
print_status "Running basic component tests..."
python examples/test_basic.py || print_warning "Basic tests had some issues, but setup continues"

print_success "Setup completed successfully!"
echo ""
echo "🎉 GenHealth is ready to use!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Start the API server: python -m genhealth.api.main"
echo "3. Test the API: python examples/test_api.py"
echo "4. View API docs: http://localhost:8000/docs"
echo ""
echo "For more information, check the README.md file."