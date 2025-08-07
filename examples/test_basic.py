#!/usr/bin/env python3
"""
Basic testing script for GenHealth multimodal medical analysis.
This script demonstrates how to use the core components without running the full API.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from PIL import Image, ImageDraw
import time

from genhealth.models import MultimodalMedicalModel
from genhealth.data import MedicalReportProcessor, ImageProcessor


def create_sample_medical_image(size=(224, 224)):
    """Create a sample medical image for testing."""
    # Create a simple synthetic medical image (chest X-ray like)
    image = Image.new('L', size, color=50)  # Dark background
    draw = ImageDraw.Draw(image)
    
    # Draw lung-like shapes
    draw.ellipse([40, 60, 100, 140], fill=120)  # Left lung
    draw.ellipse([124, 60, 184, 140], fill=120)  # Right lung
    
    # Add some ribs
    for i in range(5):
        y = 70 + i * 15
        draw.arc([20, y-5, 204, y+5], 0, 180, fill=180, width=2)
    
    # Convert to RGB
    return image.convert('RGB')


def test_text_processing():
    """Test medical text processing."""
    print("Testing Medical Text Processing...")
    
    processor = MedicalReportProcessor()
    
    # Sample medical reports with realistic clinical content
    reports = [
        "A 45-year-old female presents to the emergency department with acute onset chest pain that began 3 hours ago while at rest. The pain is described as substernal, crushing in nature, radiating to the left arm and jaw. Associated symptoms include diaphoresis, nausea, and shortness of breath. Vital signs: HR 110 bpm, BP 140/90 mmHg, RR 20, O2 sat 96% on room air, temp 37.1°C. Physical examination reveals an anxious-appearing female in mild distress. Heart sounds are regular with no murmurs. Lungs are clear to auscultation bilaterally. Initial ECG shows sinus tachycardia with no acute ST changes. Chest X-ray demonstrates clear lung fields with normal cardiac silhouette.",
        
        "This 72-year-old male with a past medical history significant for hypertension, hyperlipidemia, and smoking presents with crushing chest pain that started 4 hours ago during physical exertion. Pain is 9/10, substernal, radiating to left shoulder and arm. Patient appears diaphoretic and anxious. Vital signs: HR 95 bpm, BP 180/100 mmHg, RR 22, O2 sat 94% on room air. ECG reveals ST-segment elevation in leads V1-V4 consistent with anterior STEMI. Troponin I level is significantly elevated at 15.2 ng/mL (normal <0.04). Patient was started on aspirin 325mg, clopidogrel 600mg loading dose, and heparin infusion. Emergent cardiac catheterization revealed 100% occlusion of the left anterior descending artery.",
        
        "Patient is a 58-year-old female with type 2 diabetes mellitus presenting for routine follow-up. She reports good adherence to diet and exercise recommendations. Currently taking metformin 1000mg twice daily and glipizide 10mg daily. Home glucose monitoring shows values ranging from 120-160 mg/dL. Recent HbA1c is 7.2%. Physical examination is unremarkable. Feet examination shows no evidence of diabetic foot complications. Dilated eye exam performed 6 months ago showed mild nonproliferative diabetic retinopathy. Laboratory results: creatinine 0.9 mg/dL, estimated GFR >60, microalbumin/creatinine ratio 25 mg/g (mildly elevated)."
    ]
    
    for i, report_text in enumerate(reports, 1):
        print(f"\nProcessing Report {i}:")
        print(f"Text preview: {report_text[:120]}...")
        
        # Process the report
        start_time = time.time()
        medical_report = processor.process_report(report_text, f"test_report_{i}")
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.3f}s")
        print(f"Diagnoses extracted: {len(medical_report.diagnosis_codes)}")
        if medical_report.diagnosis_codes:
            print(f"  Found: {', '.join(medical_report.diagnosis_codes[:3])}")
            
        print(f"Medications identified: {len(medical_report.medications)}")
        if medical_report.medications:
            print(f"  Found: {', '.join(medical_report.medications[:3])}")
            
        print(f"Procedures detected: {len(medical_report.procedures)}")
        if medical_report.procedures:
            print(f"  Found: {', '.join(medical_report.procedures[:3])}")
        
        # Show extracted entities
        entities = medical_report.findings.get('entities', {})
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        if entity_count > 0:
            print(f"Medical entities extracted: {entity_count}")
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  {entity_type}: {', '.join(entity_list[:3])}")
        
        # Show vital signs if found
        vital_signs = medical_report.findings.get('vital_signs', {})
        if vital_signs:
            print(f"Vital signs extracted: {vital_signs}")
    
    print("\nText processing test completed successfully")


def test_image_processing():
    """Test medical image processing."""
    print("\nTesting Medical Image Processing...")
    
    processor = ImageProcessor()
    
    # Create realistic medical image samples
    images = []
    descriptions = []
    
    # Create more realistic chest X-ray simulation
    for i in range(3):
        sample_image = create_sample_medical_image()
        images.append(np.array(sample_image))
        descriptions.append(f"Synthetic chest X-ray {i+1}")
        print(f"Created sample medical image {i+1}: {sample_image.size}")
    
    # Process individual image
    print(f"\nProcessing individual image...")
    start_time = time.time()
    processed_image = processor.preprocess_for_model(images[0])
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.3f}s")
    print(f"Input shape: {images[0].shape}")
    print(f"Output tensor shape: {processed_image.shape}")
    print(f"Normalized range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
    
    # Process batch of images
    print(f"\nBatch processing test...")
    start_time = time.time()
    batch_processed = processor.process_batch(images)
    processing_time = time.time() - start_time
    
    print(f"Batch processing time: {processing_time:.3f}s")
    print(f"Batch tensor shape: {batch_processed.shape}")
    print(f"Memory usage: ~{batch_processed.numel() * 4 / (1024**2):.1f} MB")
    
    # Extract quantitative image features
    print(f"\nExtracting quantitative features...")
    features = processor.extract_image_features(images[0])
    print("Statistical image features:")
    for feature_name, value in features.items():
        if 'intensity' in feature_name:
            print(f"  {feature_name}: {value:.1f}")
        else:
            print(f"  {feature_name}: {value:.4f}")
    
    # Test different image processing steps
    print("\nTesting image enhancement pipeline...")
    enhanced_image = processor.enhance_medical_image(images[0])
    print(f"Enhancement completed - output shape: {enhanced_image.shape}")
    
    # Test ROI extraction
    roi_image, roi_info = processor.extract_roi(images[0])
    print(f"ROI extraction: {roi_info.get('method', 'unknown')} method")
    print(f"ROI dimensions: {roi_image.shape}")
    
    print("\nImage processing test completed successfully")


def test_multimodal_model():
    """Test the complete multimodal model."""
    print("\nTesting Multimodal Medical Model...")
    
    # Initialize components
    print("Initializing model components...")
    model = MultimodalMedicalModel(
        num_classes=10,
        hidden_dim=768,
        fusion_dim=512
    )
    model.eval()
    
    text_processor = MedicalReportProcessor()
    image_processor = ImageProcessor()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Estimated model size: {total_params * 4 / (1024**3):.2f} GB")
    
    # Use realistic medical case data
    sample_cases = [
        {
            "text": "A 45-year-old female presents with acute substernal chest pain radiating to left arm. ECG shows sinus tachycardia. Troponin I elevated at 0.12 ng/mL. Chest X-ray demonstrates clear lung fields with normal cardiac silhouette.",
            "expected": "cardiac event"
        },
        {
            "text": "34-year-old male with 5-day history of productive cough and fever. Chest X-ray reveals right lower lobe consolidation. WBC count 15,200 with left shift. Diagnosis: community-acquired pneumonia.",
            "expected": "pneumonia"
        },
        {
            "text": "58-year-old female with type 2 diabetes for routine follow-up. HbA1c 7.2%, taking metformin and glipizide. Physical exam unremarkable. No diabetic complications noted.",
            "expected": "normal follow-up"
        }
    ]
    
    print(f"\nTesting {len(sample_cases)} clinical scenarios...")
    
    diagnosis_labels = ["Normal", "Abnormal", "Pneumonia", "COVID-19", "Tuberculosis", 
                       "Lung Cancer", "Heart Disease", "Fracture", "Inflammation", "Other"]
    
    for i, case in enumerate(sample_cases, 1):
        print(f"\nCase {i}: {case['expected']}")
        print(f"Text preview: {case['text'][:80]}...")
        
        # Create corresponding synthetic medical image
        sample_image = np.array(create_sample_medical_image())
        
        print(f"Input image shape: {sample_image.shape}")
        
        # Process inputs
        text_input = text_processor.tokenize_report(case['text'])
        image_input = image_processor.preprocess_for_model(sample_image)
        
        # Add batch dimension
        text_input = {k: v.unsqueeze(0) for k, v in text_input.items()}
        image_input = image_input.unsqueeze(0)
        
        print(f"Processed text shape: {text_input['input_ids'].shape}")
        print(f"Processed image shape: {image_input.shape}")
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(text_input, image_input)
        
        inference_time = time.time() - start_time
        
        # Analyze outputs
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0]
        uncertainty = outputs.get('uncertainty', torch.tensor([0.0]))
        
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Predicted class: {predicted_class.item()}")
        print(f"Confidence: {confidence.item():.3f}")
        print(f"Uncertainty: {uncertainty.item():.3f}")
        
        # Show top 3 predictions
        probs_np = probabilities.cpu().numpy()[0]
        top_indices = np.argsort(probs_np)[::-1][:3]
        
        print("Top 3 predictions:")
        for j, idx in enumerate(top_indices, 1):
            label = diagnosis_labels[idx] if idx < len(diagnosis_labels) else f"Class_{idx}"
            print(f"  {j}. {label}: {probs_np[idx]:.3f}")
    
    print("\nMultimodal model test completed successfully")


def test_text_only_mode():
    """Test model with text input only."""
    print("\nTesting Text-Only Mode...")
    
    model = MultimodalMedicalModel(num_classes=10)
    model.eval()
    text_processor = MedicalReportProcessor()
    
    # Test cases for text-only inference
    text_only_cases = [
        "Patient has normal vital signs and unremarkable physical examination.",
        "Laboratory results show elevated troponin levels suggestive of myocardial infarction.",
        "Chest X-ray interpretation pending, clinical suspicion for pneumonia based on symptoms."
    ]
    
    print(f"Testing {len(text_only_cases)} text-only cases...")
    
    for i, sample_text in enumerate(text_only_cases, 1):
        print(f"\nText-only case {i}:")
        print(f"Input: {sample_text[:60]}...")
        
        text_input = text_processor.tokenize_report(sample_text)
        text_input = {k: v.unsqueeze(0) for k, v in text_input.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(text_input, image_input=None)  # No image
        inference_time = time.time() - start_time
        
        probabilities = torch.softmax(outputs['logits'], dim=-1)
        confidence = torch.max(probabilities, dim=-1)[0]
        predicted_class = torch.argmax(probabilities, dim=-1)
        
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Predicted class: {predicted_class.item()}")
        print(f"Confidence: {confidence.item():.3f}")
    
    print("\nText-only mode test completed successfully")


def run_all_tests():
    """Run all tests."""
    print("Starting GenHealth Component Tests")
    print("=" * 50)
    
    test_functions = [
        ("Medical Text Processing", test_text_processing),
        ("Medical Image Processing", test_image_processing),
        ("Multimodal Model Inference", test_multimodal_model),
        ("Text-Only Mode", test_text_only_mode)
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            test_func()
            passed_tests += 1
            print(f"PASSED: {test_name}")
            
        except Exception as e:
            print(f"FAILED: {test_name} - {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("All tests completed successfully!")
        print("GenHealth components are working properly!")
        return True
    else:
        print("Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)