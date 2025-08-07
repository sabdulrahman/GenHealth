#!/usr/bin/env python3
"""
API testing script for GenHealth.
Tests the FastAPI endpoints with sample data.
"""

import requests
import base64
import json
import time
from PIL import Image, ImageDraw
import io
import sys
import os


def create_sample_medical_image():
    """Create a sample medical image and encode as base64."""
    # Create a simple synthetic X-ray-like image
    image = Image.new('L', (224, 224), color=50)
    draw = ImageDraw.Draw(image)
    
    # Draw lung-like shapes
    draw.ellipse([40, 60, 100, 140], fill=120)
    draw.ellipse([124, 60, 184, 140], fill=120)
    
    # Add some ribs
    for i in range(5):
        y = 70 + i * 15
        draw.arc([20, y-5, 204, y+5], 0, 180, fill=180, width=2)
    
    # Convert to RGB and encode as base64
    image_rgb = image.convert('RGB')
    
    # Save to bytes
    img_buffer = io.BytesIO()
    image_rgb.save(img_buffer, format='JPEG')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    
    return img_str


def test_health_check(base_url="http://localhost:8000"):
    """Test health check endpoints."""
    print("🏥 Testing Health Check...")
    
    try:
        # Basic health check
        response = requests.get(f"{base_url}/health")
        print(f"📊 Health Status: {response.status_code}")
        print(f"📋 Response: {response.json()}")
        
        # Detailed health check
        response = requests.get(f"{base_url}/api/v1/health/detailed")
        if response.status_code == 200:
            data = response.json()
            print(f"🔧 Detailed Health: {data.get('status')}")
            print(f"⚙️  Components: {data.get('components', {})}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the API server is running!")
        print("💡 Start the server with: python -m genhealth.api.main")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_model_info(base_url="http://localhost:8000"):
    """Test model information endpoint."""
    print("\n🤖 Testing Model Info...")
    
    try:
        response = requests.get(f"{base_url}/api/v1/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"📝 Model Name: {data.get('name')}")
            print(f"🔢 Version: {data.get('version')}")
            print(f"🏗️  Architecture: {data.get('architecture')}")
            print(f"📊 Performance: {data.get('performance_metrics', {})}")
            return True
        else:
            print(f"❌ Failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False


def test_single_analysis(base_url="http://localhost:8000"):
    """Test single medical report analysis."""
    print("\n📋 Testing Single Report Analysis...")
    
    # Sample medical reports
    sample_reports = [
        {
            "text": "Patient presents with acute chest pain and shortness of breath. Heart rate elevated at 110 bpm. Chest X-ray shows bilateral infiltrates consistent with pneumonia.",
            "patient_id": "PAT001",
            "include_entities": True,
            "include_uncertainty": True
        },
        {
            "text": "72-year-old male with crushing chest pain radiating to left arm. ECG shows ST elevation in leads V1-V4. Troponin levels significantly elevated at 15.2 ng/mL.",
            "patient_id": "PAT002",
            "include_entities": True
        },
        {
            "text": "Follow-up visit shows normal vital signs. Patient reports feeling well with no complaints. Physical examination unremarkable.",
            "patient_id": "PAT003"
        }
    ]
    
    for i, report in enumerate(sample_reports, 1):
        print(f"\n📄 Testing Report {i}...")
        print(f"📝 Text: {report['text'][:80]}...")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/v1/analyze",
                json=report,
                timeout=30
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', {})
                
                print(f"✅ Analysis successful ({request_time:.2f}s)")
                print(f"🎯 Diagnosis: {prediction.get('diagnosis')}")
                print(f"💯 Confidence: {prediction.get('confidence', 0):.3f}")
                print(f"🔀 Uncertainty: {prediction.get('uncertainty', 'N/A')}")
                print(f"🏷️  Entities found: {len(data.get('entities', []))}")
                
                # Show top predictions
                prob_dist = prediction.get('probability_distribution', {})
                if prob_dist:
                    top_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    print("🏆 Top predictions:")
                    for diagnosis, prob in top_probs:
                        print(f"   • {diagnosis}: {prob:.3f}")
                
            else:
                print(f"❌ Failed with status: {response.status_code}")
                print(f"📄 Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    return True


def test_multimodal_analysis(base_url="http://localhost:8000"):
    """Test multimodal analysis with text and image."""
    print("\n🖼️📝 Testing Multimodal Analysis...")
    
    # Create sample image
    print("🎨 Creating sample medical image...")
    sample_image_b64 = create_sample_medical_image()
    
    multimodal_request = {
        "text": "Chest X-ray shows bilateral lower lobe consolidation consistent with pneumonia. Patient presents with fever, cough, and difficulty breathing.",
        "image_base64": sample_image_b64,
        "patient_id": "PAT004",
        "include_entities": True,
        "include_uncertainty": True
    }
    
    try:
        print("🚀 Sending multimodal analysis request...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/v1/analyze",
            json=multimodal_request,
            timeout=60  # Longer timeout for multimodal
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', {})
            features = data.get('features', {})
            
            print(f"✅ Multimodal analysis successful ({request_time:.2f}s)")
            print(f"🎯 Diagnosis: {prediction.get('diagnosis')}")
            print(f"💯 Confidence: {prediction.get('confidence', 0):.3f}")
            print(f"🖼️  Has image: {features.get('has_image', False)}")
            print(f"📏 Text length: {features.get('text_length', 0)}")
            print(f"🏷️  Entities: {len(data.get('entities', []))}")
            
            return True
            
        else:
            print(f"❌ Multimodal analysis failed: {response.status_code}")
            print(f"📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Multimodal analysis failed: {e}")
        return False


def test_batch_analysis(base_url="http://localhost:8000"):
    """Test batch analysis."""
    print("\n📦 Testing Batch Analysis...")
    
    batch_request = {
        "requests": [
            {
                "text": "Patient has normal blood pressure and heart rate.",
                "patient_id": "BATCH001"
            },
            {
                "text": "Severe chest pain with elevated cardiac enzymes.",
                "patient_id": "BATCH002"
            },
            {
                "text": "Routine checkup shows all vitals within normal limits.",
                "patient_id": "BATCH003"
            }
        ],
        "priority": "normal"
    }
    
    try:
        print(f"📊 Sending batch of {len(batch_request['requests'])} reports...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/api/v1/analyze/batch",
            json=batch_request,
            timeout=60
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Batch analysis completed ({request_time:.2f}s)")
            print(f"📊 Total requests: {data.get('total_count', 0)}")
            print(f"✅ Successful: {data.get('successful_count', 0)}")
            print(f"❌ Failed: {data.get('failed_count', 0)}")
            print(f"⏱️  Total processing time: {data.get('total_processing_time', 0):.2f}s")
            
            # Show first result
            results = data.get('results', [])
            if results:
                first_result = results[0]
                prediction = first_result.get('prediction', {})
                print(f"🏆 First result diagnosis: {prediction.get('diagnosis')}")
            
            return True
            
        else:
            print(f"❌ Batch analysis failed: {response.status_code}")
            print(f"📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        return False


def test_error_handling(base_url="http://localhost:8000"):
    """Test error handling."""
    print("\n🚨 Testing Error Handling...")
    
    # Test with invalid data
    invalid_requests = [
        {
            "name": "Empty text",
            "data": {"text": ""},  # Too short
            "expected_status": 422
        },
        {
            "name": "Invalid image",
            "data": {
                "text": "Valid text here",
                "image_base64": "invalid_base64_data"
            },
            "expected_status": 400
        }
    ]
    
    for test_case in invalid_requests:
        print(f"\n🧪 Testing {test_case['name']}...")
        
        try:
            response = requests.post(
                f"{base_url}/api/v1/analyze",
                json=test_case['data'],
                timeout=10
            )
            
            if response.status_code == test_case['expected_status']:
                print(f"✅ Correctly returned status {response.status_code}")
            else:
                print(f"❌ Expected {test_case['expected_status']}, got {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
    
    return True


def run_api_tests(base_url="http://localhost:8000"):
    """Run all API tests."""
    print("🚀 Starting GenHealth API Tests")
    print("=" * 50)
    
    # Check if server is running
    if not test_health_check(base_url):
        print("\n💡 To start the API server, run:")
        print("   python -m genhealth.api.main")
        return False
    
    tests = [
        ("Model Info", lambda: test_model_info(base_url)),
        ("Single Analysis", lambda: test_single_analysis(base_url)),
        ("Multimodal Analysis", lambda: test_multimodal_analysis(base_url)),
        ("Batch Analysis", lambda: test_batch_analysis(base_url)),
        ("Error Handling", lambda: test_error_handling(base_url))
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
    
    print("\n" + "=" * 50)
    print(f"🎯 API Test Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print("🎉 All API tests completed successfully!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    # Allow custom base URL
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    success = run_api_tests(base_url)
    sys.exit(0 if success else 1)