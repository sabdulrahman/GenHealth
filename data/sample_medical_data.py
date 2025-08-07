"""
Sample medical data for testing GenHealth system.
Contains realistic (but synthetic) medical reports and image descriptions.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os


# Realistic medical case reports
MEDICAL_REPORTS = [
    {
        "case_id": "CASE_001",
        "patient_age": 45,
        "gender": "female",
        "chief_complaint": "chest pain",
        "report": """
CLINICAL PRESENTATION:
A 45-year-old female presents to the emergency department with acute onset of substernal chest pain that began 3 hours ago while at rest. The pain is described as crushing in nature with radiation to the left arm and jaw. Associated symptoms include diaphoresis, nausea, and mild shortness of breath.

PHYSICAL EXAMINATION:
Vital Signs: Heart rate 110 bpm, blood pressure 140/90 mmHg, respiratory rate 20/min, oxygen saturation 96% on room air, temperature 37.1°C.
General: Anxious-appearing female in mild distress.
Cardiovascular: Regular heart sounds, no murmurs, rubs, or gallops.
Pulmonary: Clear to auscultation bilaterally.
Extremities: No edema, pulses intact.

DIAGNOSTIC STUDIES:
ECG: Sinus tachycardia with no acute ST-T wave changes.
Chest X-ray: Clear lung fields, normal cardiac silhouette.
Troponin I: 0.12 ng/mL (elevated, normal <0.04).
CBC: White blood cell count 8,500/µL, hemoglobin 12.5 g/dL, platelets 275,000/µL.

ASSESSMENT AND PLAN:
Non-ST elevation myocardial infarction (NSTEMI). Patient started on dual antiplatelet therapy with aspirin 325mg and clopidogrel 75mg, atorvastatin 80mg, and metoprolol 25mg twice daily. Cardiology consultation obtained for risk stratification and potential cardiac catheterization.
        """,
        "diagnosis": "NSTEMI",
        "medications": ["aspirin", "clopidogrel", "atorvastatin", "metoprolol"],
        "procedures": ["ECG", "chest X-ray", "troponin", "CBC"]
    },
    
    {
        "case_id": "CASE_002", 
        "patient_age": 72,
        "gender": "male",
        "chief_complaint": "crushing chest pain",
        "report": """
CLINICAL PRESENTATION:
This 72-year-old male with past medical history significant for hypertension, hyperlipidemia, and 30-pack-year smoking history presents with crushing substernal chest pain that began 4 hours ago during snow shoveling. Pain is rated 9/10 in severity with radiation to the left shoulder and arm.

PHYSICAL EXAMINATION:
Vital Signs: Heart rate 95 bpm, blood pressure 180/100 mmHg, respiratory rate 22/min, oxygen saturation 94% on room air.
General: Diaphoretic, anxious male in significant distress.
Cardiovascular: Regular rate and rhythm, S4 gallop present.
Pulmonary: Bilateral fine crackles at lung bases.

DIAGNOSTIC STUDIES:
ECG: ST-segment elevation in leads V1-V4 consistent with anterior STEMI.
Chest X-ray: Mild pulmonary vascular congestion.
Troponin I: 15.2 ng/mL (significantly elevated).
BNP: 450 pg/mL (elevated).

TREATMENT:
Patient received aspirin 325mg chewed, clopidogrel 600mg loading dose, atorvastatin 80mg, and heparin infusion. Emergent cardiac catheterization revealed 100% occlusion of the left anterior descending artery. Successful primary percutaneous coronary intervention with drug-eluting stent placement performed.
        """,
        "diagnosis": "STEMI",
        "medications": ["aspirin", "clopidogrel", "atorvastatin", "heparin"],
        "procedures": ["ECG", "chest X-ray", "cardiac catheterization", "PCI"]
    },
    
    {
        "case_id": "CASE_003",
        "patient_age": 58,
        "gender": "female", 
        "chief_complaint": "diabetes follow-up",
        "report": """
CLINICAL PRESENTATION:
Patient is a 58-year-old female with type 2 diabetes mellitus for 8 years presenting for routine follow-up appointment. She reports good adherence to dietary recommendations and regular exercise routine including 30 minutes of walking 5 days per week.

CURRENT MEDICATIONS:
- Metformin 1000mg twice daily
- Glipizide 10mg once daily
- Lisinopril 10mg once daily
- Atorvastatin 20mg once daily

PHYSICAL EXAMINATION:
Vital Signs: Heart rate 72 bpm, blood pressure 128/78 mmHg, BMI 28.5 kg/m².
General: Well-appearing female in no distress.
Cardiovascular: Regular rate and rhythm, no murmurs.
Feet: No evidence of diabetic foot complications, pulses intact, sensation normal.

LABORATORY RESULTS:
HbA1c: 7.2% (target <7%)
Fasting glucose: 145 mg/dL
Serum creatinine: 0.9 mg/dL (eGFR >60)
Microalbumin/creatinine ratio: 25 mg/g (mildly elevated)
LDL cholesterol: 95 mg/dL

ASSESSMENT:
Type 2 diabetes with good glycemic control. Mild diabetic nephropathy. Continue current regimen with close monitoring.
        """,
        "diagnosis": "type 2 diabetes",
        "medications": ["metformin", "glipizide", "lisinopril", "atorvastatin"],
        "procedures": ["HbA1c", "glucose", "creatinine", "microalbumin"]
    },
    
    {
        "case_id": "CASE_004",
        "patient_age": 34,
        "gender": "male",
        "chief_complaint": "shortness of breath",
        "report": """
CLINICAL PRESENTATION:
A 34-year-old previously healthy male presents with 5-day history of progressively worsening shortness of breath, productive cough with yellow-green sputum, and fever up to 39.2°C. No recent travel or sick contacts.

PHYSICAL EXAMINATION:
Vital Signs: Heart rate 105 bpm, blood pressure 120/75 mmHg, respiratory rate 24/min, oxygen saturation 92% on room air, temperature 38.8°C.
General: Ill-appearing male with labored breathing.
Pulmonary: Decreased breath sounds and dullness to percussion over right lower lobe, inspiratory crackles present.

DIAGNOSTIC STUDIES:
Chest X-ray: Right lower lobe consolidation consistent with pneumonia.
CBC: White blood cell count 15,200/µL with left shift.
Blood cultures: Pending.
Sputum culture: Pending.
Procalcitonin: 2.8 ng/mL (elevated).

TREATMENT:
Community-acquired pneumonia. Started on ceftriaxone 1g IV daily and azithromycin 500mg IV daily. Supportive care with oxygen therapy to maintain saturation >92%.
        """,
        "diagnosis": "pneumonia",
        "medications": ["ceftriaxone", "azithromycin"],
        "procedures": ["chest X-ray", "CBC", "blood cultures", "sputum culture"]
    }
]


def create_realistic_chest_xray(pathology="normal", size=(224, 224)):
    """
    Create more realistic chest X-ray images with pathological variations.
    """
    # Create base chest X-ray structure
    image = Image.new('L', size, color=40)  # Dark background
    draw = ImageDraw.Draw(image)
    
    # Draw anatomical structures
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Chest cavity outline
    chest_width = int(size[0] * 0.8)
    chest_height = int(size[1] * 0.6)
    chest_x = (size[0] - chest_width) // 2
    chest_y = int(size[1] * 0.2)
    
    # Draw chest cavity
    draw.ellipse([chest_x, chest_y, chest_x + chest_width, chest_y + chest_height], 
                 outline=100, width=2)
    
    # Draw lung fields
    lung_color = 110 if pathology == "normal" else 85
    
    # Left lung
    left_lung_x = int(size[0] * 0.15)
    left_lung_w = int(size[0] * 0.3)
    left_lung_y = int(size[1] * 0.25)
    left_lung_h = int(size[1] * 0.5)
    draw.ellipse([left_lung_x, left_lung_y, left_lung_x + left_lung_w, left_lung_y + left_lung_h], 
                 fill=lung_color)
    
    # Right lung
    right_lung_x = int(size[0] * 0.55)
    right_lung_w = int(size[0] * 0.3)
    right_lung_y = int(size[1] * 0.25)
    right_lung_h = int(size[1] * 0.5)
    draw.ellipse([right_lung_x, right_lung_y, right_lung_x + right_lung_w, right_lung_y + right_lung_h], 
                 fill=lung_color)
    
    # Draw ribs (multiple arcs)
    for i in range(6):
        rib_y = int(size[1] * 0.3) + i * int(size[1] * 0.08)
        rib_start_x = int(size[0] * 0.1)
        rib_end_x = int(size[0] * 0.9)
        
        # Create curved rib structure
        points = []
        for x in range(rib_start_x, rib_end_x, 5):
            curve_y = rib_y - int(15 * np.sin(np.pi * (x - rib_start_x) / (rib_end_x - rib_start_x)))
            points.append((x, curve_y))
        
        if len(points) > 1:
            draw.line(points, fill=130, width=1)
    
    # Draw heart shadow
    heart_x = int(size[0] * 0.4)
    heart_y = int(size[1] * 0.4)
    heart_w = int(size[0] * 0.2)
    heart_h = int(size[1] * 0.3)
    draw.ellipse([heart_x, heart_y, heart_x + heart_w, heart_y + heart_h], fill=75)
    
    # Add pathology-specific findings
    if pathology == "pneumonia":
        # Add infiltrates/consolidation
        # Right lower lobe infiltrate
        infiltrate_x = int(size[0] * 0.6)
        infiltrate_y = int(size[1] * 0.6)
        infiltrate_w = int(size[0] * 0.15)
        infiltrate_h = int(size[1] * 0.15)
        draw.ellipse([infiltrate_x, infiltrate_y, infiltrate_x + infiltrate_w, infiltrate_y + infiltrate_h], 
                     fill=60)
        
        # Add some patchy opacities
        for _ in range(5):
            patch_x = infiltrate_x + np.random.randint(-20, 20)
            patch_y = infiltrate_y + np.random.randint(-20, 20)
            patch_size = np.random.randint(8, 15)
            draw.ellipse([patch_x, patch_y, patch_x + patch_size, patch_y + patch_size], fill=65)
    
    elif pathology == "cardiomegaly":
        # Enlarge heart shadow
        enlarged_heart_x = int(size[0] * 0.35)
        enlarged_heart_y = int(size[1] * 0.35)
        enlarged_heart_w = int(size[0] * 0.3)
        enlarged_heart_h = int(size[1] * 0.4)
        draw.ellipse([enlarged_heart_x, enlarged_heart_y, 
                     enlarged_heart_x + enlarged_heart_w, enlarged_heart_y + enlarged_heart_h], 
                     fill=70)
    
    elif pathology == "pleural_effusion":
        # Add fluid at lung bases
        fluid_y = int(size[1] * 0.7)
        # Right pleural effusion
        draw.rectangle([int(size[0] * 0.55), fluid_y, int(size[0] * 0.85), int(size[1] * 0.8)], 
                      fill=50)
    
    # Apply some realistic medical imaging noise and blur
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    # Add some random noise to simulate digital noise
    img_array = np.array(image)
    noise = np.random.normal(0, 3, img_array.shape).astype(np.uint8)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array).convert('RGB')


def generate_sample_dataset(num_cases=10):
    """Generate a sample dataset of medical cases with corresponding images."""
    
    dataset = []
    pathologies = ["normal", "pneumonia", "cardiomegaly", "pleural_effusion"]
    
    for i in range(num_cases):
        # Select a base case and modify it
        base_case = MEDICAL_REPORTS[i % len(MEDICAL_REPORTS)].copy()
        pathology = pathologies[i % len(pathologies)]
        
        # Generate corresponding image
        chest_xray = create_realistic_chest_xray(pathology=pathology)
        
        case_data = {
            "case_id": f"GENERATED_{i+1:03d}",
            "report": base_case["report"],
            "diagnosis": base_case["diagnosis"],
            "medications": base_case["medications"],
            "procedures": base_case["procedures"],
            "image": chest_xray,
            "pathology": pathology,
            "patient_age": base_case["patient_age"],
            "gender": base_case["gender"]
        }
        
        dataset.append(case_data)
    
    return dataset


def save_sample_images(output_dir="data/sample_images"):
    """Save sample medical images to disk for testing."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    pathologies = ["normal", "pneumonia", "cardiomegaly", "pleural_effusion"]
    
    for i, pathology in enumerate(pathologies):
        for j in range(3):  # 3 samples per pathology
            image = create_realistic_chest_xray(pathology=pathology)
            filename = f"{pathology}_{j+1:02d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")
    
    return output_dir


if __name__ == "__main__":
    # Generate and save sample images
    save_sample_images()
    
    # Generate sample dataset
    dataset = generate_sample_dataset(5)
    print(f"Generated dataset with {len(dataset)} cases")
    
    for case in dataset:
        print(f"Case {case['case_id']}: {case['diagnosis']} ({case['pathology']})")