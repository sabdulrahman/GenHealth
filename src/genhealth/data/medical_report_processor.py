import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import spacy
from transformers import AutoTokenizer
import torch


@dataclass
class MedicalReport:
    """Structured representation of a medical report."""
    report_id: str
    patient_id: str
    report_text: str
    diagnosis_codes: List[str]
    procedures: List[str]
    medications: List[str]
    findings: Dict[str, Any]
    metadata: Dict[str, Any]


class MedicalReportProcessor:
    """
    Advanced processor for medical reports with clinical NLP capabilities.
    Handles text cleaning, entity extraction, and structured data extraction.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        spacy_model: str = "en_core_sci_sm",  # SciSpacy model for medical NER
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load medical NLP model (requires: pip install scispacy)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Warning: {spacy_model} not found. Using en_core_web_sm instead.")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Medical terminology patterns
        self.diagnosis_patterns = [
            r"(?i)(?:diagnosis|dx|impression|assessment):\s*([^.]+)",
            r"(?i)primary diagnosis:\s*([^.]+)",
            r"(?i)final diagnosis:\s*([^.]+)",
            # More comprehensive patterns
            r"(?i)\bpresents with\s+([^.]+)",
            r"(?i)\bshows\s+([^.]+)",
            r"(?i)\bconsistent with\s+([^.]+)",
            r"(?i)\bhistory of\s+([^.,]+)",
            r"(?i)\bsigns of\s+([^.,]+)",
            r"(?i)\bindicates\s+([^.,]+)",
            r"(?i)\bsuggests\s+([^.,]+)",
            r"(?i)\bdiagnosed with\s+([^.,]+)",
            r"(?i)\bsuffering from\s+([^.,]+)",
        ]
        
        self.medication_patterns = [
            r"(?i)(?:medications?|meds?|drugs?):\s*([^.]+)",
            r"(?i)(?:prescribed|taking|on):\s*([^.]+)",
            r"(?i)\bmetformin\b",
            r"(?i)\baspirin\b",
            r"(?i)\binsulin\b",
            r"(?i)\blisinopril\b",
            r"(?i)\bwarfarin\b",
        ]
        
        self.procedure_patterns = [
            r"(?i)(?:procedure|operation|surgery):\s*([^.]+)",
            r"(?i)(?:performed|underwent):\s*([^.]+)",
            r"(?i)\bECG\b",
            r"(?i)\bchest X-ray\b",
            r"(?i)\bX-ray\b",
            r"(?i)\bCT scan\b",
            r"(?i)\bMRI\b",
            r"(?i)\bblood test\b",
        ]
        
        # Medical abbreviations dictionary
        self.medical_abbrevs = {
            "BP": "blood pressure",
            "HR": "heart rate",
            "RR": "respiratory rate",
            "Temp": "temperature",
            "O2 Sat": "oxygen saturation",
            "CBC": "complete blood count",
            "ECG": "electrocardiogram",
            "CXR": "chest X-ray",
            "CT": "computed tomography",
            "MRI": "magnetic resonance imaging",
            "IV": "intravenous",
            "PO": "per os",
            "PRN": "as needed",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
        }
        
        # Medical conditions dictionary for better detection
        self.medical_conditions = [
            "chest pain", "shortness of breath", "pneumonia", "hypertension", 
            "diabetes", "heart disease", "myocardial infarction", "stroke",
            "fever", "cough", "headache", "nausea", "vomiting", "diarrhea",
            "abdominal pain", "back pain", "joint pain", "fatigue", "dizziness",
            "infection", "inflammation", "fracture", "laceration", "contusion",
            "edema", "rash", "allergic reaction", "asthma", "COPD", "bronchitis",
            "tuberculosis", "cancer", "tumor", "mass", "lesion", "nodule",
            "arrhythmia", "bradycardia", "tachycardia", "hypotension",
            "bilateral infiltrates", "consolidation", "elevated troponin",
            "ST elevation", "crushing chest pain", "radiating pain"
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess medical report text."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Expand medical abbreviations
        for abbrev, full_form in self.medical_abbrevs.items():
            text = re.sub(rf'\b{re.escape(abbrev)}\b', full_form, text, flags=re.IGNORECASE)
        
        # Clean special characters but preserve medical notation
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\/\%\+\<\>\=]', ' ', text)
        
        # Normalize common medical terms
        text = re.sub(r'(?i)\bpatient\b', 'patient', text)
        text = re.sub(r'(?i)\bdoctor\b', 'physician', text)
        
        return text.strip()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using NER."""
        doc = self.nlp(text)
        
        entities = {
            'PERSON': [],
            'CONDITION': [],
            'MEDICATION': [],
            'PROCEDURE': [],
            'ANATOMY': [],
            'TEST': [],
            'DATE': [],
        }
        
        for ent in doc.ents:
            label = ent.label_
            if label in entities:
                entities[label].append(ent.text.strip())
            elif label in ['DISEASE', 'SYMPTOM']:
                entities['CONDITION'].append(ent.text.strip())
            elif label in ['DRUG', 'CHEMICAL']:
                entities['MEDICATION'].append(ent.text.strip())
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured information from medical reports."""
        structured_data = {
            'diagnoses': [],
            'medications': [],
            'procedures': [],
            'vital_signs': {},
            'lab_values': {},
        }
        
        # Extract diagnoses using patterns
        for pattern in self.diagnosis_patterns:
            matches = re.findall(pattern, text)
            structured_data['diagnoses'].extend([match.strip() for match in matches])
        
        # Also look for known medical conditions in the text
        text_lower = text.lower()
        for condition in self.medical_conditions:
            if condition.lower() in text_lower:
                structured_data['diagnoses'].append(condition)
        
        # Extract medications using patterns
        for pattern in self.medication_patterns:
            if '\\b' in pattern:  # Single word patterns
                if re.search(pattern, text):
                    med_name = pattern.replace(r'(?i)\b', '').replace(r'\b', '')
                    structured_data['medications'].append(med_name)
            else:  # Full sentence patterns
                matches = re.findall(pattern, text)
                structured_data['medications'].extend([match.strip() for match in matches])
        
        # Extract procedures using patterns
        for pattern in self.procedure_patterns:
            if '\\b' in pattern:  # Single word patterns
                if re.search(pattern, text):
                    proc_name = pattern.replace(r'(?i)\b', '').replace(r'\b', '')
                    structured_data['procedures'].append(proc_name)
            else:  # Full sentence patterns
                matches = re.findall(pattern, text)
                structured_data['procedures'].extend([match.strip() for match in matches])
        
        # Extract vital signs
        vital_patterns = {
            'blood_pressure': r'(?i)(?:BP|blood pressure):\s*(\d+/\d+)',
            'heart_rate': r'(?i)(?:HR|heart rate):\s*(\d+)',
            'temperature': r'(?i)(?:temp|temperature):\s*(\d+\.?\d*)',
            'respiratory_rate': r'(?i)(?:RR|respiratory rate):\s*(\d+)',
        }
        
        for vital, pattern in vital_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                structured_data['vital_signs'][vital] = matches[0]
        
        # Extract lab values (simplified)
        lab_patterns = {
            'hemoglobin': r'(?i)(?:hgb|hemoglobin):\s*(\d+\.?\d*)',
            'glucose': r'(?i)glucose:\s*(\d+\.?\d*)',
            'creatinine': r'(?i)creatinine:\s*(\d+\.?\d*)',
        }
        
        for lab, pattern in lab_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                structured_data['lab_values'][lab] = float(matches[0])
        
        # Remove duplicates and clean up
        structured_data['diagnoses'] = list(set([d.strip() for d in structured_data['diagnoses'] if d.strip()]))
        structured_data['medications'] = list(set([m.strip() for m in structured_data['medications'] if m.strip()]))
        structured_data['procedures'] = list(set([p.strip() for p in structured_data['procedures'] if p.strip()]))
        
        return structured_data
    
    def tokenize_report(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize medical report text."""
        processed_text = self.preprocess_text(text)
        
        encoded = self.tokenizer(
            processed_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }
    
    def process_report(self, report_text: str, report_id: str = None) -> MedicalReport:
        """Process a complete medical report."""
        # Preprocess text
        cleaned_text = self.preprocess_text(report_text)
        
        # Extract entities
        entities = self.extract_medical_entities(cleaned_text)
        
        # Extract structured data
        structured_data = self.extract_structured_data(cleaned_text)
        
        # Create medical report object
        medical_report = MedicalReport(
            report_id=report_id or f"report_{hash(report_text) % 10000}",
            patient_id="unknown",  # Would be extracted from metadata
            report_text=cleaned_text,
            diagnosis_codes=structured_data['diagnoses'],
            procedures=structured_data['procedures'],
            medications=structured_data['medications'],
            findings={
                'entities': entities,
                'vital_signs': structured_data['vital_signs'],
                'lab_values': structured_data['lab_values'],
            },
            metadata={'processed': True}
        )
        
        return medical_report
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of medical reports."""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        encoded = self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return [
            {
                'input_ids': encoded['input_ids'][i],
                'attention_mask': encoded['attention_mask'][i],
            }
            for i in range(len(texts))
        ]
    
    def extract_clinical_features(self, report: MedicalReport) -> torch.Tensor:
        """Extract numerical features for clinical analysis."""
        features = []
        
        # Text-based features
        features.append(len(report.report_text.split()))  # Word count
        features.append(len(report.diagnosis_codes))      # Number of diagnoses
        features.append(len(report.medications))          # Number of medications
        features.append(len(report.procedures))           # Number of procedures
        
        # Entity counts
        entities = report.findings.get('entities', {})
        for entity_type in ['CONDITION', 'MEDICATION', 'PROCEDURE', 'ANATOMY', 'TEST']:
            features.append(len(entities.get(entity_type, [])))
        
        # Vital signs (normalized)
        vital_signs = report.findings.get('vital_signs', {})
        if 'heart_rate' in vital_signs:
            try:
                hr = float(vital_signs['heart_rate'])
                features.append(hr / 100.0)  # Normalize
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        if 'temperature' in vital_signs:
            try:
                temp = float(vital_signs['temperature'])
                features.append((temp - 98.6) / 10.0)  # Normalize around normal
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Lab values (normalized)
        lab_values = report.findings.get('lab_values', {})
        for lab in ['hemoglobin', 'glucose', 'creatinine']:
            if lab in lab_values:
                features.append(lab_values[lab] / 100.0)  # Simple normalization
            else:
                features.append(0.0)
        
        return torch.tensor(features, dtype=torch.float32)