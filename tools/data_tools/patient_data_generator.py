"""
Patient Data Generator Tool for creating synthetic chronic care patient profiles.
Generates realistic patient demographics, medical histories, and baseline characteristics.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from faker import Faker
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PatientDataGeneratorInput(BaseModel):
    """Input schema for patient data generation."""
    num_patients: int = Field(description="Number of patients to generate")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    conditions: List[str] = Field(default=["diabetes", "heart_failure", "obesity"], 
                                 description="List of chronic conditions to include")

class PatientDataGeneratorTool(BaseTool):
    """Tool for generating synthetic chronic care patient data."""
    
    name: str = "Patient Data Generator"
    description: str = "Generates comprehensive synthetic patient profiles for chronic care modeling"
    args_schema: type[BaseModel] = PatientDataGeneratorInput
    
    def __init__(self):
        super().__init__()
        self.fake = Faker()
        Faker.seed(42)
        np.random.seed(42)
        logger.info("PatientDataGeneratorTool initialized")
    
    def _run(self, num_patients: int, seed: Optional[int] = 42, 
             conditions: List[str] = None) -> str:
        """Generate synthetic patient data."""
        try:
            if seed:
                np.random.seed(seed)
                Faker.seed(seed)
            
            if conditions is None:
                conditions = ["diabetes", "heart_failure", "obesity"]
            
            logger.info(f"Generating {num_patients} synthetic patients")
            
            patients = []
            for i in range(num_patients):
                patient = self._generate_single_patient(i + 1, conditions)
                patients.append(patient)
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(patients)
            
            # Save to JSON for CrewAI consumption
            result = {
                "patients": patients,
                "metadata": {
                    "count": num_patients,
                    "conditions_included": conditions,
                    "generation_timestamp": datetime.now().isoformat(),
                    "seed": seed
                },
                "summary_statistics": self._generate_summary_stats(df)
            }
            
            logger.info(f"Generated {len(patients)} synthetic patients successfully")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error generating patient data: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _generate_single_patient(self, patient_id: int, conditions: List[str]) -> Dict[str, Any]:
        """Generate a single synthetic patient profile."""
        # Basic demographics
        age = np.random.normal(65, 15)  # Chronic care population tends to be older
        age = max(18, min(90, int(age)))  # Constrain to realistic range
        
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
        
        # Race/ethnicity with realistic distribution
        race_ethnicity = np.random.choice([
            'White', 'Black/African American', 'Hispanic/Latino', 
            'Asian/Pacific Islander', 'Native American', 'Mixed/Other'
        ], p=[0.60, 0.18, 0.12, 0.06, 0.02, 0.02])
        
        # Socioeconomic indicators
        insurance_type = np.random.choice([
            'Commercial', 'Medicare', 'Medicaid', 'Self-pay', 'Other'
        ], p=[0.45, 0.35, 0.15, 0.03, 0.02])
        
        # Geographic and social factors
        zip_code = self.fake.zipcode()
        marital_status = np.random.choice([
            'Single', 'Married', 'Divorced', 'Widowed'
        ], p=[0.25, 0.55, 0.15, 0.05])
        
        # Primary chronic condition assignment
        primary_condition = np.random.choice(conditions)
        
        # Generate condition-specific data
        condition_data = self._generate_condition_data(primary_condition, age, gender)
        
        # Comorbidities (realistic co-occurrence)
        comorbidities = self._generate_comorbidities(primary_condition, age)
        
        # Baseline vitals and measurements
        vitals = self._generate_baseline_vitals(age, gender, condition_data)
        
        # Healthcare utilization history
        utilization = self._generate_utilization_history(age, primary_condition, comorbidities)
        
        # Social determinants of health
        social_determinants = self._generate_social_determinants(insurance_type, zip_code)
        
        # Risk factors and lifestyle
        risk_factors = self._generate_risk_factors(age, gender, primary_condition)
        
        # Generate deterioration risk (ground truth for modeling)
        deterioration_risk = self._calculate_deterioration_risk(
            age, condition_data, comorbidities, vitals, utilization, risk_factors
        )
        
        patient = {
            "patient_id": f"SYNTH_{patient_id:06d}",
            "demographics": {
                "age": age,
                "gender": gender,
                "race_ethnicity": race_ethnicity,
                "marital_status": marital_status,
                "insurance_type": insurance_type,
                "zip_code": zip_code
            },
            "clinical_profile": {
                "primary_condition": primary_condition,
                "condition_data": condition_data,
                "comorbidities": comorbidities,
                "baseline_vitals": vitals,
                "social_determinants": social_determinants,
                "risk_factors": risk_factors
            },
            "healthcare_utilization": utilization,
            "deterioration_risk": deterioration_risk,
            "generated_timestamp": datetime.now().isoformat()
        }
        
        return patient
    
    def _generate_condition_data(self, condition: str, age: int, gender: str) -> Dict[str, Any]:
        """Generate condition-specific clinical data."""
        if condition == "diabetes":
            return {
                "diabetes_type": np.random.choice(['Type_1', 'Type_2'], p=[0.10, 0.90]),
                "years_since_diagnosis": max(0, int(np.random.exponential(8))),
                "last_hba1c": max(5.5, min(15.0, np.random.normal(8.2, 1.5))),
                "insulin_dependent": np.random.choice([True, False], p=[0.35, 0.65]),
                "diabetic_complications": np.random.choice([
                    'None', 'Retinopathy', 'Nephropathy', 'Neuropathy', 'Multiple'
                ], p=[0.40, 0.20, 0.15, 0.15, 0.10])
            }
        
        elif condition == "heart_failure":
            return {
                "hf_type": np.random.choice(['HFrEF', 'HFpEF'], p=[0.60, 0.40]),
                "nyha_class": np.random.choice(['I', 'II', 'III', 'IV'], p=[0.25, 0.45, 0.25, 0.05]),
                "ejection_fraction": max(15, min(70, int(np.random.normal(45, 15)))),
                "years_since_diagnosis": max(0, int(np.random.exponential(5))),
                "last_bnp": max(50, int(np.random.lognormal(6, 1))),
                "device_therapy": np.random.choice([
                    'None', 'Pacemaker', 'ICD', 'CRT-D'
                ], p=[0.60, 0.15, 0.15, 0.10])
            }
        
        elif condition == "obesity":
            # BMI calculation based on realistic height/weight
            height_cm = np.random.normal(170 if gender == 'Male' else 160, 8)
            bmi = max(30, min(50, np.random.normal(35, 5)))  # Obesity BMI range
            weight_kg = bmi * (height_cm / 100) ** 2
            
            return {
                "bmi": round(bmi, 1),
                "weight_kg": round(weight_kg, 1),
                "height_cm": round(height_cm, 1),
                "obesity_class": "Class_I" if bmi < 35 else "Class_II" if bmi < 40 else "Class_III",
                "weight_loss_attempts": max(0, int(np.random.poisson(2))),
                "bariatric_surgery_candidate": np.random.choice([True, False], p=[0.15, 0.85]),
                "metabolic_syndrome": np.random.choice([True, False], p=[0.75, 0.25])
            }
        
        return {}
    
    def _generate_comorbidities(self, primary_condition: str, age: int) -> List[str]:
        """Generate realistic comorbidities based on primary condition and age."""
        comorbidities = []
        
        # Age-related probability adjustments
        age_factor = min(2.0, age / 60)  # Increase probability with age
        
        # Common comorbidities by primary condition
        if primary_condition == "diabetes":
            if np.random.random() < 0.70 * age_factor:
                comorbidities.append("Hypertension")
            if np.random.random() < 0.45 * age_factor:
                comorbidities.append("Dyslipidemia")
            if np.random.random() < 0.25 * age_factor:
                comorbidities.append("Coronary_Artery_Disease")
            if np.random.random() < 0.20 * age_factor:
                comorbidities.append("Chronic_Kidney_Disease")
        
        elif primary_condition == "heart_failure":
            if np.random.random() < 0.85 * age_factor:
                comorbidities.append("Hypertension")
            if np.random.random() < 0.60 * age_factor:
                comorbidities.append("Coronary_Artery_Disease")
            if np.random.random() < 0.40 * age_factor:
                comorbidities.append("Atrial_Fibrillation")
            if np.random.random() < 0.35 * age_factor:
                comorbidities.append("Diabetes")
            if np.random.random() < 0.30 * age_factor:
                comorbidities.append("Chronic_Kidney_Disease")
        
        elif primary_condition == "obesity":
            if np.random.random() < 0.65 * age_factor:
                comorbidities.append("Hypertension")
            if np.random.random() < 0.55 * age_factor:
                comorbidities.append("Diabetes")
            if np.random.random() < 0.50 * age_factor:
                comorbidities.append("Dyslipidemia")
            if np.random.random() < 0.35 * age_factor:
                comorbidities.append("Sleep_Apnea")
            if np.random.random() < 0.25 * age_factor:
                comorbidities.append("Osteoarthritis")
        
        # Add some random additional conditions
        additional_conditions = [
            "COPD", "Depression", "Anxiety", "Osteoporosis", "GERD"
        ]
        for condition in additional_conditions:
            if np.random.random() < 0.15 * age_factor:
                comorbidities.append(condition)
        
        return list(set(comorbidities))  # Remove duplicates
    
    def _generate_baseline_vitals(self, age: int, gender: str, condition_data: Dict) -> Dict[str, Any]:
        """Generate baseline vital signs and measurements."""
        # Blood pressure (influenced by age and conditions)
        if age > 60:
            sbp_mean, dbp_mean = 140, 85
        else:
            sbp_mean, dbp_mean = 125, 80
        
        systolic_bp = max(90, min(200, int(np.random.normal(sbp_mean, 20))))
        diastolic_bp = max(60, min(120, int(np.random.normal(dbp_mean, 15))))
        
        # Heart rate
        hr_mean = 75 if gender == 'Female' else 70
        heart_rate = max(50, min(120, int(np.random.normal(hr_mean, 12))))
        
        # Temperature (mostly normal with some variation)
        temperature = round(np.random.normal(98.6, 0.8), 1)
        
        # Oxygen saturation
        o2_sat = max(88, min(100, int(np.random.normal(97, 2))))
        
        return {
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
            "temperature_f": temperature,
            "oxygen_saturation": o2_sat,
            "respiratory_rate": max(12, min(24, int(np.random.normal(16, 3))))
        }
    
    def _generate_utilization_history(self, age: int, primary_condition: str, 
                                    comorbidities: List[str]) -> Dict[str, Any]:
        """Generate healthcare utilization history."""
        # Adjust utilization based on age and condition complexity
        complexity_factor = len(comorbidities) * 0.5 + (age / 80)
        
        # Emergency visits (higher for complex patients)
        ed_visits_12m = max(0, int(np.random.poisson(1 * complexity_factor)))
        
        # Hospitalizations
        hospitalizations_12m = max(0, int(np.random.poisson(0.5 * complexity_factor)))
        
        # Outpatient visits
        outpatient_visits_12m = max(2, int(np.random.poisson(6 * complexity_factor)))
        
        # Specialist visits
        specialist_visits_12m = max(0, int(np.random.poisson(3 * complexity_factor)))
        
        return {
            "ed_visits_12_months": ed_visits_12m,
            "hospitalizations_12_months": hospitalizations_12m,
            "outpatient_visits_12_months": outpatient_visits_12m,
            "specialist_visits_12_months": specialist_visits_12m,
            "last_hospitalization_days_ago": max(0, int(np.random.exponential(180))) if hospitalizations_12m > 0 else None,
            "primary_care_provider": np.random.choice([True, False], p=[0.85, 0.15])
        }
    
    def _generate_social_determinants(self, insurance_type: str, zip_code: str) -> Dict[str, Any]:
        """Generate social determinants of health."""
        # Income estimation based on insurance type
        if insurance_type == "Commercial":
            income_category = np.random.choice(['Middle', 'Upper_Middle', 'High'], p=[0.4, 0.4, 0.2])
        elif insurance_type == "Medicare":
            income_category = np.random.choice(['Low', 'Middle', 'Upper_Middle'], p=[0.3, 0.5, 0.2])
        elif insurance_type == "Medicaid":
            income_category = np.random.choice(['Low', 'Very_Low'], p=[0.7, 0.3])
        else:
            income_category = np.random.choice(['Low', 'Middle'], p=[0.6, 0.4])
        
        return {
            "estimated_income_category": income_category,
            "education_level": np.random.choice([
                'Less_than_HS', 'High_School', 'Some_College', 'College_Grad', 'Advanced_Degree'
            ], p=[0.15, 0.30, 0.25, 0.20, 0.10]),
            "employment_status": np.random.choice([
                'Employed', 'Retired', 'Unemployed', 'Disabled'
            ], p=[0.40, 0.35, 0.15, 0.10]),
            "housing_stability": np.random.choice(['Stable', 'Unstable'], p=[0.80, 0.20]),
            "transportation_access": np.random.choice(['Reliable', 'Limited', 'None'], p=[0.70, 0.25, 0.05]),
            "food_security": np.random.choice(['Secure', 'Insecure'], p=[0.75, 0.25])
        }
    
    def _generate_risk_factors(self, age: int, gender: str, primary_condition: str) -> Dict[str, Any]:
        """Generate lifestyle and behavioral risk factors."""
        return {
            "smoking_status": np.random.choice([
                'Never', 'Former', 'Current'
            ], p=[0.50, 0.35, 0.15]),
            "alcohol_use": np.random.choice([
                'None', 'Moderate', 'Heavy'
            ], p=[0.40, 0.50, 0.10]),
            "exercise_frequency": np.random.choice([
                'Never', 'Rarely', 'Sometimes', 'Regularly'
            ], p=[0.25, 0.30, 0.30, 0.15]),
            "diet_quality": np.random.choice([
                'Poor', 'Fair', 'Good', 'Excellent'
            ], p=[0.20, 0.40, 0.30, 0.10]),
            "medication_adherence": np.random.choice([
                'Poor', 'Fair', 'Good', 'Excellent'
            ], p=[0.15, 0.25, 0.35, 0.25]),
            "sleep_quality": np.random.choice([
                'Poor', 'Fair', 'Good'
            ], p=[0.30, 0.45, 0.25]),
            "stress_level": np.random.choice([
                'Low', 'Moderate', 'High'
            ], p=[0.25, 0.50, 0.25])
        }
    
    def _calculate_deterioration_risk(self, age: int, condition_data: Dict, 
                                    comorbidities: List[str], vitals: Dict,
                                    utilization: Dict, risk_factors: Dict) -> Dict[str, Any]:
        """Calculate synthetic deterioration risk (ground truth for modeling)."""
        risk_score = 0.0
        
        # Age factor (higher risk with age)
        risk_score += min(30, age - 40) * 0.5
        
        # Comorbidity burden
        risk_score += len(comorbidities) * 5
        
        # Vital signs (abnormal values increase risk)
        if vitals['systolic_bp'] > 160:
            risk_score += 10
        if vitals['heart_rate'] > 100:
            risk_score += 5
        
        # Healthcare utilization (recent hospitalizations increase risk)
        risk_score += utilization['hospitalizations_12_months'] * 15
        risk_score += utilization['ed_visits_12_months'] * 8
        
        # Lifestyle factors
        if risk_factors['medication_adherence'] == 'Poor':
            risk_score += 20
        if risk_factors['smoking_status'] == 'Current':
            risk_score += 10
        
        # Condition-specific risk factors
        if 'last_hba1c' in condition_data and condition_data['last_hba1c'] > 9.0:
            risk_score += 15
        if 'nyha_class' in condition_data and condition_data['nyha_class'] in ['III', 'IV']:
            risk_score += 20
        
        # Add some random variation
        risk_score += np.random.normal(0, 5)
        
        # Convert to probability (0-100%)
        probability = min(95, max(5, risk_score))
        
        # Categorize risk level
        if probability >= 70:
            risk_level = "High"
        elif probability >= 30:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Determine if deterioration occurs (for ground truth)
        deterioration_occurs = np.random.random() < (probability / 100)
        
        return {
            "deterioration_probability": round(probability, 1),
            "risk_level": risk_level,
            "deterioration_occurs": deterioration_occurs,
            "days_to_deterioration": max(1, int(np.random.exponential(45))) if deterioration_occurs else None,
            "risk_factors_contributing": self._identify_top_risk_factors(
                age, comorbidities, vitals, utilization, risk_factors, condition_data
            )
        }
    
    def _identify_top_risk_factors(self, age: int, comorbidities: List[str], 
                                 vitals: Dict, utilization: Dict, 
                                 risk_factors: Dict, condition_data: Dict) -> List[str]:
        """Identify top contributing risk factors for this patient."""
        factors = []
        
        if age > 75:
            factors.append("Advanced_Age")
        if len(comorbidities) > 3:
            factors.append("Multiple_Comorbidities")
        if utilization['hospitalizations_12_months'] > 0:
            factors.append("Recent_Hospitalization")
        if risk_factors['medication_adherence'] in ['Poor', 'Fair']:
            factors.append("Poor_Medication_Adherence")
        if vitals['systolic_bp'] > 160:
            factors.append("Uncontrolled_Hypertension")
        if 'last_hba1c' in condition_data and condition_data['last_hba1c'] > 8.5:
            factors.append("Poor_Glycemic_Control")
        
        return factors[:5]  # Return top 5 factors
    
    def _generate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the generated patient cohort."""
        try:
            return {
                "demographics": {
                    "mean_age": float(df['demographics'].apply(lambda x: x['age']).mean()),
                    "gender_distribution": df['demographics'].apply(lambda x: x['gender']).value_counts().to_dict(),
                    "race_distribution": df['demographics'].apply(lambda x: x['race_ethnicity']).value_counts().to_dict()
                },
                "clinical": {
                    "primary_conditions": df['clinical_profile'].apply(lambda x: x['primary_condition']).value_counts().to_dict(),
                    "risk_level_distribution": df['deterioration_risk'].apply(lambda x: x['risk_level']).value_counts().to_dict(),
                    "deterioration_rate": float(df['deterioration_risk'].apply(lambda x: x['deterioration_occurs']).mean())
                },
                "healthcare_utilization": {
                    "mean_hospitalizations": float(df['healthcare_utilization'].apply(lambda x: x['hospitalizations_12_months']).mean()),
                    "mean_ed_visits": float(df['healthcare_utilization'].apply(lambda x: x['ed_visits_12_months']).mean())
                }
            }
        except Exception as e:
            logger.error(f"Error generating summary stats: {str(e)}")
            return {"error": "Unable to generate summary statistics"}
