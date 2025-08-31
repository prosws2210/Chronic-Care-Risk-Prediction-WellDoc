"""
Medication Adherence Tool for simulating realistic medication adherence patterns.
Generates medication lists, dosing schedules, and adherence behaviors for chronic patients.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MedicationAdherenceInput(BaseModel):
    """Input schema for medication adherence simulation."""
    patient_id: str = Field(description="Patient identifier")
    chronic_conditions: List[str] = Field(description="Patient's chronic conditions")
    days_history: int = Field(default=90, description="Number of days to simulate")
    patient_profile: Optional[Dict[str, Any]] = Field(default=None, description="Patient demographic and social profile")

class MedicationAdherenceTool(BaseTool):
    """Tool for simulating realistic medication adherence patterns."""
    
    name: str = "Medication Adherence Simulator"
    description: str = "Generates realistic medication regimens and adherence patterns for chronic care patients"
    args_schema: type[BaseModel] = MedicationAdherenceInput
    
    def __init__(self):
        super().__init__()
        
        # Define medication formularies by condition
        self.medication_formulary = {
            "diabetes": {
                "metformin": {"class": "Biguanide", "frequency": "twice_daily", "adherence_baseline": 0.85},
                "glipizide": {"class": "Sulfonylurea", "frequency": "once_daily", "adherence_baseline": 0.80},
                "insulin_glargine": {"class": "Long-acting Insulin", "frequency": "once_daily", "adherence_baseline": 0.75},
                "insulin_lispro": {"class": "Rapid-acting Insulin", "frequency": "three_times_daily", "adherence_baseline": 0.70},
                "linagliptin": {"class": "DPP-4 Inhibitor", "frequency": "once_daily", "adherence_baseline": 0.85},
                "semaglutide": {"class": "GLP-1 Agonist", "frequency": "weekly", "adherence_baseline": 0.90}
            },
            "heart_failure": {
                "lisinopril": {"class": "ACE Inhibitor", "frequency": "once_daily", "adherence_baseline": 0.82},
                "metoprolol": {"class": "Beta Blocker", "frequency": "twice_daily", "adherence_baseline": 0.78},
                "furosemide": {"class": "Loop Diuretic", "frequency": "once_daily", "adherence_baseline": 0.85},
                "spironolactone": {"class": "Aldosterone Antagonist", "frequency": "once_daily", "adherence_baseline": 0.75},
                "sacubitril_valsartan": {"class": "ARNI", "frequency": "twice_daily", "adherence_baseline": 0.80},
                "dapagliflozin": {"class": "SGLT-2 Inhibitor", "frequency": "once_daily", "adherence_baseline": 0.85}
            },
            "hypertension": {
                "amlodipine": {"class": "Calcium Channel Blocker", "frequency": "once_daily", "adherence_baseline": 0.85},
                "lisinopril": {"class": "ACE Inhibitor", "frequency": "once_daily", "adherence_baseline": 0.82},
                "hydrochlorothiazide": {"class": "Thiazide Diuretic", "frequency": "once_daily", "adherence_baseline": 0.80},
                "losartan": {"class": "ARB", "frequency": "once_daily", "adherence_baseline": 0.83}
            },
            "obesity": {
                "orlistat": {"class": "Lipase Inhibitor", "frequency": "three_times_daily", "adherence_baseline": 0.60},
                "liraglutide": {"class": "GLP-1 Agonist", "frequency": "once_daily", "adherence_baseline": 0.75},
                "semaglutide": {"class": "GLP-1 Agonist", "frequency": "weekly", "adherence_baseline": 0.85}
            },
            "dyslipidemia": {
                "atorvastatin": {"class": "Statin", "frequency": "once_daily", "adherence_baseline": 0.78},
                "rosuvastatin": {"class": "Statin", "frequency": "once_daily", "adherence_baseline": 0.80},
                "ezetimibe": {"class": "Cholesterol Absorption Inhibitor", "frequency": "once_daily", "adherence_baseline": 0.85}
            }
        }
        
        logger.info("MedicationAdherenceTool initialized")
    
    def _run(self, patient_id: str, chronic_conditions: List[str], 
             days_history: int = 90, patient_profile: Optional[Dict[str, Any]] = None) -> str:
        """Generate medication adherence simulation."""
        try:
            logger.info(f"Generating {days_history} days of medication adherence for patient {patient_id}")
            
            # Generate medication regimen based on conditions
            medication_regimen = self._generate_medication_regimen(chronic_conditions, patient_profile)
            
            # Generate adherence patterns over time
            adherence_data = self._generate_adherence_patterns(
                medication_regimen, days_history, patient_profile
            )
            
            # Calculate adherence metrics
            adherence_metrics = self._calculate_adherence_metrics(adherence_data)
            
            # Identify adherence barriers and interventions
            barriers_and_interventions = self._identify_barriers_and_interventions(
                adherence_data, patient_profile
            )
            
            result = {
                "patient_id": patient_id,
                "medication_regimen": medication_regimen,
                "adherence_data": adherence_data,
                "adherence_metrics": adherence_metrics,
                "barriers_and_interventions": barriers_and_interventions,
                "metadata": {
                    "chronic_conditions": chronic_conditions,
                    "days_simulated": days_history,
                    "total_medications": len(medication_regimen)
                }
            }
            
            logger.info(f"Generated medication adherence data for {len(medication_regimen)} medications")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error generating medication adherence data: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _add_supportive_medications(self, conditions: List[str], age: int, 
                                   patient_profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add common supportive medications based on age and conditions."""
        supportive_meds = []
        
        # Add aspirin for cardiovascular protection
        if age > 50 or any(cond in conditions for cond in ["diabetes", "heart_failure", "hypertension"]):
            if np.random.random() < 0.7:  # 70% of eligible patients
                supportive_meds.append({
                    "medication_name": "aspirin",
                    "indication": "cardiovascular_protection",
                    "drug_class": "Antiplatelet",
                    "frequency": "once_daily",
                    "prescribed_date": self._generate_prescription_date(),
                    "dosage": "81 mg",
                    "prescriber": {"provider_type": "Primary Care", "provider_name": "Dr. Smith"},
                    "pharmacy": self._generate_pharmacy_info(patient_profile),
                    "cost_info": {"monthly_cost": 5, "patient_copay": 2, "insurance_coverage": 60},
                    "baseline_adherence": 0.85
                })
        
        # Add vitamin D for elderly patients
        if age > 65:
            if np.random.random() < 0.5:  # 50% of elderly patients
                supportive_meds.append({
                    "medication_name": "vitamin_d3",
                    "indication": "bone_health",
                    "drug_class": "Vitamin Supplement",
                    "frequency": "once_daily",
                    "prescribed_date": self._generate_prescription_date(),
                    "dosage": "1000 IU",
                    "prescriber": {"provider_type": "Primary Care", "provider_name": "Dr. Johnson"},
                    "pharmacy": self._generate_pharmacy_info(patient_profile),
                    "cost_info": {"monthly_cost": 8, "patient_copay": 8, "insurance_coverage": 0},
                    "baseline_adherence": 0.70
                })
        
        return supportive_meds

    def _generate_adherence_patterns(self, medication_regimen: List[Dict[str, Any]], 
                                   days_history: int, patient_profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate daily adherence patterns for each medication."""
        adherence_data = []
        
        # Patient-level adherence factors
        patient_adherence_modifier = self._calculate_patient_adherence_modifier(patient_profile)
        
        for day in range(days_history):
            current_date = datetime.now() - timedelta(days=days_history - day)
            daily_data = {
                "date": current_date.isoformat(),
                "medications": []
            }
            
            for medication in medication_regimen:
                # Calculate daily adherence for this medication
                daily_adherence = self._calculate_daily_adherence(
                    medication, day, days_history, patient_adherence_modifier, current_date
                )
                
                medication_data = {
                    "medication_name": medication["medication_name"],
                    "doses_prescribed": self._get_daily_doses(medication["frequency"]),
                    "doses_taken": daily_adherence["doses_taken"],
                    "adherence_rate": daily_adherence["adherence_rate"],
                    "missed_reasons": daily_adherence.get("missed_reasons", []),
                    "timing_accuracy": daily_adherence["timing_accuracy"]
                }
                daily_data["medications"].append(medication_data)
            
            adherence_data.append(daily_data)
        
        return adherence_data
    
    def _calculate_patient_adherence_modifier(self, patient_profile: Optional[Dict[str, Any]]) -> float:
        """Calculate patient-level adherence modifier based on demographics and social factors."""
        modifier = 1.0
        
        if not patient_profile:
            return modifier
        
        # Age effects
        age = patient_profile.get('age', 65)
        if age > 75:
            modifier *= 0.90  # Slightly lower adherence in very elderly
        elif 65 <= age <= 75:
            modifier *= 1.05  # Better adherence in younger elderly
        
        # Education effects
        education = patient_profile.get('education_level', 'High_School')
        if education in ['College_Grad', 'Advanced_Degree']:
            modifier *= 1.10
        elif education == 'Less_than_HS':
            modifier *= 0.85
        
        # Income effects
        income = patient_profile.get('estimated_income_category', 'Middle')
        if income in ['Low', 'Very_Low']:
            modifier *= 0.80  # Financial barriers
        elif income == 'High':
            modifier *= 1.05
        
        # Insurance effects
        insurance = patient_profile.get('insurance_type', 'Commercial')
        if insurance == 'Self-pay':
            modifier *= 0.70  # Major cost barrier
        elif insurance == 'Medicaid':
            modifier *= 0.90  # Some access barriers
        
        # Transportation effects
        transportation = patient_profile.get('transportation_access', 'Reliable')
        if transportation in ['Limited', 'None']:
            modifier *= 0.85  # Pharmacy access issues
        
        return max(0.3, min(1.3, modifier))  # Constrain between 30% and 130%
    
    def _calculate_daily_adherence(self, medication: Dict[str, Any], day_index: int, 
                                 total_days: int, patient_modifier: float, 
                                 current_date: datetime) -> Dict[str, Any]:
        """Calculate adherence for a specific medication on a specific day."""
        
        baseline_adherence = medication["baseline_adherence"] * patient_modifier
        daily_doses = self._get_daily_doses(medication["frequency"])
        
        # Day of week effects
        day_of_week = current_date.weekday()
        if day_of_week in [5, 6]:  # Weekend
            weekend_modifier = 0.95  # Slightly lower weekend adherence
        else:
            weekend_modifier = 1.0
        
        # Medication complexity effects
        complexity_modifier = 1.0
        if medication["frequency"] == "three_times_daily":
            complexity_modifier = 0.85  # More complex regimens have lower adherence
        elif medication["frequency"] == "twice_daily":
            complexity_modifier = 0.92
        elif medication["frequency"] == "weekly":
            complexity_modifier = 1.05  # Weekly dosing is easier
        
        # Side effects simulation (random events)
        side_effect_modifier = 1.0
        if np.random.random() < 0.05:  # 5% chance of side effects affecting adherence
            side_effect_modifier = 0.30
        
        # Calculate final adherence probability
        adherence_probability = baseline_adherence * weekend_modifier * complexity_modifier * side_effect_modifier
        adherence_probability = max(0, min(1, adherence_probability))
        
        # Determine how many doses were taken
        doses_taken = 0
        timing_accuracy = []
        missed_reasons = []
        
        for dose in range(daily_doses):
            if np.random.random() < adherence_probability:
                doses_taken += 1
                # Timing accuracy (percentage of doses taken on time)
                timing_accuracy.append(np.random.choice([True, False], p=[0.8, 0.2]))
            else:
                # Determine reason for missed dose
                reason = np.random.choice([
                    "forgot", "too_busy", "felt_better", "side_effects", 
                    "cost_concerns", "ran_out"
                ], p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
                missed_reasons.append(reason)
        
        return {
            "doses_taken": doses_taken,
            "adherence_rate": doses_taken / daily_doses if daily_doses > 0 else 0,
            "timing_accuracy": sum(timing_accuracy) / len(timing_accuracy) if timing_accuracy else 0,
            "missed_reasons": missed_reasons
        }
    
    def _get_daily_doses(self, frequency: str) -> int:
        """Get number of daily doses based on frequency."""
        frequency_map = {
            "once_daily": 1,
            "twice_daily": 2,
            "three_times_daily": 3,
            "four_times_daily": 4,
            "weekly": 1/7,  # Will be handled specially
            "as_needed": 1  # Simplified
        }
        
        if frequency == "weekly":
            # For weekly medications, only dose on one day per week
            return 1 if np.random.randint(0, 7) == 0 else 0
        
        return int(frequency_map.get(frequency, 1))
    
    def _calculate_adherence_metrics(self, adherence_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive adherence metrics."""
        metrics = {}
        
        # Group by medication
        medication_names = set()
        for daily_data in adherence_data:
            for med_data in daily_data["medications"]:
                medication_names.add(med_data["medication_name"])
        
        for med_name in medication_names:
            med_data = []
            for daily_data in adherence_data:
                for med_daily in daily_data["medications"]:
                    if med_daily["medication_name"] == med_name:
                        med_data.append(med_daily)
                        break
            
            if med_data:
                # Medication Taking Ratio (MTR)
                total_prescribed = sum(d["doses_prescribed"] for d in med_data)
                total_taken = sum(d["doses_taken"] for d in med_data)
                mtr = total_taken / total_prescribed if total_prescribed > 0 else 0
                
                # Proportion of Days Covered (PDC)
                days_with_medication = sum(1 for d in med_data if d["doses_taken"] > 0)
                pdc = days_with_medication / len(med_data) if med_data else 0
                
                # Average timing accuracy
                timing_accuracies = [d["timing_accuracy"] for d in med_data if d["timing_accuracy"] > 0]
                avg_timing = np.mean(timing_accuracies) if timing_accuracies else 0
                
                # Missed dose reasons analysis
                all_missed_reasons = []
                for d in med_data:
                    all_missed_reasons.extend(d["missed_reasons"])
                
                reason_counts = {}
                for reason in all_missed_reasons:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                metrics[med_name] = {
                    "medication_taking_ratio": round(mtr, 3),
                    "proportion_days_covered": round(pdc, 3),
                    "average_timing_accuracy": round(avg_timing, 3),
                    "adherence_category": self._categorize_adherence(mtr),
                    "missed_dose_reasons": reason_counts,
                    "total_prescribed_doses": total_prescribed,
                    "total_taken_doses": total_taken
                }
        
        # Overall patient adherence
        all_mtrs = [metrics[med]["medication_taking_ratio"] for med in metrics]
        overall_adherence = np.mean(all_mtrs) if all_mtrs else 0
        
        metrics["overall"] = {
            "average_adherence": round(overall_adherence, 3),
            "adherence_category": self._categorize_adherence(overall_adherence),
            "number_of_medications": len(medication_names),
            "consistent_medications": sum(1 for med in metrics.values() 
                                        if isinstance(med, dict) and med.get("medication_taking_ratio", 0) >= 0.8)
        }
        
        return metrics
    
    def _categorize_adherence(self, adherence_rate: float) -> str:
        """Categorize adherence based on rate."""
        if adherence_rate >= 0.8:
            return "Good"
        elif adherence_rate >= 0.6:
            return "Moderate" 
        else:
            return "Poor"
    
    def _identify_barriers_and_interventions(self, adherence_data: List[Dict[str, Any]], 
                                           patient_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify adherence barriers and suggest interventions."""
        
        # Analyze missed dose reasons across all medications
        all_missed_reasons = []
        for daily_data in adherence_data:
            for med_data in daily_data["medications"]:
                all_missed_reasons.extend(med_data["missed_reasons"])
        
        reason_counts = {}
        for reason in all_missed_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Identify primary barriers
        total_missed = sum(reason_counts.values())
        primary_barriers = []
        
        if total_missed > 0:
            for reason, count in reason_counts.items():
                if count / total_missed > 0.2:  # >20% of missed doses
                    primary_barriers.append({
                        "barrier": reason,
                        "frequency": count,
                        "percentage": round(count / total_missed * 100, 1)
                    })
        
        # Suggest interventions based on barriers
        interventions = self._suggest_interventions(primary_barriers, patient_profile)
        
        return {
            "identified_barriers": primary_barriers,
            "suggested_interventions": interventions,
            "risk_level": self._assess_adherence_risk(reason_counts, patient_profile)
        }
    
    def _suggest_interventions(self, barriers: List[Dict[str, Any]], 
                             patient_profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest interventions based on identified barriers."""
        interventions = []
        
        for barrier in barriers:
            barrier_type = barrier["barrier"]
            
            if barrier_type == "forgot":
                interventions.append({
                    "intervention": "Medication Reminder System",
                    "description": "Implement daily medication reminders via phone app or pillbox",
                    "evidence_level": "Strong",
                    "implementation_difficulty": "Low"
                })
                
            elif barrier_type == "too_busy":
                interventions.append({
                    "intervention": "Medication Synchronization",
                    "description": "Align all medication refills to same date for convenience",
                    "evidence_level": "Moderate",
                    "implementation_difficulty": "Low"
                })
                
            elif barrier_type == "cost_concerns":
                interventions.append({
                    "intervention": "Financial Assistance Program",
                    "description": "Connect with pharmaceutical assistance programs or generic alternatives",
                    "evidence_level": "Strong",
                    "implementation_difficulty": "Moderate"
                })
                
            elif barrier_type == "side_effects":
                interventions.append({
                    "intervention": "Medication Review and Adjustment",
                    "description": "Clinical review to address side effects and adjust regimen",
                    "evidence_level": "Strong",
                    "implementation_difficulty": "Moderate"
                })
                
            elif barrier_type == "felt_better":
                interventions.append({
                    "intervention": "Patient Education Program",
                    "description": "Education about importance of continued medication despite feeling better",
                    "evidence_level": "Moderate",
                    "implementation_difficulty": "Low"
                })
        
        return interventions
    
    def _assess_adherence_risk(self, missed_reasons: Dict[str, int], 
                             patient_profile: Optional[Dict[str, Any]]) -> str:
        """Assess overall adherence risk level."""
        
        total_missed = sum(missed_reasons.values())
        
        # High risk indicators
        high_risk_reasons = ["cost_concerns", "side_effects", "ran_out"]
        high_risk_count = sum(missed_reasons.get(reason, 0) for reason in high_risk_reasons)
        
        if patient_profile:
            # Social risk factors
            if patient_profile.get('estimated_income_category') in ['Low', 'Very_Low']:
                high_risk_count += 5
            if patient_profile.get('transportation_access') in ['Limited', 'None']:
                high_risk_count += 3
        
        if high_risk_count > 10 or total_missed > 20:
            return "High"
        elif high_risk_count > 5 or total_missed > 10:
            return "Medium"
        else:
            return "Low"

    def _generate_medication_regimen(self, conditions: List[str], 
                                   patient_profile: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a realistic medication regimen based on chronic conditions."""
        regimen = []
        condition_lower = [c.lower() for c in conditions]
        
        # Age-based considerations
        age = patient_profile.get('age', 65) if patient_profile else 65
        
        for condition in condition_lower:
            if condition in self.medication_formulary:
                condition_meds = self.medication_formulary[condition]
                
                # Select medications based on condition severity and patient factors
                selected_meds = self._select_condition_medications(
                    condition, condition_meds, age, patient_profile
                )
                
                for med_name, med_info in selected_meds.items():
                    medication = {
                        "medication_name": med_name,
                        "indication": condition,
                        "drug_class": med_info["class"],
                        "frequency": med_info["frequency"],
                        "prescribed_date": self._generate_prescription_date(),
                        "dosage": self._generate_dosage(med_name, age),
                        "prescriber": self._generate_prescriber_info(condition),
                        "pharmacy": self._generate_pharmacy_info(patient_profile),
                        "cost_info": self._generate_cost_info(med_name, patient_profile),
                        "baseline_adherence": med_info["adherence_baseline"]
                    }
                    regimen.append(medication)
        
        # Add common supportive medications
        regimen.extend(self._add_supportive_medications(condition_lower, age, patient_profile))
        
        return regimen
    
    def _select_condition_medications(self, condition: str, available_meds: Dict[str, Any],
                                    age: int, patient_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select specific medications for a condition based on patient factors."""
        selected = {}
        
        if condition == "diabetes":
            # Always start with metformin unless contraindicated
            selected["metformin"] = available_meds["metformin"]
            
            # Add additional diabetes medications based on severity
            diabetes_severity = np.random.choice(["mild", "moderate", "severe"], p=[0.3, 0.5, 0.2])
            
            if diabetes_severity in ["moderate", "severe"]:
                second_line = np.random.choice(["glipizide", "linagliptin"], p=[0.4, 0.6])
                selected[second_line] = available_meds[second_line]
            
            if diabetes_severity == "severe":
                insulin_type = np.random.choice(["insulin_glargine", "both"], p=[0.6, 0.4])
                if insulin_type == "both":
                    selected["insulin_glargine"] = available_meds["insulin_glargine"]
                    selected["insulin_lispro"] = available_meds["insulin_lispro"]
                else:
                    selected["insulin_glargine"] = available_meds["insulin_glargine"]
        
        elif condition == "heart_failure":
            # Guideline-directed medical therapy
            selected["lisinopril"] = available_meds["lisinopril"]
            selected["metoprolol"] = available_meds["metoprolol"]
            
            # Add diuretic if symptomatic
            if np.random.random() < 0.8:  # 80% need diuretics
                selected["furosemide"] = available_meds["furosemide"]
            
            # Add spironolactone for reduced EF
            if np.random.random() < 0.6:  # 60% have reduced EF
                selected["spironolactone"] = available_meds["spironolactone"]
        
        elif condition == "hypertension":
            # Start with single agent
            first_line = np.random.choice(["amlodipine", "lisinopril", "hydrochlorothiazide"], p=[0.4, 0.4, 0.2])
            selected[first_line] = available_meds[first_line]
            
            # Add second agent if needed
            if np.random.random() < 0.6:  # 60% need combination therapy
                remaining_meds = [med for med in available_meds.keys() if med not in selected]
                if remaining_meds:
                    second_med = np.random.choice(remaining_meds)
                    selected[second_med] = available_meds[second_med]
        
        return selected
    
    def _generate_prescription_date(self) -> str:
        """Generate a realistic prescription date."""
        # Prescriptions typically within last 6 months
        days_ago = np.random.randint(1, 180)
        prescription_date = datetime.now() - timedelta(days=days_ago)
        return prescription_date.isoformat()
    
    def _generate_dosage(self, medication_name: str, age: int) -> str:
        """Generate realistic dosage information."""
        dosages = {
            "metformin": ["500 mg", "850 mg", "1000 mg"],
            "lisinopril": ["5 mg", "10 mg", "20 mg"],
            "atorvastatin": ["20 mg", "40 mg", "80 mg"],
            "amlodipine": ["5 mg", "10 mg"],
            "furosemide": ["20 mg", "40 mg", "80 mg"],
            "metoprolol": ["25 mg", "50 mg", "100 mg"]
        }
        
        if medication_name in dosages:
            # Elderly patients often get lower doses
            if age > 75:
                return dosages[medication_name][0]  # Lowest dose
            else:
                return np.random.choice(dosages[medication_name])
        
        return "Standard dose"
    
    def _generate_prescriber_info(self, condition: str) -> Dict[str, str]:
        """Generate prescriber information."""
        prescriber_types = {
            "diabetes": ["Endocrinologist", "Primary Care", "Internal Medicine"],
            "heart_failure": ["Cardiologist", "Internal Medicine", "Primary Care"],
            "hypertension": ["Primary Care", "Internal Medicine", "Cardiologist"],
            "obesity": ["Endocrinologist", "Primary Care", "Bariatrician"]
        }
        
        if condition in prescriber_types:
            prescriber_type = np.random.choice(prescriber_types[condition])
        else:
            prescriber_type = "Primary Care"
        
        return {
            "provider_type": prescriber_type,
            "provider_name": f"Dr. {np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}"
        }
    
    def _generate_pharmacy_info(self, patient_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate pharmacy information."""
        pharmacy_types = ["Chain Pharmacy", "Independent Pharmacy", "Mail Order", "Hospital Pharmacy"]
        
        return {
            "pharmacy_type": np.random.choice(pharmacy_types, p=[0.5, 0.2, 0.2, 0.1]),
            "distance_from_home": np.random.choice(["<1 mile", "1-5 miles", "5-10 miles", ">10 miles"], 
                                                  p=[0.4, 0.3, 0.2, 0.1]),
            "accepts_insurance": np.random.choice([True, False], p=[0.9, 0.1])
        }
    
    def _generate_cost_info(self, medication_name: str, 
                          patient_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate medication cost information."""
        
        # Base costs for medications (monthly)
        base_costs = {
            "metformin": 15, "lisinopril": 20, "atorvastatin": 25,
            "amlodipine": 18, "furosemide": 12, "metoprolol": 22,
            "insulin_glargine": 250, "semaglutide": 800
        }
        
        base_cost = base_costs.get(medication_name, 50)  # Default $50
        
        # Insurance coverage simulation
        insurance_type = patient_profile.get('insurance_type', 'Commercial') if patient_profile else 'Commercial'
        
        if insurance_type == "Medicare":
            copay = min(base_cost * 0.2, 47)  # 20% or max $47
        elif insurance_type == "Medicaid":
            copay = min(base_cost * 0.05, 5)   # 5% or max $5
        elif insurance_type == "Commercial":
            copay = min(base_cost * 0.15, 35)  # 15% or max $35
        else:  # Self-pay
            copay = base_cost
        
        return {
            "monthly_cost": base_cost,
            "patient_copay": round(copay, 2),
            "insurance_coverage": round((base_cost - copay) / base_cost * 100, 1)
        }
