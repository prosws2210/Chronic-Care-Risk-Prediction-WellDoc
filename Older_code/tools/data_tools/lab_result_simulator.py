"""
Lab Result Simulator Tool for generating realistic laboratory test results.
Creates temporal patterns for blood tests, biomarkers, and diagnostic values.
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

class LabResultSimulatorInput(BaseModel):
    """Input schema for lab result simulation."""
    patient_id: str = Field(description="Patient identifier")
    chronic_conditions: List[str] = Field(description="Patient's chronic conditions")
    days_history: int = Field(default=180, description="Number of days of history to generate")
    test_frequency: str = Field(default="monthly", description="Testing frequency")
    baseline_labs: Optional[Dict[str, float]] = Field(default=None, description="Baseline lab values")

class LabResultSimulatorTool(BaseTool):
    """Tool for simulating realistic laboratory test results over time."""
    
    name: str = "Laboratory Result Simulator"
    description: str = "Generates realistic temporal laboratory test patterns for chronic care patients"
    args_schema: type[BaseModel] = LabResultSimulatorInput
    
    def __init__(self):
        super().__init__()
        # Define normal ranges for lab tests
        self.lab_ranges = {
            # Basic Metabolic Panel
            "glucose_fasting": {"normal": (70, 100), "unit": "mg/dL"},
            "sodium": {"normal": (135, 145), "unit": "mEq/L"},
            "potassium": {"normal": (3.5, 5.1), "unit": "mEq/L"},
            "chloride": {"normal": (98, 107), "unit": "mEq/L"},
            "bun": {"normal": (7, 20), "unit": "mg/dL"},
            "creatinine": {"normal": (0.6, 1.3), "unit": "mg/dL"},
            
            # Lipid Panel
            "total_cholesterol": {"normal": (0, 200), "unit": "mg/dL"},
            "ldl_cholesterol": {"normal": (0, 100), "unit": "mg/dL"},
            "hdl_cholesterol": {"normal": (40, 60), "unit": "mg/dL"},
            "triglycerides": {"normal": (0, 150), "unit": "mg/dL"},
            
            # Diabetes Markers
            "hba1c": {"normal": (4.0, 5.6), "unit": "%"},
            "glucose_random": {"normal": (70, 140), "unit": "mg/dL"},
            
            # Cardiac Markers
            "bnp": {"normal": (0, 100), "unit": "pg/mL"},
            "nt_probnp": {"normal": (0, 125), "unit": "pg/mL"},
            "troponin_i": {"normal": (0, 0.04), "unit": "ng/mL"},
            
            # Liver Function
            "alt": {"normal": (7, 56), "unit": "U/L"},
            "ast": {"normal": (10, 40), "unit": "U/L"},
            "albumin": {"normal": (3.5, 5.0), "unit": "g/dL"},
            
            # Complete Blood Count
            "hemoglobin": {"normal": (12.0, 18.0), "unit": "g/dL"},
            "hematocrit": {"normal": (36, 54), "unit": "%"},
            "wbc": {"normal": (4.5, 11.0), "unit": "K/uL"},
            "platelets": {"normal": (150, 450), "unit": "K/uL"},
            
            # Inflammatory Markers
            "crp": {"normal": (0, 3.0), "unit": "mg/L"},
            "esr": {"normal": (0, 30), "unit": "mm/hr"},
            
            # Kidney Function
            "egfr": {"normal": (90, 120), "unit": "mL/min/1.73m2"},
            "microalbumin": {"normal": (0, 30), "unit": "mg/g"},
            
            # Thyroid Function
            "tsh": {"normal": (0.4, 4.0), "unit": "mIU/L"},
            "t4_free": {"normal": (0.8, 1.8), "unit": "ng/dL"}
        }
        logger.info("LabResultSimulatorTool initialized")
    
    def _run(self, patient_id: str, chronic_conditions: List[str], 
             days_history: int = 180, test_frequency: str = "monthly",
             baseline_labs: Optional[Dict[str, float]] = None) -> str:
        """Generate time-series laboratory results."""
        try:
            logger.info(f"Generating {days_history} days of lab results for patient {patient_id}")
            
            # Generate test dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_history)
            test_dates = self._generate_test_dates(start_date, end_date, test_frequency)
            
            # Determine which tests to perform based on conditions
            test_panels = self._determine_test_panels(chronic_conditions)
            
            # Generate baseline values if not provided
            if baseline_labs is None:
                baseline_labs = self._generate_baseline_labs(chronic_conditions, test_panels)
            
            # Generate lab results over time
            lab_results = []
            for date in test_dates:
                result = self._generate_lab_result_set(
                    date, baseline_labs, chronic_conditions, test_panels,
                    len(lab_results), len(test_dates)
                )
                lab_results.append(result)
            
            result = {
                "patient_id": patient_id,
                "lab_results": lab_results,
                "metadata": {
                    "chronic_conditions": chronic_conditions,
                    "days_history": days_history,
                    "test_frequency": test_frequency,
                    "total_test_dates": len(test_dates),
                    "test_panels_included": test_panels
                },
                "trend_analysis": self._analyze_trends(lab_results, test_panels)
            }
            
            logger.info(f"Generated lab results for {len(test_dates)} test dates")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error generating lab results: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _generate_test_dates(self, start_date: datetime, end_date: datetime,
                           frequency: str) -> List[datetime]:
        """Generate realistic test dates based on frequency."""
        test_dates = []
        
        if frequency == "monthly":
            current = start_date
            while current <= end_date:
                # Add some variation (±5 days)
                variation = timedelta(days=np.random.randint(-5, 6))
                test_date = current + variation
                if start_date <= test_date <= end_date:
                    test_dates.append(test_date)
                current += timedelta(days=30)
                
        elif frequency == "quarterly":
            current = start_date
            while current <= end_date:
                variation = timedelta(days=np.random.randint(-7, 8))
                test_date = current + variation
                if start_date <= test_date <= end_date:
                    test_dates.append(test_date)
                current += timedelta(days=90)
                
        elif frequency == "biannual":
            current = start_date
            while current <= end_date:
                variation = timedelta(days=np.random.randint(-14, 15))
                test_date = current + variation
                if start_date <= test_date <= end_date:
                    test_dates.append(test_date)
                current += timedelta(days=180)
        
        # Simulate some missed appointments (10% chance)
        if len(test_dates) > 2:
            missed_count = max(0, int(len(test_dates) * 0.1))
            missed_indices = np.random.choice(len(test_dates), missed_count, replace=False)
            test_dates = [date for i, date in enumerate(test_dates) if i not in missed_indices]
        
        return sorted(test_dates)
    
    def _determine_test_panels(self, conditions: List[str]) -> List[str]:
        """Determine which lab tests to include based on chronic conditions."""
        panels = ["basic_metabolic"]  # Always include basic panel
        
        condition_lower = [c.lower() for c in conditions]
        
        if any(cond in condition_lower for cond in ["diabetes", "type_1", "type_2"]):
            panels.extend(["diabetes_monitoring", "lipid_panel"])
            
        if any(cond in condition_lower for cond in ["heart_failure", "cardiovascular", "coronary"]):
            panels.extend(["cardiac_markers", "lipid_panel"])
            
        if "obesity" in condition_lower:
            panels.extend(["lipid_panel", "diabetes_screening", "liver_function"])
            
        if any(cond in condition_lower for cond in ["hypertension", "kidney", "renal"]):
            panels.extend(["kidney_function"])
            
        if any(cond in condition_lower for cond in ["copd", "inflammatory"]):
            panels.extend(["inflammatory_markers"])
        
        # Add routine monitoring
        panels.append("complete_blood_count")
        
        return list(set(panels))  # Remove duplicates
    
    def _generate_baseline_labs(self, conditions: List[str], panels: List[str]) -> Dict[str, float]:
        """Generate realistic baseline laboratory values."""
        baseline = {}
        
        for panel in panels:
            if panel == "basic_metabolic":
                baseline.update({
                    "glucose_fasting": self._generate_baseline_value("glucose_fasting", conditions),
                    "sodium": self._generate_baseline_value("sodium", conditions),
                    "potassium": self._generate_baseline_value("potassium", conditions),
                    "chloride": self._generate_baseline_value("chloride", conditions),
                    "bun": self._generate_baseline_value("bun", conditions),
                    "creatinine": self._generate_baseline_value("creatinine", conditions)
                })
                
            elif panel == "diabetes_monitoring":
                baseline.update({
                    "hba1c": self._generate_baseline_value("hba1c", conditions),
                    "glucose_random": self._generate_baseline_value("glucose_random", conditions)
                })
                
            elif panel == "lipid_panel":
                baseline.update({
                    "total_cholesterol": self._generate_baseline_value("total_cholesterol", conditions),
                    "ldl_cholesterol": self._generate_baseline_value("ldl_cholesterol", conditions),
                    "hdl_cholesterol": self._generate_baseline_value("hdl_cholesterol", conditions),
                    "triglycerides": self._generate_baseline_value("triglycerides", conditions)
                })
                
            elif panel == "cardiac_markers":
                baseline.update({
                    "bnp": self._generate_baseline_value("bnp", conditions),
                    "nt_probnp": self._generate_baseline_value("nt_probnp", conditions),
                    "troponin_i": self._generate_baseline_value("troponin_i", conditions)
                })
                
            elif panel == "complete_blood_count":
                baseline.update({
                    "hemoglobin": self._generate_baseline_value("hemoglobin", conditions),
                    "hematocrit": self._generate_baseline_value("hematocrit", conditions),
                    "wbc": self._generate_baseline_value("wbc", conditions),
                    "platelets": self._generate_baseline_value("platelets", conditions)
                })
                
            elif panel == "kidney_function":
                baseline.update({
                    "egfr": self._generate_baseline_value("egfr", conditions),
                    "microalbumin": self._generate_baseline_value("microalbumin", conditions)
                })
                
            elif panel == "liver_function":
                baseline.update({
                    "alt": self._generate_baseline_value("alt", conditions),
                    "ast": self._generate_baseline_value("ast", conditions),
                    "albumin": self._generate_baseline_value("albumin", conditions)
                })
                
            elif panel == "inflammatory_markers":
                baseline.update({
                    "crp": self._generate_baseline_value("crp", conditions),
                    "esr": self._generate_baseline_value("esr", conditions)
                })
        
        return baseline
    
    def _generate_baseline_value(self, test_name: str, conditions: List[str]) -> float:
        """Generate a realistic baseline value for a specific test."""
        if test_name not in self.lab_ranges:
            return 0.0
        
        normal_range = self.lab_ranges[test_name]["normal"]
        condition_lower = [c.lower() for c in conditions]
        
        # Condition-specific adjustments
        if test_name == "hba1c":
            if any(cond in condition_lower for cond in ["diabetes", "type_1", "type_2"]):
                # Diabetic patients have higher HbA1c
                return max(6.5, min(12.0, np.random.normal(8.0, 1.5)))
            else:
                return np.random.uniform(4.5, 5.5)
                
        elif test_name == "glucose_fasting":
            if any(cond in condition_lower for cond in ["diabetes", "type_1", "type_2"]):
                return max(100, min(300, np.random.normal(150, 40)))
            else:
                return np.random.uniform(normal_range[0], normal_range[1])
                
        elif test_name in ["bnp", "nt_probnp"]:
            if "heart_failure" in condition_lower:
                # Heart failure patients have elevated BNP
                if test_name == "bnp":
                    return max(100, int(np.random.lognormal(5.5, 0.8)))
                else:  # nt_probnp
                    return max(125, int(np.random.lognormal(6.5, 0.8)))
            else:
                return np.random.uniform(normal_range[0], normal_range[1])
                
        elif test_name in ["total_cholesterol", "ldl_cholesterol", "triglycerides"]:
            if any(cond in condition_lower for cond in ["diabetes", "obesity", "cardiovascular"]):
                # Higher cholesterol in these conditions
                multiplier = 1.3 if test_name == "triglycerides" else 1.2
                return np.random.uniform(normal_range[1] * 0.8, normal_range[1] * multiplier)
            else:
                return np.random.uniform(normal_range[0], normal_range[1])
                
        elif test_name == "creatinine":
            if "kidney" in condition_lower or "renal" in condition_lower:
                return max(1.3, min(5.0, np.random.normal(2.0, 0.8)))
            else:
                return np.random.uniform(normal_range[0], normal_range[1])
        
        # Default: normal range with slight variation
        range_center = (normal_range[0] + normal_range[1]) / 2
        range_width = normal_range[1] - normal_range[0]
        return np.random.normal(range_center, range_width * 0.2)
    
    def _generate_lab_result_set(self, test_date: datetime, baseline_labs: Dict[str, float],
                               conditions: List[str], panels: List[str], 
                               time_index: int, total_tests: int) -> Dict[str, Any]:
        """Generate a complete set of lab results for a given date."""
        
        # Calculate progression factor (how much time has passed)
        progression = time_index / max(1, total_tests - 1) if total_tests > 1 else 0
        
        results = {
            "test_date": test_date.isoformat(),
            "results": {},
            "abnormal_flags": [],
            "clinical_significance": []
        }
        
        for test_name, baseline_value in baseline_labs.items():
            # Generate temporal variation
            current_value = self._apply_temporal_variation(
                test_name, baseline_value, progression, conditions, time_index
            )
            
            results["results"][test_name] = {
                "value": round(current_value, 2),
                "unit": self.lab_ranges[test_name]["unit"],
                "reference_range": f"{self.lab_ranges[test_name]['normal'][0]}-{self.lab_ranges[test_name]['normal'][1]}",
                "flag": self._determine_abnormal_flag(test_name, current_value)
            }
            
            # Check for abnormal values
            if results["results"][test_name]["flag"] != "Normal":
                results["abnormal_flags"].append({
                    "test": test_name,
                    "value": current_value,
                    "flag": results["results"][test_name]["flag"]
                })
        
        # Add clinical significance notes
        results["clinical_significance"] = self._generate_clinical_significance(
            results["results"], conditions
        )
        
        return results
    
    def _apply_temporal_variation(self, test_name: str, baseline_value: float,
                                progression: float, conditions: List[str],
                                time_index: int) -> float:
        """Apply realistic temporal variation to lab values."""
        
        # Base trend over time
        trend_factor = 1.0
        condition_lower = [c.lower() for c in conditions]
        
        # Condition-specific trends
        if test_name == "hba1c" and any(cond in condition_lower for cond in ["diabetes"]):
            # HbA1c may worsen over time without good control
            trend_factor = 1 + (progression * 0.15 * np.random.uniform(0.5, 1.5))
            
        elif test_name == "creatinine" and any(cond in condition_lower for cond in ["diabetes", "hypertension"]):
            # Kidney function may decline over time
            trend_factor = 1 + (progression * 0.20 * np.random.uniform(0.3, 1.2))
            
        elif test_name in ["bnp", "nt_probnp"] and "heart_failure" in condition_lower:
            # BNP may fluctuate with heart failure management
            trend_factor = 1 + (progression * 0.10 * np.random.uniform(-1.0, 1.5))
        
        # Seasonal variation for some tests
        seasonal_factor = self._get_seasonal_factor(test_name, time_index)
        
        # Random day-to-day variation
        cv = self._get_biological_variation(test_name)  # Coefficient of variation
        random_factor = np.random.normal(1, cv)
        
        # Medication effect simulation (improvement in some cases)
        medication_factor = self._simulate_medication_effect(test_name, conditions, progression)
        
        new_value = baseline_value * trend_factor * seasonal_factor * random_factor * medication_factor
        
        # Ensure values stay within reasonable physiological ranges
        return self._constrain_to_physiological_range(test_name, new_value)
    
    def _get_biological_variation(self, test_name: str) -> float:
        """Get biological coefficient of variation for different tests."""
        cv_values = {
            "glucose_fasting": 0.15,
            "hba1c": 0.03,
            "creatinine": 0.05,
            "bnp": 0.25,
            "total_cholesterol": 0.08,
            "triglycerides": 0.20,
            "sodium": 0.02,
            "potassium": 0.05,
            "hemoglobin": 0.04
        }
        return cv_values.get(test_name, 0.10)  # Default 10% CV
    
    def _get_seasonal_factor(self, test_name: str, time_index: int) -> float:
        """Apply seasonal variation to certain lab values."""
        # Simulate seasonal effects (simplified sine wave)
        if test_name in ["total_cholesterol", "ldl_cholesterol"]:
            # Cholesterol tends to be higher in winter
            seasonal_cycle = np.sin(2 * np.pi * time_index / 12) * 0.05
            return 1 + seasonal_cycle
        
        return 1.0  # No seasonal effect for most tests
    
    def _simulate_medication_effect(self, test_name: str, conditions: List[str], 
                                  progression: float) -> float:
        """Simulate the effect of medications on lab values."""
        condition_lower = [c.lower() for c in conditions]
        
        # Simulate gradual improvement with medication adherence
        if test_name == "hba1c" and any(cond in condition_lower for cond in ["diabetes"]):
            # Assume some patients improve with better medication management
            if np.random.random() < 0.6:  # 60% show improvement
                return 1 - (progression * 0.10)  # Up to 10% improvement
        
        elif test_name in ["total_cholesterol", "ldl_cholesterol"] and any(cond in condition_lower for cond in ["cardiovascular", "diabetes"]):
            # Statin effect
            if np.random.random() < 0.7:  # 70% on statins show improvement
                return 1 - (progression * 0.15)  # Up to 15% reduction
        
        elif test_name in ["bnp", "nt_probnp"] and "heart_failure" in condition_lower:
            # Heart failure medications may improve BNP
            if np.random.random() < 0.5:  # 50% show improvement
                return 1 - (progression * 0.20)  # Up to 20% improvement
        
        return 1.0  # No medication effect
    
    def _constrain_to_physiological_range(self, test_name: str, value: float) -> float:
        """Constrain lab values to physiologically reasonable ranges."""
        constraints = {
            "glucose_fasting": (40, 500),
            "hba1c": (3.0, 20.0),
            "creatinine": (0.1, 15.0),
            "bnp": (1, 5000),
            "nt_probnp": (1, 35000),
            "total_cholesterol": (50, 500),
            "ldl_cholesterol": (10, 400),
            "hdl_cholesterol": (10, 150),
            "triglycerides": (20, 1000),
            "sodium": (120, 160),
            "potassium": (2.0, 7.0),
            "hemoglobin": (5.0, 20.0)
        }
        
        if test_name in constraints:
            min_val, max_val = constraints[test_name]
            return max(min_val, min(max_val, value))
        
        return max(0, value)  # Default: ensure positive values
    
    def _determine_abnormal_flag(self, test_name: str, value: float) -> str:
        """Determine if a lab value is abnormal and assign appropriate flag."""
        if test_name not in self.lab_ranges:
            return "Normal"
        
        normal_range = self.lab_ranges[test_name]["normal"]
        
        if value < normal_range[0] * 0.8:  # Significantly low
            return "Critical Low"
        elif value < normal_range[0]:
            return "Low"
        elif value > normal_range[1] * 1.5:  # Significantly high
            return "Critical High"
        elif value > normal_range[1]:
            return "High"
        else:
            return "Normal"
    
    def _generate_clinical_significance(self, results: Dict[str, Dict], 
                                      conditions: List[str]) -> List[str]:
        """Generate clinical significance notes for abnormal results."""
        significance = []
        
        # Check for critical values
        for test_name, result in results.items():
            if result["flag"] in ["Critical High", "Critical Low"]:
                significance.append(f"Critical {test_name}: {result['value']} {result['unit']} - Immediate attention required")
        
        # Condition-specific significance
        condition_lower = [c.lower() for c in conditions]
        
        if any(cond in condition_lower for cond in ["diabetes"]):
            if "hba1c" in results and results["hba1c"]["value"] > 9.0:
                significance.append("Poor diabetes control - HbA1c >9% indicates high risk for complications")
        
        if "heart_failure" in condition_lower:
            if "bnp" in results and results["bnp"]["value"] > 400:
                significance.append("Elevated BNP suggests worsening heart failure")
        
        return significance
    
    def _analyze_trends(self, lab_results: List[Dict], panels: List[str]) -> Dict[str, Any]:
        """Analyze trends in lab values over time."""
        if len(lab_results) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        trends = {}
        
        # Extract time series for each test
        for test_name in self.lab_ranges.keys():
            values = []
            dates = []
            
            for result in lab_results:
                if test_name in result["results"]:
                    values.append(result["results"][test_name]["value"])
                    dates.append(result["test_date"])
            
            if len(values) >= 2:
                # Calculate trend
                trend_slope = (values[-1] - values[0]) / max(1, len(values) - 1)
                percent_change = ((values[-1] - values[0]) / values[0]) * 100
                
                trends[test_name] = {
                    "trend_direction": "Increasing" if trend_slope > 0 else "Decreasing" if trend_slope < 0 else "Stable",
                    "percent_change": round(percent_change, 1),
                    "current_value": values[-1],
                    "baseline_value": values[0]
                }
        
        return trends
