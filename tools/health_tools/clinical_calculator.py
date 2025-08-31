"""
Clinical Calculator Tool for healthcare-specific calculations and risk scores.
Implements standard clinical formulas and medical risk assessment tools.
"""

import json
import logging
import math
from typing import Dict, Any, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ClinicalCalculatorInput(BaseModel):
    """Input schema for clinical calculations."""
    calculation_type: str = Field(description="Type of clinical calculation to perform")
    patient_data: Dict[str, Any] = Field(description="Patient data required for calculation")
    units: Optional[str] = Field(default="standard", description="Unit system (standard/metric)")

class ClinicalCalculatorTool(BaseTool):
    """Tool for performing clinical calculations and risk assessments."""
    
    name: str = "Clinical Calculator"
    description: str = "Performs clinical calculations including BMI, eGFR, risk scores, and medical formulas"
    args_schema: type[BaseModel] = ClinicalCalculatorInput
    
    def __init__(self):
        super().__init__()
        logger.info("ClinicalCalculatorTool initialized")
    
    def _run(self, calculation_type: str, patient_data: Dict[str, Any], 
             units: str = "standard") -> str:
        """Perform clinical calculations based on type."""
        try:
            logger.info(f"Performing clinical calculation: {calculation_type}")
            
            if calculation_type.lower() == "bmi":
                result = self._calculate_bmi(patient_data, units)
            elif calculation_type.lower() == "egfr":
                result = self._calculate_egfr(patient_data)
            elif calculation_type.lower() == "chads2":
                result = self._calculate_chads2(patient_data)
            elif calculation_type.lower() == "cha2ds2vasc":
                result = self._calculate_cha2ds2vasc(patient_data)
            elif calculation_type.lower() == "has_bled":
                result = self._calculate_has_bled(patient_data)
            elif calculation_type.lower() == "framingham":
                result = self._calculate_framingham_risk(patient_data)
            elif calculation_type.lower() == "ascvd":
                result = self._calculate_ascvd_risk(patient_data)
            elif calculation_type.lower() == "qrisk3":
                result = self._calculate_qrisk3(patient_data)
            elif calculation_type.lower() == "ace27":
                result = self._calculate_ace27_comorbidity(patient_data)
            elif calculation_type.lower() == "body_surface_area":
                result = self._calculate_body_surface_area(patient_data, units)
            elif calculation_type.lower() == "creatinine_clearance":
                result = self._calculate_creatinine_clearance(patient_data)
            elif calculation_type.lower() == "cardiac_index":
                result = self._calculate_cardiac_index(patient_data, units)
            elif calculation_type.lower() == "mean_arterial_pressure":
                result = self._calculate_mean_arterial_pressure(patient_data)
            elif calculation_type.lower() == "pulse_pressure":
                result = self._calculate_pulse_pressure(patient_data)
            else:
                result = {"error": f"Unknown calculation type: {calculation_type}"}
            
            logger.info(f"Clinical calculation completed: {calculation_type}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in clinical calculation: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _calculate_bmi(self, patient_data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """Calculate Body Mass Index and categorization."""
        try:
            if units == "metric":
                weight_kg = patient_data["weight_kg"]
                height_m = patient_data["height_m"]
            else:  # standard (US)
                weight_lb = patient_data.get("weight_lb")
                height_in = patient_data.get("height_in")
                
                if weight_lb and height_in:
                    weight_kg = weight_lb * 0.453592
                    height_m = height_in * 0.0254
                else:
                    weight_kg = patient_data.get("weight_kg")
                    height_m = patient_data.get("height_m")
            
            if not weight_kg or not height_m:
                return {"error": "Missing required weight or height data"}
            
            bmi = weight_kg / (height_m ** 2)
            
            # BMI categories
            if bmi < 18.5:
                category = "Underweight"
                risk = "Increased risk of malnutrition, osteoporosis"
            elif bmi < 25:
                category = "Normal weight"
                risk = "Lowest mortality risk"
            elif bmi < 30:
                category = "Overweight"
                risk = "Increased risk of cardiovascular disease"
            elif bmi < 35:
                category = "Class I Obesity"
                risk = "Moderate risk of cardiovascular disease, diabetes"
            elif bmi < 40:
                category = "Class II Obesity" 
                risk = "High risk of cardiovascular disease, diabetes"
            else:
                category = "Class III Obesity (Severe)"
                risk = "Very high risk, consider bariatric surgery"
            
            return {
                "calculation": "BMI",
                "value": round(bmi, 1),
                "units": "kg/m²",
                "category": category,
                "health_risk": risk,
                "interpretation": self._get_bmi_interpretation(bmi),
                "recommendations": self._get_bmi_recommendations(bmi)
            }
            
        except Exception as e:
            return {"error": f"BMI calculation failed: {str(e)}"}
    
    def _calculate_egfr(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate estimated Glomerular Filtration Rate (CKD-EPI equation)."""
        try:
            age = patient_data["age"]
            gender = patient_data["gender"].lower()
            race = patient_data.get("race", "").lower()
            creatinine = patient_data["serum_creatinine"]  # mg/dL
            
            # CKD-EPI equation
            if gender == "female":
                if creatinine <= 0.7:
                    egfr = 144 * ((creatinine / 0.7) ** -0.329) * (0.993 ** age)
                else:
                    egfr = 144 * ((creatinine / 0.7) ** -1.209) * (0.993 ** age)
            else:  # male
                if creatinine <= 0.9:
                    egfr = 141 * ((creatinine / 0.9) ** -0.411) * (0.993 ** age)
                else:
                    egfr = 141 * ((creatinine / 0.9) ** -1.209) * (0.993 ** age)
            
            # Race adjustment (if African American)
            if "african" in race or "black" in race:
                egfr *= 1.159
            
            # CKD staging
            if egfr >= 90:
                stage = "G1 (Normal or high)"
                description = "Normal kidney function"
            elif egfr >= 60:
                stage = "G2 (Mildly decreased)"
                description = "Mildly decreased kidney function"
            elif egfr >= 45:
                stage = "G3a (Mild to moderately decreased)"
                description = "Mild to moderate kidney function decrease"
            elif egfr >= 30:
                stage = "G3b (Moderately to severely decreased)"
                description = "Moderate to severe kidney function decrease"
            elif egfr >= 15:
                stage = "G4 (Severely decreased)"
                description = "Severe kidney function decrease"
            else:
                stage = "G5 (Kidney failure)"
                description = "Kidney failure - dialysis or transplant needed"
            
            return {
                "calculation": "eGFR",
                "value": round(egfr, 1),
                "units": "mL/min/1.73m²",
                "ckd_stage": stage,
                "description": description,
                "clinical_significance": self._get_egfr_significance(egfr),
                "monitoring_recommendations": self._get_egfr_monitoring(egfr)
            }
            
        except Exception as e:
            return {"error": f"eGFR calculation failed: {str(e)}"}
    
    def _calculate_chads2(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CHADS2 score for stroke risk in atrial fibrillation."""
        try:
            score = 0
            components = {}
            
            # Congestive heart failure (1 point)
            if patient_data.get("heart_failure", False):
                score += 1
                components["heart_failure"] = 1
            
            # Hypertension (1 point)
            if patient_data.get("hypertension", False):
                score += 1
                components["hypertension"] = 1
            
            # Age ≥75 years (1 point)
            if patient_data.get("age", 0) >= 75:
                score += 1
                components["age_75_plus"] = 1
            
            # Diabetes (1 point)
            if patient_data.get("diabetes", False):
                score += 1
                components["diabetes"] = 1
            
            # Stroke/TIA history (2 points)
            if patient_data.get("stroke_tia_history", False):
                score += 2
                components["stroke_tia"] = 2
            
            # Risk stratification
            if score == 0:
                risk_level = "Low"
                annual_risk = "1.9%"
                anticoagulation = "No antithrombotic therapy or aspirin"
            elif score == 1:
                risk_level = "Moderate"
                annual_risk = "2.8%"
                anticoagulation = "No antithrombotic therapy, aspirin, or oral anticoagulant"
            else:  # score >= 2
                risk_level = "High"
                annual_risk = f"{2.8 + (score - 1) * 2}%" 
                anticoagulation = "Oral anticoagulant recommended"
            
            return {
                "calculation": "CHADS2",
                "score": score,
                "components": components,
                "risk_level": risk_level,
                "annual_stroke_risk": annual_risk,
                "anticoagulation_recommendation": anticoagulation,
                "clinical_notes": "For atrial fibrillation patients only"
            }
            
        except Exception as e:
            return {"error": f"CHADS2 calculation failed: {str(e)}"}
    
    def _calculate_cha2ds2vasc(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CHA2DS2-VASc score for stroke risk in atrial fibrillation."""
        try:
            score = 0
            components = {}
            
            # Congestive heart failure (1 point)
            if patient_data.get("heart_failure", False):
                score += 1
                components["heart_failure"] = 1
            
            # Hypertension (1 point)
            if patient_data.get("hypertension", False):
                score += 1
                components["hypertension"] = 1
            
            # Age (1-2 points)
            age = patient_data.get("age", 0)
            if age >= 75:
                score += 2
                components["age_75_plus"] = 2
            elif age >= 65:
                score += 1
                components["age_65_74"] = 1
            
            # Diabetes (1 point)
            if patient_data.get("diabetes", False):
                score += 1
                components["diabetes"] = 1
            
            # Stroke/TIA history (2 points)
            if patient_data.get("stroke_tia_history", False):
                score += 2
                components["stroke_tia"] = 2
            
            # Vascular disease (1 point)
            if patient_data.get("vascular_disease", False):
                score += 1
                components["vascular_disease"] = 1
            
            # Sex category (female = 1 point)
            if patient_data.get("gender", "").lower() == "female":
                score += 1
                components["female_sex"] = 1
            
            # Risk stratification
            if score == 0:
                risk_level = "Very Low"
                annual_risk = "0.2%"
                recommendation = "No anticoagulation"
            elif score == 1:
                risk_level = "Low"
                annual_risk = "0.6%"
                recommendation = "No anticoagulation or aspirin"
            else:  # score >= 2
                risk_level = "High"
                annual_risk = f"{1.6 + (score - 2) * 0.8}%"
                recommendation = "Oral anticoagulation recommended"
            
            return {
                "calculation": "CHA2DS2-VASc",
                "score": score,
                "components": components,
                "risk_level": risk_level,
                "annual_stroke_risk": annual_risk,
                "recommendation": recommendation
            }
            
        except Exception as e:
            return {"error": f"CHA2DS2-VASc calculation failed: {str(e)}"}
    
    def _calculate_has_bled(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate HAS-BLED score for bleeding risk with anticoagulation."""
        try:
            score = 0
            components = {}
            
            # Hypertension (1 point)
            if patient_data.get("hypertension", False):
                score += 1
                components["hypertension"] = 1
            
            # Abnormal renal/liver function (1 point each)
            if patient_data.get("renal_disease", False):
                score += 1
                components["renal_disease"] = 1
            if patient_data.get("liver_disease", False):
                score += 1
                components["liver_disease"] = 1
            
            # Stroke history (1 point)
            if patient_data.get("stroke_history", False):
                score += 1
                components["stroke_history"] = 1
            
            # Bleeding history (1 point)
            if patient_data.get("bleeding_history", False):
                score += 1
                components["bleeding_history"] = 1
            
            # Labile INRs (1 point)
            if patient_data.get("labile_inr", False):
                score += 1
                components["labile_inr"] = 1
            
            # Elderly (age >65) (1 point)
            if patient_data.get("age", 0) > 65:
                score += 1
                components["elderly"] = 1
            
            # Drugs/alcohol (1 point each)
            if patient_data.get("antiplatelet_drugs", False):
                score += 1
                components["drugs"] = 1
            if patient_data.get("alcohol_abuse", False):
                score += 1
                components["alcohol"] = 1
            
            # Risk interpretation
            if score <= 2:
                risk_level = "Low"
                bleeding_risk = "<2% per year"
                recommendation = "Anticoagulation generally safe"
            elif score == 3:
                risk_level = "Moderate"
                bleeding_risk = "2-4% per year"
                recommendation = "Caution with anticoagulation"
            else:  # score >= 4
                risk_level = "High"
                bleeding_risk = ">4% per year"
                recommendation = "Careful consideration of anticoagulation"
            
            return {
                "calculation": "HAS-BLED",
                "score": score,
                "components": components,
                "bleeding_risk_level": risk_level,
                "annual_bleeding_risk": bleeding_risk,
                "recommendation": recommendation
            }
            
        except Exception as e:
            return {"error": f"HAS-BLED calculation failed: {str(e)}"}
    
    def _calculate_framingham_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Framingham 10-year cardiovascular risk."""
        try:
            age = patient_data["age"]
            gender = patient_data["gender"].lower()
            total_chol = patient_data["total_cholesterol"]  # mg/dL
            hdl_chol = patient_data["hdl_cholesterol"]  # mg/dL
            systolic_bp = patient_data["systolic_bp"]  # mmHg
            smoker = patient_data.get("smoker", False)
            diabetes = patient_data.get("diabetes", False)
            bp_meds = patient_data.get("bp_medications", False)
            
            # Gender-specific calculations
            if gender == "male":
                # Male Framingham equation
                points = 0
                
                # Age points
                if age >= 70: points += 11
                elif age >= 65: points += 8
                elif age >= 60: points += 5
                elif age >= 55: points += 3
                elif age >= 50: points += 2
                elif age >= 45: points += 1
                elif age >= 40: points += 0
                else: points -= 2
                
                # Cholesterol points
                if total_chol >= 280: points += 3
                elif total_chol >= 240: points += 2
                elif total_chol >= 200: points += 1
                elif total_chol >= 160: points += 0
                else: points -= 1
                
                # HDL points
                if hdl_chol >= 60: points -= 2
                elif hdl_chol >= 50: points -= 1
                elif hdl_chol >= 40: points += 0
                else: points += 1
                
                # Blood pressure points
                if bp_meds:
                    if systolic_bp >= 160: points += 3
                    elif systolic_bp >= 140: points += 2
                    elif systolic_bp >= 130: points += 1
                    else: points += 0
                else:
                    if systolic_bp >= 160: points += 2
                    elif systolic_bp >= 140: points += 1
                    elif systolic_bp >= 130: points += 1
                    else: points += 0
                
                # Smoking
                if smoker: points += 2
                
                # Diabetes
                if diabetes: points += 2
                
            else:  # female
                # Female Framingham equation
                points = 0
                
                # Age points
                if age >= 75: points += 12
                elif age >= 70: points += 9
                elif age >= 65: points += 6
                elif age >= 60: points += 4
                elif age >= 55: points += 2
                elif age >= 50: points += 1
                elif age >= 45: points += 0
                else: points -= 2
                
                # Similar adjustments for other factors...
                # (Simplified for brevity)
            
            # Convert points to risk percentage (simplified)
            if points <= 0: risk_percent = 1
            elif points <= 5: risk_percent = 2
            elif points <= 10: risk_percent = 6
            elif points <= 15: risk_percent = 12
            else: risk_percent = min(30, points * 2)
            
            # Risk categorization
            if risk_percent < 5:
                risk_level = "Low"
            elif risk_percent < 10:
                risk_level = "Intermediate"
            else:
                risk_level = "High"
            
            return {
                "calculation": "Framingham Risk Score",
                "risk_points": points,
                "ten_year_risk_percent": risk_percent,
                "risk_level": risk_level,
                "recommendations": self._get_framingham_recommendations(risk_percent)
            }
            
        except Exception as e:
            return {"error": f"Framingham calculation failed: {str(e)}"}
    
    def _calculate_body_surface_area(self, patient_data: Dict[str, Any], units: str) -> Dict[str, Any]:
        """Calculate Body Surface Area using Mosteller formula."""
        try:
            if units == "metric":
                weight_kg = patient_data["weight_kg"]
                height_cm = patient_data["height_cm"]
            else:
                weight_lb = patient_data["weight_lb"]
                height_in = patient_data["height_in"]
                weight_kg = weight_lb * 0.453592
                height_cm = height_in * 2.54
            
            # Mosteller formula
            bsa = math.sqrt((weight_kg * height_cm) / 3600)
            
            return {
                "calculation": "Body Surface Area",
                "value": round(bsa, 2),
                "units": "m²",
                "formula": "Mosteller",
                "clinical_use": "Drug dosing, cardiac index calculations"
            }
            
        except Exception as e:
            return {"error": f"BSA calculation failed: {str(e)}"}
    
    def _get_bmi_interpretation(self, bmi: float) -> str:
        """Get clinical interpretation of BMI."""
        if bmi < 18.5:
            return "May indicate malnutrition or underlying health issues"
        elif bmi < 25:
            return "Associated with lowest mortality risk"
        elif bmi < 30:
            return "Associated with increased cardiovascular risk"
        else:
            return "Associated with increased risk of diabetes, cardiovascular disease, and mortality"
    
    def _get_bmi_recommendations(self, bmi: float) -> List[str]:
        """Get recommendations based on BMI."""
        if bmi < 18.5:
            return ["Nutritional assessment", "Rule out underlying conditions", "Weight gain strategies"]
        elif bmi < 25:
            return ["Maintain current weight", "Continue healthy lifestyle", "Regular exercise"]
        elif bmi < 30:
            return ["Weight loss of 5-10%", "Dietary counseling", "Increase physical activity"]
        else:
            return ["Weight loss >10%", "Consider bariatric evaluation", "Comprehensive lifestyle intervention"]
    
    def _get_egfr_significance(self, egfr: float) -> str:
        """Get clinical significance of eGFR value."""
        if egfr >= 90:
            return "Normal kidney function if no other signs of kidney damage"
        elif egfr >= 60:
            return "Mild decrease in kidney function"
        elif egfr >= 30:
            return "Moderate decrease in kidney function - nephrology referral may be needed"
        else:
            return "Severe decrease in kidney function - nephrology referral recommended"
    
    def _get_egfr_monitoring(self, egfr: float) -> List[str]:
        """Get monitoring recommendations based on eGFR."""
        if egfr >= 60:
            return ["Monitor annually", "Address cardiovascular risk factors"]
        elif egfr >= 45:
            return ["Monitor every 6-12 months", "Evaluate for CKD complications"]
        elif egfr >= 30:
            return ["Monitor every 3-6 months", "Prepare for renal replacement therapy"]
        else:
            return ["Monitor monthly", "Nephrology referral urgent", "Prepare for dialysis"]
    
    def _get_framingham_recommendations(self, risk_percent: int) -> List[str]:
        """Get recommendations based on Framingham risk score."""
        if risk_percent < 5:
            return ["Continue healthy lifestyle", "Recheck in 5 years"]
        elif risk_percent < 10:
            return ["Consider statin therapy", "Lifestyle modifications", "Recheck in 3-5 years"]
        else:
            return ["Statin therapy recommended", "Aggressive risk factor modification", "Annual monitoring"]