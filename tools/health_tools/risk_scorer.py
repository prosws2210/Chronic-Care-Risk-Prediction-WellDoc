"""
Risk Scorer Tool for calculating disease-specific risk scores and prognostic indices.
Implements various clinical risk assessment tools for chronic care management.
"""

import json
import logging
import math
from typing import Dict, Any, Optional, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class RiskScorerInput(BaseModel):
    """Input schema for risk scoring."""
    score_type: str = Field(description="Type of risk score to calculate")
    patient_data: Dict[str, Any] = Field(description="Patient data for risk calculation")
    timeframe: Optional[str] = Field(default="1_year", description="Risk assessment timeframe")

class RiskScorerTool(BaseTool):
    """Tool for calculating disease-specific risk scores and prognosis."""
    
    name: str = "Risk Scorer"
    description: str = "Calculates disease-specific risk scores and prognostic indices for chronic care patients"
    args_schema: type[BaseModel] = RiskScorerInput
    
    def __init__(self):
        super().__init__()
        logger.info("RiskScorerTool initialized")
    
    def _run(self, score_type: str, patient_data: Dict[str, Any], 
             timeframe: str = "1_year") -> str:
        """Calculate risk scores based on type."""
        try:
            logger.info(f"Calculating risk score: {score_type}")
            
            score_type_lower = score_type.lower()
            
            if score_type_lower == "diabetes_complications":
                result = self._calculate_diabetes_complications_risk(patient_data, timeframe)
            elif score_type_lower == "heart_failure_mortality":
                result = self._calculate_heart_failure_mortality_risk(patient_data, timeframe)
            elif score_type_lower == "seattle_heart_failure":
                result = self._calculate_seattle_heart_failure_model(patient_data)
            elif score_type_lower == "maggic_heart_failure":
                result = self._calculate_maggic_heart_failure(patient_data)
            elif score_type_lower == "cardiovascular_mortality":
                result = self._calculate_cardiovascular_mortality_risk(patient_data, timeframe)
            elif score_type_lower == "kidney_failure_risk":
                result = self._calculate_kidney_failure_risk(patient_data, timeframe)
            elif score_type_lower == "hospital_readmission":
                result = self._calculate_hospital_readmission_risk(patient_data, timeframe)
            elif score_type_lower == "falls_risk":
                result = self._calculate_falls_risk(patient_data, timeframe)
            elif score_type_lower == "medication_adherence_risk":
                result = self._calculate_medication_adherence_risk(patient_data, timeframe)
            elif score_type_lower == "composite_deterioration":
                result = self._calculate_composite_deterioration_risk(patient_data, timeframe)
            else:
                result = {"error": f"Unknown risk score type: {score_type}"}
            
            logger.info(f"Risk score calculation completed: {score_type}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"Error in risk score calculation: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _calculate_diabetes_complications_risk(self, patient_data: Dict[str, Any], 
                                             timeframe: str) -> Dict[str, Any]:
        """Calculate risk of diabetes complications."""
        try:
            risk_score = 0
            risk_factors = {}
            
            # HbA1c (strongest predictor)
            hba1c = patient_data.get("hba1c", 7.0)
            if hba1c >= 9.0:
                risk_score += 25
                risk_factors["poor_glycemic_control"] = "HbA1c ≥9%"
            elif hba1c >= 8.0:
                risk_score += 15
                risk_factors["suboptimal_glycemic_control"] = "HbA1c 8-8.9%"
            elif hba1c >= 7.0:
                risk_score += 8
                risk_factors["borderline_glycemic_control"] = "HbA1c 7-7.9%"
            
            # Duration of diabetes
            diabetes_duration = patient_data.get("diabetes_duration_years", 5)
            if diabetes_duration >= 15:
                risk_score += 20
                risk_factors["long_diabetes_duration"] = f"{diabetes_duration} years"
            elif diabetes_duration >= 10:
                risk_score += 12
                risk_factors["moderate_diabetes_duration"] = f"{diabetes_duration} years"
            elif diabetes_duration >= 5:
                risk_score += 5
                risk_factors["established_diabetes"] = f"{diabetes_duration} years"
            
            # Blood pressure
            systolic_bp = patient_data.get("systolic_bp", 130)
            if systolic_bp >= 160:
                risk_score += 15
                risk_factors["uncontrolled_hypertension"] = f"SBP {systolic_bp} mmHg"
            elif systolic_bp >= 140:
                risk_score += 8
                risk_factors["elevated_bp"] = f"SBP {systolic_bp} mmHg"
            
            # Kidney function
            egfr = patient_data.get("egfr", 90)
            if egfr < 45:
                risk_score += 20
                risk_factors["moderate_kidney_disease"] = f"eGFR {egfr}"
            elif egfr < 60:
                risk_score += 10
                risk_factors["mild_kidney_disease"] = f"eGFR {egfr}"
            
            # Lipid control
            ldl = patient_data.get("ldl_cholesterol", 100)
            if ldl >= 130:
                risk_score += 10
                risk_factors["elevated_ldl"] = f"LDL {ldl} mg/dL"
            
            # Existing complications
            if patient_data.get("retinopathy", False):
                risk_score += 15
                risk_factors["existing_retinopathy"] = "Present"
            if patient_data.get("neuropathy", False):
                risk_score += 10
                risk_factors["existing_neuropathy"] = "Present"
            if patient_data.get("nephropathy", False):
                risk_score += 15
                risk_factors["existing_nephropathy"] = "Present"
            
            # Smoking
            if patient_data.get("smoker", False):
                risk_score += 12
                risk_factors["smoking"] = "Current smoker"
            
            # Age
            age = patient_data.get("age", 65)
            if age >= 75:
                risk_score += 8
                risk_factors["advanced_age"] = f"{age} years"
            
            # Convert to probability based on timeframe
            if timeframe == "1_year":
                base_rate = 0.15  # 15% annual complication rate
            elif timeframe == "5_year":
                base_rate = 0.50  # 50% 5-year complication rate
            else:
                base_rate = 0.25  # 25% default
            
            # Calculate final risk percentage
            risk_percentage = min(95, base_rate * 100 * (1 + risk_score / 100))
            
            # Risk categorization
            if risk_percentage >= 60:
                risk_level = "Very High"
                interventions = ["Intensive diabetes management", "Ophthalmology referral", "Nephrology consultation"]
            elif risk_percentage >= 40:
                risk_level = "High"
                interventions = ["Optimize glucose control", "Annual screening", "Cardiology evaluation"]
            elif risk_percentage >= 20:
                risk_level = "Moderate"
                interventions = ["Standard diabetes care", "Regular screening", "Lifestyle counseling"]
            else:
                risk_level = "Low"
                interventions = ["Continue current management", "Routine monitoring"]
            
            return {
                "score_type": "Diabetes Complications Risk",
                "timeframe": timeframe,
                "risk_score": risk_score,
                "risk_percentage": round(risk_percentage, 1),
                "risk_level": risk_level,
                "contributing_factors": risk_factors,
                "recommended_interventions": interventions,
                "monitoring_frequency": "Every 3-6 months" if risk_percentage >= 40 else "Every 6-12 months"
            }
            
        except Exception as e:
            return {"error": f"Diabetes complications risk calculation failed: {str(e)}"}
    
    def _calculate_heart_failure_mortality_risk(self, patient_data: Dict[str, Any], 
                                              timeframe: str) -> Dict[str, Any]:
        """Calculate heart failure mortality risk using clinical factors."""
        try:
            risk_score = 0
            risk_factors = {}
            
            # Age (major predictor)
            age = patient_data.get("age", 65)
            if age >= 80:
                risk_score += 20
                risk_factors["advanced_age"] = f"{age} years"
            elif age >= 70:
                risk_score += 12
                risk_factors["elderly"] = f"{age} years"
            elif age >= 60:
                risk_score += 5
                risk_factors["older_adult"] = f"{age} years"
            
            # NYHA Class
            nyha_class = patient_data.get("nyha_class", "II")
            if nyha_class == "IV":
                risk_score += 30
                risk_factors["severe_symptoms"] = "NYHA Class IV"
            elif nyha_class == "III":
                risk_score += 20
                risk_factors["moderate_symptoms"] = "NYHA Class III"
            elif nyha_class == "II":
                risk_score += 8
                risk_factors["mild_symptoms"] = "NYHA Class II"
            
            # Ejection Fraction
            ef = patient_data.get("ejection_fraction", 50)
            if ef < 25:
                risk_score += 25
                risk_factors["severely_reduced_ef"] = f"EF {ef}%"
            elif ef < 35:
                risk_score += 15
                risk_factors["reduced_ef"] = f"EF {ef}%"
            elif ef < 45:
                risk_score += 8
                risk_factors["mildly_reduced_ef"] = f"EF {ef}%"
            
            # BNP/NT-proBNP levels
            bnp = patient_data.get("bnp")
            nt_probnp = patient_data.get("nt_probnp")
            if bnp and bnp > 1000:
                risk_score += 20
                risk_factors["very_elevated_bnp"] = f"BNP {bnp} pg/mL"
            elif bnp and bnp > 400:
                risk_score += 12
                risk_factors["elevated_bnp"] = f"BNP {bnp} pg/mL"
            elif nt_probnp and nt_probnp > 4000:
                risk_score += 20
                risk_factors["very_elevated_nt_probnp"] = f"NT-proBNP {nt_probnp} pg/mL"
            elif nt_probnp and nt_probnp > 1800:
                risk_score += 12
                risk_factors["elevated_nt_probnp"] = f"NT-proBNP {nt_probnp} pg/mL"
            
            # Kidney function
            egfr = patient_data.get("egfr", 90)
            if egfr < 30:
                risk_score += 20
                risk_factors["severe_kidney_disease"] = f"eGFR {egfr}"
            elif egfr < 45:
                risk_score += 12
                risk_factors["moderate_kidney_disease"] = f"eGFR {egfr}"
            elif egfr < 60:
                risk_score += 5
                risk_factors["mild_kidney_disease"] = f"eGFR {egfr}"
            
            # Recent hospitalizations
            recent_hf_hosp = patient_data.get("hf_hospitalizations_6mo", 0)
            if recent_hf_hosp >= 2:
                risk_score += 18
                risk_factors["frequent_hospitalizations"] = f"{recent_hf_hosp} in 6 months"
            elif recent_hf_hosp == 1:
                risk_score += 10
                risk_factors["recent_hospitalization"] = "1 in 6 months"
            
            # Comorbidities
            if patient_data.get("diabetes", False):
                risk_score += 8
                risk_factors["diabetes"] = "Present"
            if patient_data.get("copd", False):
                risk_score += 10
                risk_factors["copd"] = "Present"
            if patient_data.get("atrial_fibrillation", False):
                risk_score += 6
                risk_factors["atrial_fibrillation"] = "Present"
            
            # Medication adherence
            med_adherence = patient_data.get("medication_adherence", "good")
            if med_adherence == "poor":
                risk_score += 15
                risk_factors["poor_med_adherence"] = "Poor adherence"
            elif med_adherence == "fair":
                risk_score += 8
                risk_factors["fair_med_adherence"] = "Fair adherence"
            
            # Convert to mortality probability
            if timeframe == "1_year":
                base_mortality = 0.10  # 10% annual mortality
            elif timeframe == "6_month":
                base_mortality = 0.06  # 6% 6-month mortality
            else:
                base_mortality = 0.15  # 15% default
            
            mortality_risk = min(90, base_mortality * 100 * (1 + risk_score / 100))
            
            # Risk stratification
            if mortality_risk >= 50:
                risk_level = "Very High"
                interventions = ["Palliative care consultation", "Advanced directive discussion", "Intensive management"]
            elif mortality_risk >= 30:
                risk_level = "High"
                interventions = ["Heart failure specialist referral", "Device therapy evaluation", "Close monitoring"]
            elif mortality_risk >= 15:
                risk_level = "Moderate"
                interventions = ["Optimize medical therapy", "Regular follow-up", "Lifestyle counseling"]
            else:
                risk_level = "Low"
                interventions = ["Continue standard care", "Routine monitoring"]
            
            return {
                "score_type": "Heart Failure Mortality Risk",
                "timeframe": timeframe,
                "risk_score": risk_score,
                "mortality_risk_percentage": round(mortality_risk, 1),
                "risk_level": risk_level,
                "contributing_factors": risk_factors,
                "recommended_interventions": interventions,
                "follow_up_frequency": self._get_hf_followup_frequency(risk_level)
            }
            
        except Exception as e:
            return {"error": f"Heart failure mortality risk calculation failed: {str(e)}"}
    
    def _calculate_seattle_heart_failure_model(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Seattle Heart Failure Model score."""
        try:
            # This is a simplified version of the Seattle HF model
            points = 0
            
            # Age
            age = patient_data.get("age", 65)
            points += (age - 50) * 0.7
            
            # Gender (male = higher risk)
            if patient_data.get("gender", "").lower() == "male":
                points += 1.5
            
            # NYHA Class
            nyha_class = patient_data.get("nyha_class", "II")
            nyha_points = {"I": 0, "II": 2, "III": 6, "IV": 8}
            points += nyha_points.get(nyha_class, 2)
            
            # Ejection Fraction
            ef = patient_data.get("ejection_fraction", 35)
            points += (45 - ef) * 0.1
            
            # Systolic BP
            sbp = patient_data.get("systolic_bp", 120)
            points += (140 - sbp) * 0.02
            
            # Medications (protective factors)
            if patient_data.get("ace_inhibitor", False):
                points -= 1.0
            if patient_data.get("beta_blocker", False):
                points -= 1.2
            if patient_data.get("aldosterone_antagonist", False):
                points -= 0.8
            
            # Comorbidities
            if patient_data.get("diabetes", False):
                points += 1.0
            if patient_data.get("copd", False):
                points += 1.2
            
            # Convert points to survival probability
            one_year_survival = max(0.5, min(0.95, 0.88 - (points * 0.03)))
            mortality_risk = (1 - one_year_survival) * 100
            
            return {
                "score_type": "Seattle Heart Failure Model",
                "total_points": round(points, 1),
                "one_year_survival": round(one_year_survival * 100, 1),
                "one_year_mortality_risk": round(mortality_risk, 1),
                "interpretation": self._interpret_seattle_hf_score(mortality_risk)
            }
            
        except Exception as e:
            return {"error": f"Seattle Heart Failure Model calculation failed: {str(e)}"}
    
    def _calculate_hospital_readmission_risk(self, patient_data: Dict[str, Any], 
                                           timeframe: str) -> Dict[str, Any]:
        """Calculate hospital readmission risk."""
        try:
            risk_score = 0
            risk_factors = {}
            
            # Recent discharge
            days_since_discharge = patient_data.get("days_since_last_discharge", 30)
            if days_since_discharge <= 7:
                risk_score += 25
                risk_factors["very_recent_discharge"] = f"{days_since_discharge} days ago"
            elif days_since_discharge <= 30:
                risk_score += 15
                risk_factors["recent_discharge"] = f"{days_since_discharge} days ago"
            
            # Number of previous admissions
            admissions_12mo = patient_data.get("admissions_past_12_months", 0)
            if admissions_12mo >= 3:
                risk_score += 20
                risk_factors["frequent_admissions"] = f"{admissions_12mo} admissions"
            elif admissions_12mo >= 2:
                risk_score += 12
                risk_factors["multiple_admissions"] = f"{admissions_12mo} admissions"
            elif admissions_12mo == 1:
                risk_score += 5
                risk_factors["prior_admission"] = "1 admission"
            
            # Age
            age = patient_data.get("age", 65)
            if age >= 80:
                risk_score += 12
                risk_factors["advanced_age"] = f"{age} years"
            elif age >= 65:
                risk_score += 6
                risk_factors["elderly"] = f"{age} years"
            
            # Comorbidity burden
            comorbidities = patient_data.get("comorbidity_count", 2)
            if comorbidities >= 5:
                risk_score += 15
                risk_factors["high_comorbidity_burden"] = f"{comorbidities} conditions"
            elif comorbidities >= 3:
                risk_score += 8
                risk_factors["moderate_comorbidity_burden"] = f"{comorbidities} conditions"
            
            # Specific high-risk conditions
            if patient_data.get("heart_failure", False):
                risk_score += 10
                risk_factors["heart_failure"] = "Present"
            if patient_data.get("copd", False):
                risk_score += 8
                risk_factors["copd"] = "Present"
            if patient_data.get("diabetes", False):
                risk_score += 5
                risk_factors["diabetes"] = "Present"
            
            # Social factors
            if patient_data.get("lives_alone", False):
                risk_score += 8
                risk_factors["social_isolation"] = "Lives alone"
            if patient_data.get("transportation_issues", False):
                risk_score += 6
                risk_factors["transportation_barriers"] = "Limited access"
            if patient_data.get("medication_adherence", "good") == "poor":
                risk_score += 10
                risk_factors["poor_adherence"] = "Medication non-adherence"
            
            # Insurance/access issues
            if patient_data.get("insurance_type", "Commercial") == "Medicaid":
                risk_score += 5
                risk_factors["insurance_barriers"] = "Medicaid"
            elif patient_data.get("insurance_type", "Commercial") == "Uninsured":
                risk_score += 12
                risk_factors["uninsured"] = "No insurance"
            
            # Calculate readmission probability
            if timeframe == "30_day":
                base_rate = 0.18  # 18% 30-day readmission rate
            elif timeframe == "90_day":
                base_rate = 0.35  # 35% 90-day readmission rate
            else:
                base_rate = 0.25  # 25% default
            
            readmission_risk = min(85, base_rate * 100 * (1 + risk_score / 100))
            
            # Risk stratification
            if readmission_risk >= 60:
                risk_level = "Very High"
                interventions = ["Intensive case management", "Home health services", "Daily contact"]
            elif readmission_risk >= 40:
                risk_level = "High"
                interventions = ["Care transitions program", "Medication reconciliation", "48-hour follow-up"]
            elif readmission_risk >= 25:
                risk_level = "Moderate"
                interventions = ["Standard discharge planning", "7-day follow-up", "Patient education"]
            else:
                risk_level = "Low"
                interventions = ["Routine discharge", "Standard follow-up"]
            
            return {
                "score_type": "Hospital Readmission Risk",
                "timeframe": timeframe,
                "risk_score": risk_score,
                "readmission_risk_percentage": round(readmission_risk, 1),
                "risk_level": risk_level,
                "contributing_factors": risk_factors,
                "recommended_interventions": interventions
            }
            
        except Exception as e:
            return {"error": f"Hospital readmission risk calculation failed: {str(e)}"}
    
    def _calculate_composite_deterioration_risk(self, patient_data: Dict[str, Any], 
                                              timeframe: str) -> Dict[str, Any]:
        """Calculate composite risk of clinical deterioration."""
        try:
            # This combines multiple risk factors for overall deterioration
            total_risk = 0
            risk_components = {}
            
            # Get individual risk scores
            diabetes_risk = self._calculate_diabetes_complications_risk(patient_data, timeframe)
            if "risk_percentage" in diabetes_risk:
                diabetes_weight = 0.3
                total_risk += diabetes_risk["risk_percentage"] * diabetes_weight
                risk_components["diabetes_complications"] = diabetes_risk["risk_percentage"]
            
            hf_risk = self._calculate_heart_failure_mortality_risk(patient_data, timeframe)
            if "mortality_risk_percentage" in hf_risk:
                hf_weight = 0.25
                total_risk += hf_risk["mortality_risk_percentage"] * hf_weight
                risk_components["heart_failure_mortality"] = hf_risk["mortality_risk_percentage"]
            
            readmission_risk = self._calculate_hospital_readmission_risk(patient_data, timeframe)
            if "readmission_risk_percentage" in readmission_risk:
                readmission_weight = 0.2
                total_risk += readmission_risk["readmission_risk_percentage"] * readmission_weight
                risk_components["hospital_readmission"] = readmission_risk["readmission_risk_percentage"]
            
            # Additional factors
            functional_decline_risk = self._assess_functional_decline_risk(patient_data)
            medication_risk = self._assess_medication_related_risk(patient_data)
            
            total_risk += functional_decline_risk * 0.15
            total_risk += medication_risk * 0.1
            
            risk_components["functional_decline"] = functional_decline_risk
            risk_components["medication_related"] = medication_risk
            
            # Overall risk level
            if total_risk >= 70:
                overall_risk_level = "Critical"
                priority = "Immediate intervention required"
            elif total_risk >= 50:
                overall_risk_level = "High"
                priority = "Urgent care coordination needed"
            elif total_risk >= 30:
                overall_risk_level = "Moderate"
                priority = "Enhanced monitoring recommended"
            else:
                overall_risk_level = "Low"
                priority = "Standard care appropriate"
            
            return {
                "score_type": "Composite Deterioration Risk",
                "timeframe": timeframe,
                "overall_risk_percentage": round(total_risk, 1),
                "overall_risk_level": overall_risk_level,
                "priority_level": priority,
                "risk_components": risk_components,
                "top_risk_factors": self._identify_top_composite_risks(risk_components),
                "comprehensive_interventions": self._get_composite_interventions(overall_risk_level)
            }
            
        except Exception as e:
            return {"error": f"Composite deterioration risk calculation failed: {str(e)}"}
    
    def _assess_functional_decline_risk(self, patient_data: Dict[str, Any]) -> float:
        """Assess risk of functional decline."""
        risk = 0
        age = patient_data.get("age", 65)
        if age >= 80: risk += 20
        elif age >= 70: risk += 10
        
        if patient_data.get("falls_history", False): risk += 15
        if patient_data.get("mobility_issues", False): risk += 12
        if patient_data.get("cognitive_impairment", False): risk += 18
        
        return min(risk, 50)  # Cap at 50%
    
    def _assess_medication_related_risk(self, patient_data: Dict[str, Any]) -> float:
        """Assess medication-related risks."""
        risk = 0
        
        med_count = patient_data.get("medication_count", 5)
        if med_count >= 10: risk += 15
        elif med_count >= 7: risk += 8
        elif med_count >= 5: risk += 3
        
        if patient_data.get("medication_adherence", "good") == "poor": risk += 20
        if patient_data.get("drug_interactions", False): risk += 10
        if patient_data.get("kidney_disease", False): risk += 8
        
        return min(risk, 45)  # Cap at 45%
    
    def _identify_top_composite_risks(self, risk_components: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify top risk factors from composite assessment."""
        sorted_risks = sorted(risk_components.items(), key=lambda x: x[1], reverse=True)
        
        top_risks = []
        for risk_type, risk_value in sorted_risks[:3]:  # Top 3 risks
            top_risks.append({
                "risk_factor": risk_type.replace("_", " ").title(),
                "risk_percentage": round(risk_value, 1),
                "priority": "High" if risk_value >= 40 else "Medium" if risk_value >= 25 else "Low"
            })
        
        return top_risks
    
    def _get_composite_interventions(self, risk_level: str) -> List[str]:
        """Get comprehensive interventions based on composite risk."""
        if risk_level == "Critical":
            return [
                "Immediate clinical review within 24 hours",
                "Intensive case management activation",
                "Home health services initiation",
                "Emergency care plan development",
                "Family/caregiver involvement",
                "Daily medication reconciliation"
            ]
        elif risk_level == "High":
            return [
                "Clinical review within 48-72 hours",
                "Care coordination team involvement",
                "Medication therapy management",
                "Social services referral",
                "Remote monitoring consideration"
            ]
        elif risk_level == "Moderate":
            return [
                "Enhanced follow-up scheduling",
                "Patient education reinforcement",
                "Caregiver support assessment",
                "Preventive care optimization"
            ]
        else:
            return [
                "Continue standard care protocols",
                "Routine monitoring maintenance",
                "Preventive care adherence"
            ]
    
    def _get_hf_followup_frequency(self, risk_level: str) -> str:
        """Get heart failure follow-up frequency based on risk."""
        if risk_level == "Very High":
            return "Weekly or twice weekly"
        elif risk_level == "High":
            return "Every 1-2 weeks"
        elif risk_level == "Moderate":
            return "Every 2-4 weeks"
        else:
            return "Every 1-3 months"
    
    def _interpret_seattle_hf_score(self, mortality_risk: float) -> str:
        """Interpret Seattle Heart Failure Model results."""
        if mortality_risk >= 30:
            return "Very high risk - consider advanced heart failure therapies"
        elif mortality_risk >= 20:
            return "High risk - optimize medical therapy and close monitoring"
        elif mortality_risk >= 10:
            return "Moderate risk - standard heart failure management"
        else:
            return "Lower risk - continue current management with routine follow-up"