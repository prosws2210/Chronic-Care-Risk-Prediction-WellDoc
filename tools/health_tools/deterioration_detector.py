"""
Deterioration Detector Tool for identifying early signs of clinical deterioration.
Analyzes trends and patterns in patient data to detect worsening conditions.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DeteriorationDetectorInput(BaseModel):
    """Input schema for deterioration detection."""
    patient_id: str = Field(description="Patient identifier")
    time_series_data: Dict[str, Any] = Field(description="Time series patient data")
    lookback_days: Optional[int] = Field(default=30, description="Days to look back for trend analysis")
    alert_threshold: Optional[str] = Field(default="moderate", description="Alert sensitivity (low/moderate/high)")

class DeteriorationDetectorTool(BaseTool):
    """Tool for detecting clinical deterioration patterns in patient data."""
    
    name: str = "Deterioration Detector"
    description: str = "Analyzes patient data trends to detect early signs of clinical deterioration"
    args_schema: type[BaseModel] = DeteriorationDetectorInput
    
    def __init__(self):
        super().__init__()
        
        # Define normal ranges and deterioration thresholds
        self.normal_ranges = {
            "systolic_bp": {"min": 90, "max": 140, "critical_high": 180, "critical_low": 70},
            "diastolic_bp": {"min": 60, "max": 90, "critical_high": 110, "critical_low": 40},
            "heart_rate": {"min": 60, "max": 100, "critical_high": 120, "critical_low": 50},
            "temperature": {"min": 97.0, "max": 99.5, "critical_high": 101.5, "critical_low": 95.0},
            "oxygen_saturation": {"min": 95, "max": 100, "critical_low": 88},
            "hba1c": {"min": 4.0, "max": 7.0, "critical_high": 9.0},
            "glucose": {"min": 70, "max": 140, "critical_high": 250, "critical_low": 50},
            "creatinine": {"min": 0.6, "max": 1.3, "critical_high": 2.5},
            "bnp": {"min": 0, "max": 100, "critical_high": 400},
            "weight": {"daily_change_threshold": 2.0}  # kg
        }
        
        logger.info("DeteriorationDetectorTool initialized")
    
    def _run(self, patient_id: str, time_series_data: Dict[str, Any], 
             lookback_days: int = 30, alert_threshold: str = "moderate") -> str:
        """Detect deterioration patterns in patient data."""
        try:
            logger.info(f"Analyzing deterioration patterns for patient {patient_id}")
            
            # Analyze different types of deterioration
            vital_signs_analysis = self._analyze_vital_signs_deterioration(
                time_series_data.get("vitals", []), lookback_days, alert_threshold
            )
            
            lab_values_analysis = self._analyze_lab_values_deterioration(
                time_series_data.get("labs", []), lookback_days, alert_threshold
            )
            
            symptom_analysis = self._analyze_symptom_deterioration(
                time_series_data.get("symptoms", []), lookback_days
            )
            
            medication_analysis = self._analyze_medication_adherence_deterioration(
                time_series_data.get("medications", []), lookback_days
            )
            
            functional_analysis = self._analyze_functional_deterioration(
                time_series_data.get("functional_status", []), lookback_days
            )
            
            # Combine analyses for overall deterioration risk
            overall_assessment = self._calculate_overall_deterioration_risk(
                vital_signs_analysis, lab_values_analysis, symptom_analysis,
                medication_analysis, functional_analysis, alert_threshold
            )
            
            # Generate alerts and recommendations
            alerts = self._generate_deterioration_alerts(overall_assessment, alert_threshold)
            
            result = {
                "patient_id": patient_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "lookback_period_days": lookback_days,
                "alert_threshold": alert_threshold,
                "deterioration_analysis": {
                    "vital_signs": vital_signs_analysis,
                    "laboratory_values": lab_values_analysis,
                    "symptoms": symptom_analysis,
                    "medication_adherence": medication_analysis,
                    "functional_status": functional_analysis
                },
                "overall_assessment": overall_assessment,
                "alerts": alerts,
                "recommendations": self._generate_recommendations(overall_assessment, alerts)
            }
            
            logger.info(f"Deterioration analysis completed for patient {patient_id}")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in deterioration detection: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _analyze_vital_signs_deterioration(self, vitals_data: List[Dict[str, Any]], 
                                         lookback_days: int, alert_threshold: str) -> Dict[str, Any]:
        """Analyze vital signs for deterioration patterns."""
        if not vitals_data:
            return {"status": "insufficient_data", "message": "No vital signs data available"}
        
        analysis = {
            "data_points": len(vitals_data),
            "time_span_days": lookback_days,
            "deterioration_indicators": {},
            "trend_analysis": {},
            "critical_values": [],
            "risk_level": "low"
        }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(vitals_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Analyze each vital sign
        vital_signs = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'oxygen_saturation']
        
        for vital in vital_signs:
            if vital in df.columns:
                vital_analysis = self._analyze_single_vital_trend(df, vital, alert_threshold)
                analysis["trend_analysis"][vital] = vital_analysis
                
                # Check for deterioration indicators
                if vital_analysis["trend"] == "worsening":
                    analysis["deterioration_indicators"][vital] = {
                        "severity": vital_analysis["severity"],
                        "pattern": vital_analysis["pattern"],
                        "change_rate": vital_analysis["change_rate"]
                    }
                
                # Check for critical values
                critical_values = self._identify_critical_values(df, vital)
                if critical_values:
                    analysis["critical_values"].extend(critical_values)
        
        # Determine overall vital signs risk level
        if analysis["critical_values"]:
            analysis["risk_level"] = "critical"
        elif len(analysis["deterioration_indicators"]) >= 2:
            analysis["risk_level"] = "high"
        elif analysis["deterioration_indicators"]:
            analysis["risk_level"] = "moderate"
        
        return analysis
    
    def _analyze_single_vital_trend(self, df: pd.DataFrame, vital: str, 
                                   alert_threshold: str) -> Dict[str, Any]:
        """Analyze trend for a single vital sign."""
        values = df[vital].dropna()
        if len(values) < 3:
            return {"status": "insufficient_data"}
        
        # Calculate trend
        timestamps = pd.to_numeric(df['timestamp'].dropna())
        if len(timestamps) == len(values):
            slope, _ = np.polyfit(timestamps, values, 1)
        else:
            slope = 0
        
        # Determine normal range
        normal_range = self.normal_ranges.get(vital, {})
        
        # Analyze pattern
        recent_values = values.tail(5)  # Last 5 measurements
        baseline_values = values.head(5)  # First 5 measurements
        
        recent_mean = recent_values.mean()
        baseline_mean = baseline_values.mean()
        change_rate = ((recent_mean - baseline_mean) / baseline_mean * 100) if baseline_mean != 0 else 0
        
        # Determine trend direction and severity
        if abs(change_rate) < 5:
            trend = "stable"
            severity = "none"
        elif vital in ["systolic_bp", "diastolic_bp", "heart_rate", "temperature"]:
            if change_rate > 10:
                trend = "worsening"
                severity = "moderate" if change_rate < 20 else "severe"
            elif change_rate < -10:
                trend = "improving" if vital in ["systolic_bp", "diastolic_bp"] else "worsening"
                severity = "moderate" if abs(change_rate) < 20 else "severe"
            else:
                trend = "stable"
                severity = "mild"
        else:  # oxygen_saturation
            if change_rate < -5:
                trend = "worsening"
                severity = "moderate" if change_rate > -10 else "severe"
            else:
                trend = "stable"
                severity = "none"
        
        # Pattern recognition
        if len(values) >= 7:
            pattern = self._identify_pattern(values.tail(7))
        else:
            pattern = "insufficient_data"
        
        return {
            "trend": trend,
            "severity": severity,
            "pattern": pattern,
            "change_rate": round(change_rate, 2),
            "recent_mean": round(recent_mean, 2),
            "baseline_mean": round(baseline_mean, 2),
            "within_normal_range": self._is_within_normal_range(recent_mean, vital),
            "slope": round(slope, 6) if slope else 0
        }
    
    def _identify_pattern(self, values: pd.Series) -> str:
        """Identify patterns in vital sign measurements."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Calculate consecutive increases/decreases
        diffs = values.diff().dropna()
        
        increasing = (diffs > 0).sum()
        decreasing = (diffs < 0).sum()
        
        if increasing >= len(diffs) * 0.7:
            return "consistently_increasing"
        elif decreasing >= len(diffs) * 0.7:
            return "consistently_decreasing"
        elif values.std() > values.mean() * 0.15:
            return "highly_variable"
        else:
            return "stable_with_variation"
    
    def _identify_critical_values(self, df: pd.DataFrame, vital: str) -> List[Dict[str, Any]]:
        """Identify critical values in vital signs."""
        critical_values = []
        normal_range = self.normal_ranges.get(vital, {})
        
        if not normal_range:
            return critical_values
        
        for _, row in df.iterrows():
            value = row.get(vital)
            if pd.isna(value):
                continue
            
            is_critical = False
            criticality_type = ""
            
            if "critical_high" in normal_range and value >= normal_range["critical_high"]:
                is_critical = True
                criticality_type = "critically_high"
            elif "critical_low" in normal_range and value <= normal_range["critical_low"]:
                is_critical = True
                criticality_type = "critically_low"
            
            if is_critical:
                critical_values.append({
                    "vital_sign": vital,
                    "value": value,
                    "timestamp": row["timestamp"],
                    "criticality": criticality_type,
                    "severity": "critical"
                })
        
        return critical_values
    
    def _is_within_normal_range(self, value: float, vital: str) -> bool:
        """Check if a value is within normal range."""
        normal_range = self.normal_ranges.get(vital, {})
        if not normal_range:
            return True
        
        min_val = normal_range.get("min", float('-inf'))
        max_val = normal_range.get("max", float('inf'))
        
        return min_val <= value <= max_val
    
    def _analyze_lab_values_deterioration(self, lab_data: List[Dict[str, Any]], 
                                        lookback_days: int, alert_threshold: str) -> Dict[str, Any]:
        """Analyze laboratory values for deterioration patterns."""
        if not lab_data:
            return {"status": "insufficient_data", "message": "No laboratory data available"}
        
        analysis = {
            "data_points": len(lab_data),
            "deterioration_indicators": {},
            "trend_analysis": {},
            "abnormal_results": [],
            "risk_level": "low"
        }
        
        # Key lab values to monitor
        critical_labs = ['hba1c', 'glucose', 'creatinine', 'bnp', 'nt_probnp', 'egfr']
        
        for lab_result in lab_data:
            timestamp = lab_result.get("test_date", "")
            results = lab_result.get("results", {})
            
            for lab_name, lab_info in results.items():
                if lab_name in critical_labs:
                    value = lab_info.get("value")
                    flag = lab_info.get("flag", "Normal")
                    
                    if flag in ["Critical High", "Critical Low"]:
                        analysis["abnormal_results"].append({
                            "lab": lab_name,
                            "value": value,
                            "flag": flag,
                            "timestamp": timestamp,
                            "severity": "critical"
                        })
                    elif flag in ["High", "Low"]:
                        analysis["abnormal_results"].append({
                            "lab": lab_name,
                            "value": value,
                            "flag": flag,
                            "timestamp": timestamp,
                            "severity": "moderate"
                        })
        
        # Determine risk level based on abnormal results
        critical_results = [r for r in analysis["abnormal_results"] if r["severity"] == "critical"]
        moderate_results = [r for r in analysis["abnormal_results"] if r["severity"] == "moderate"]
        
        if critical_results:
            analysis["risk_level"] = "critical" if len(critical_results) >= 2 else "high"
        elif len(moderate_results) >= 3:
            analysis["risk_level"] = "moderate"
        elif moderate_results:
            analysis["risk_level"] = "low-moderate"
        
        return analysis
    
    def _analyze_symptom_deterioration(self, symptom_data: List[Dict[str, Any]], 
                                     lookback_days: int) -> Dict[str, Any]:
        """Analyze symptom patterns for deterioration."""
        if not symptom_data:
            return {"status": "insufficient_data", "message": "No symptom data available"}
        
        analysis = {
            "symptom_trends": {},
            "new_symptoms": [],
            "worsening_symptoms": [],
            "risk_level": "low"
        }
        
        # Analyze symptom severity trends
        symptom_severity = {}
        for entry in symptom_data:
            timestamp = entry.get("timestamp", "")
            symptoms = entry.get("symptoms", {})
            
            for symptom, severity in symptoms.items():
                if symptom not in symptom_severity:
                    symptom_severity[symptom] = []
                symptom_severity[symptom].append({
                    "timestamp": timestamp,
                    "severity": severity
                })
        
        # Identify worsening symptoms
        for symptom, severity_list in symptom_severity.items():
            if len(severity_list) >= 2:
                recent_severity = severity_list[-1]["severity"]
                initial_severity = severity_list[0]["severity"]
                
                severity_scale = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
                recent_score = severity_scale.get(recent_severity, 0)
                initial_score = severity_scale.get(initial_severity, 0)
                
                if recent_score > initial_score:
                    analysis["worsening_symptoms"].append({
                        "symptom": symptom,
                        "initial_severity": initial_severity,
                        "current_severity": recent_severity,
                        "change": recent_score - initial_score
                    })
        
        # Determine risk level
        severe_symptoms = [s for s in analysis["worsening_symptoms"] if s["change"] >= 2]
        if severe_symptoms:
            analysis["risk_level"] = "high"
        elif analysis["worsening_symptoms"]:
            analysis["risk_level"] = "moderate"
        
        return analysis
    
    def _analyze_medication_adherence_deterioration(self, med_data: List[Dict[str, Any]], 
                                                  lookback_days: int) -> Dict[str, Any]:
        """Analyze medication adherence patterns."""
        if not med_data:
            return {"status": "insufficient_data", "message": "No medication data available"}
        
        analysis = {
            "adherence_trend": {},
            "poor_adherence_medications": [],
            "risk_level": "low"
        }
        
        # Calculate adherence rates over time
        med_adherence = {}
        for entry in med_data:
            date = entry.get("date", "")
            medications = entry.get("medications", [])
            
            for med in medications:
                med_name = med.get("medication_name", "")
                adherence_rate = med.get("adherence_rate", 1.0)
                
                if med_name not in med_adherence:
                    med_adherence[med_name] = []
                med_adherence[med_name].append({
                    "date": date,
                    "adherence_rate": adherence_rate
                })
        
        # Identify medications with declining adherence
        for med_name, adherence_list in med_adherence.items():
            if len(adherence_list) >= 3:
                recent_adherence = np.mean([a["adherence_rate"] for a in adherence_list[-3:]])
                if recent_adherence < 0.8:  # Less than 80% adherence
                    analysis["poor_adherence_medications"].append({
                        "medication": med_name,
                        "recent_adherence": round(recent_adherence * 100, 1),
                        "severity": "high" if recent_adherence < 0.6 else "moderate"
                    })
        
        # Determine risk level
        high_risk_meds = [m for m in analysis["poor_adherence_medications"] if m["severity"] == "high"]
        if high_risk_meds:
            analysis["risk_level"] = "high"
        elif analysis["poor_adherence_medications"]:
            analysis["risk_level"] = "moderate"
        
        return analysis
    
    def _analyze_functional_deterioration(self, functional_data: List[Dict[str, Any]], 
                                        lookback_days: int) -> Dict[str, Any]:
        """Analyze functional status changes."""
        if not functional_data:
            return {"status": "insufficient_data", "message": "No functional status data available"}
        
        analysis = {
            "functional_trends": {},
            "declining_functions": [],
            "risk_level": "low"
        }
        
        # Analyze functional domains
        functional_domains = ["mobility", "self_care", "cognitive_function", "social_function"]
        
        for entry in functional_data:
            timestamp = entry.get("timestamp", "")
            for domain in functional_domains:
                score = entry.get(domain)
                if score is not None:
                    if domain not in analysis["functional_trends"]:
                        analysis["functional_trends"][domain] = []
                    analysis["functional_trends"][domain].append({
                        "timestamp": timestamp,
                        "score": score
                    })
        
        # Identify declining functions
        for domain, scores in analysis["functional_trends"].items():
            if len(scores) >= 2:
                recent_score = scores[-1]["score"]
                initial_score = scores[0]["score"]
                decline = initial_score - recent_score  # Positive decline indicates worsening
                
                if decline > 10:  # Significant decline
                    analysis["declining_functions"].append({
                        "domain": domain,
                        "decline_amount": decline,
                        "severity": "high" if decline > 20 else "moderate"
                    })
        
        # Determine risk level
        high_decline = [d for d in analysis["declining_functions"] if d["severity"] == "high"]
        if high_decline:
            analysis["risk_level"] = "high"
        elif analysis["declining_functions"]:
            analysis["risk_level"] = "moderate"
        
        return analysis
    
    def _calculate_overall_deterioration_risk(self, vital_signs: Dict, lab_values: Dict, 
                                            symptoms: Dict, medications: Dict, 
                                            functional: Dict, alert_threshold: str) -> Dict[str, Any]:
        """Calculate overall deterioration risk from all analyses."""
        
        # Risk scoring system
        risk_scores = {
            "critical": 4,
            "high": 3,
            "moderate": 2,
            "low-moderate": 1.5,
            "low": 1,
            "insufficient_data": 0
        }
        
        # Weight different components
        weights = {
            "vital_signs": 0.3,
            "lab_values": 0.25,
            "symptoms": 0.2,
            "medications": 0.15,
            "functional": 0.1
        }
        
        # Calculate weighted risk score
        total_risk_score = 0
        component_risks = {}
        
        components = {
            "vital_signs": vital_signs,
            "lab_values": lab_values,
            "symptoms": symptoms,
            "medications": medications,
            "functional": functional
        }
        
        for component, analysis in components.items():
            risk_level = analysis.get("risk_level", "insufficient_data")
            risk_score = risk_scores.get(risk_level, 0)
            weighted_score = risk_score * weights[component]
            total_risk_score += weighted_score
            
            component_risks[component] = {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "weighted_contribution": round(weighted_score, 2)
            }
        
        # Convert to overall risk level
        if total_risk_score >= 3.5:
            overall_risk = "critical"
            urgency = "immediate"
        elif total_risk_score >= 2.5:
            overall_risk = "high"
            urgency = "urgent"
        elif total_risk_score >= 1.5:
            overall_risk = "moderate"
            urgency = "prompt"
        else:
            overall_risk = "low"
            urgency = "routine"
        
        return {
            "overall_risk_level": overall_risk,
            "urgency": urgency,
            "total_risk_score": round(total_risk_score, 2),
            "component_risks": component_risks,
            "confidence_level": self._calculate_confidence_level(components),
            "deterioration_probability": min(95, max(5, total_risk_score * 25))  # Convert to percentage
        }
    
    def _calculate_confidence_level(self, components: Dict) -> str:
        """Calculate confidence level based on data availability."""
        data_availability = 0
        total_components = len(components)
        
        for component, analysis in components.items():
            if analysis.get("status") != "insufficient_data":
                data_availability += 1
        
        confidence_ratio = data_availability / total_components
        
        if confidence_ratio >= 0.8:
            return "high"
        elif confidence_ratio >= 0.6:
            return "moderate"
        else:
            return "low"
    
    def _generate_deterioration_alerts(self, overall_assessment: Dict, 
                                     alert_threshold: str) -> List[Dict[str, Any]]:
        """Generate alerts based on deterioration analysis."""
        alerts = []
        
        overall_risk = overall_assessment["overall_risk_level"]
        urgency = overall_assessment["urgency"]
        
        # Generate alerts based on risk level and threshold settings
        threshold_mapping = {
            "low": ["critical"],
            "moderate": ["critical", "high"],
            "high": ["critical", "high", "moderate"]
        }
        
        alert_levels = threshold_mapping.get(alert_threshold, ["critical"])
        
        if overall_risk in alert_levels:
            alerts.append({
                "alert_type": "overall_deterioration",
                "severity": overall_risk,
                "urgency": urgency,
                "message": f"Patient showing {overall_risk} risk of clinical deterioration",
                "recommended_action": self._get_recommended_action(overall_risk),
                "timestamp": datetime.now().isoformat()
            })
        
        return alerts
    
    def _get_recommended_action(self, risk_level: str) -> str:
        """Get recommended action based on risk level."""
        actions = {
            "critical": "Immediate clinical evaluation required - contact provider within 2 hours",
            "high": "Urgent clinical review needed - contact provider within 24 hours",
            "moderate": "Enhanced monitoring required - schedule follow-up within 48-72 hours",
            "low": "Continue routine monitoring with scheduled follow-up"
        }
        return actions.get(risk_level, "Continue standard care")
    
    def _generate_recommendations(self, overall_assessment: Dict, 
                                alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate clinical recommendations based on analysis."""
        recommendations = []
        
        risk_level = overall_assessment["overall_risk_level"]
        component_risks = overall_assessment["component_risks"]
        
        # Risk-level specific recommendations
        if risk_level == "critical":
            recommendations.extend([
                "Immediate clinical assessment required",
                "Consider emergency department evaluation",
                "Notify primary care provider and specialists",
                "Initiate intensive monitoring protocols"
            ])
        elif risk_level == "high":
            recommendations.extend([
                "Urgent clinical review within 24 hours",
                "Increase monitoring frequency",
                "Review and optimize current treatment plan",
                "Consider telehealth check-in"
            ])
        elif risk_level == "moderate":
            recommendations.extend([
                "Schedule clinical follow-up within 48-72 hours",
                "Enhance patient education on warning signs",
                "Review medication adherence",
                "Consider remote monitoring"
            ])
        
        # Component-specific recommendations
        for component, risk_info in component_risks.items():
            if risk_info["risk_level"] in ["high", "critical"]:
                if component == "vital_signs":
                    recommendations.append("Frequent vital sign monitoring recommended")
                elif component == "lab_values":
                    recommendations.append("Repeat laboratory studies as clinically indicated")
                elif component == "medications":
                    recommendations.append("Medication therapy management consultation")
                elif component == "symptoms":
                    recommendations.append("Symptom management optimization needed")
                elif component == "functional":
                    recommendations.append("Functional assessment and rehabilitation referral")
        
        return list(set(recommendations))  # Remove duplicates
