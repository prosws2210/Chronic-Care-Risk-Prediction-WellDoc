"""
Vital Simulator Tool for generating realistic time-series vital signs data.
Creates temporal patterns for blood pressure, heart rate, temperature, and other vitals.
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

class VitalSimulatorInput(BaseModel):
    """Input schema for vital signs simulation."""
    patient_id: str = Field(description="Patient identifier")
    baseline_vitals: Dict[str, Any] = Field(description="Baseline vital signs")
    days_history: int = Field(default=90, description="Number of days of history to generate")
    measurement_frequency: str = Field(default="daily", description="Measurement frequency (daily, weekly, etc.)")
    chronic_conditions: List[str] = Field(default=[], description="Patient's chronic conditions")

class VitalSimulatorTool(BaseTool):
    """Tool for simulating realistic time-series vital signs data."""
    
    name: str = "Vital Signs Simulator"
    description: str = "Generates realistic temporal vital signs patterns for chronic care patients"
    args_schema: type[BaseModel] = VitalSimulatorInput
    
    def __init__(self):
        super().__init__()
        logger.info("VitalSimulatorTool initialized")
    
    def _run(self, patient_id: str, baseline_vitals: Dict[str, Any], 
             days_history: int = 90, measurement_frequency: str = "daily",
             chronic_conditions: List[str] = None) -> str:
        """Generate time-series vital signs data."""
        try:
            if chronic_conditions is None:
                chronic_conditions = []
                
            logger.info(f"Generating {days_history} days of vitals for patient {patient_id}")
            
            # Generate timestamps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_history)
            timestamps = self._generate_timestamps(start_date, end_date, measurement_frequency)
            
            # Generate vital signs time series
            vitals_data = []
            for i, timestamp in enumerate(timestamps):
                vital_reading = self._generate_vital_reading(
                    baseline_vitals, i, len(timestamps), chronic_conditions, timestamp
                )
                vital_reading["timestamp"] = timestamp.isoformat()
                vitals_data.append(vital_reading)
            
            result = {
                "patient_id": patient_id,
                "vital_signs": vitals_data,
                "metadata": {
                    "baseline_vitals": baseline_vitals,
                    "days_history": days_history,
                    "measurement_frequency": measurement_frequency,
                    "chronic_conditions": chronic_conditions,
                    "total_measurements": len(vitals_data)
                },
                "summary_statistics": self._calculate_vital_statistics(vitals_data)
            }
            
            logger.info(f"Generated {len(vitals_data)} vital sign measurements")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error generating vital signs: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _generate_timestamps(self, start_date: datetime, end_date: datetime, 
                           frequency: str) -> List[datetime]:
        """Generate measurement timestamps based on frequency."""
        timestamps = []
        
        if frequency == "daily":
            current = start_date
            while current <= end_date:
                # Add some realistic variation in measurement times
                hour_variation = np.random.normal(0, 2)  # ±2 hours variation
                measurement_time = current.replace(hour=8) + timedelta(hours=hour_variation)
                timestamps.append(measurement_time)
                current += timedelta(days=1)
                
        elif frequency == "twice_daily":
            current = start_date
            while current <= end_date:
                # Morning measurement
                morning_time = current.replace(hour=8) + timedelta(hours=np.random.normal(0, 1))
                timestamps.append(morning_time)
                
                # Evening measurement
                evening_time = current.replace(hour=20) + timedelta(hours=np.random.normal(0, 1))
                timestamps.append(evening_time)
                
                current += timedelta(days=1)
                
        elif frequency == "weekly":
            current = start_date
            while current <= end_date:
                timestamps.append(current)
                current += timedelta(days=7)
        
        # Add some missed measurements (realistic non-adherence)
        if len(timestamps) > 7:  # Only for longer periods
            missed_indices = np.random.choice(
                len(timestamps), 
                size=int(len(timestamps) * 0.1),  # 10% missed measurements
                replace=False
            )
            timestamps = [ts for i, ts in enumerate(timestamps) if i not in missed_indices]
        
        return sorted(timestamps)
    
    def _generate_vital_reading(self, baseline: Dict[str, Any], day_index: int, 
                              total_days: int, conditions: List[str], 
                              timestamp: datetime) -> Dict[str, Any]:
        """Generate a single vital signs reading with realistic patterns."""
        
        # Calculate progression factor (0 to 1 over time)
        progression = day_index / total_days
        
        # Generate blood pressure with trends and variation
        bp_data = self._simulate_blood_pressure(
            baseline['systolic_bp'], baseline['diastolic_bp'], 
            progression, conditions, timestamp
        )
        
        # Generate heart rate
        heart_rate = self._simulate_heart_rate(
            baseline['heart_rate'], progression, conditions, timestamp
        )
        
        # Generate temperature
        temperature = self._simulate_temperature(
            baseline['temperature_f'], progression, conditions, timestamp
        )
        
        # Generate oxygen saturation
        oxygen_sat = self._simulate_oxygen_saturation(
            baseline['oxygen_saturation'], progression, conditions
        )
        
        # Generate respiratory rate
        resp_rate = self._simulate_respiratory_rate(
            baseline['respiratory_rate'], progression, conditions
        )
        
        # Add measurement quality indicators
        quality_indicators = self._generate_quality_indicators()
        
        return {
            "systolic_bp": bp_data['systolic'],
            "diastolic_bp": bp_data['diastolic'],
            "heart_rate": heart_rate,
            "temperature_f": temperature,
            "oxygen_saturation": oxygen_sat,
            "respiratory_rate": resp_rate,
            "measurement_quality": quality_indicators,
            "notes": self._generate_measurement_notes(bp_data, heart_rate, temperature)
        }
    
    def _simulate_blood_pressure(self, baseline_systolic: int, baseline_diastolic: int,
                               progression: float, conditions: List[str], 
                               timestamp: datetime) -> Dict[str, int]:
        """Simulate blood pressure with realistic patterns."""
        
        # Base trend (slight increase over time for chronic patients)
        trend_factor = 1 + (progression * 0.1)  # Up to 10% increase
        
        # Condition-specific adjustments
        condition_adjustment = 1.0
        if "hypertension" in [c.lower() for c in conditions]:
            condition_adjustment *= 1.15  # 15% higher for hypertensive patients
        if "heart_failure" in [c.lower() for c in conditions]:
            condition_adjustment *= 1.10  # 10% higher for heart failure
        
        # Circadian rhythm (higher in morning, lower at night)
        hour = timestamp.hour
        circadian_factor = 1.0
        if 6 <= hour <= 10:  # Morning surge
            circadian_factor = 1.15
        elif 14 <= hour <= 18:  # Afternoon elevation
            circadian_factor = 1.05
        elif 22 <= hour or hour <= 4:  # Night dip
            circadian_factor = 0.90
        
        # Seasonal variation (higher in winter)
        month = timestamp.month
        seasonal_factor = 1.0
        if month in [12, 1, 2]:  # Winter
            seasonal_factor = 1.08
        elif month in [6, 7, 8]:  # Summer
            seasonal_factor = 0.95
        
        # Random day-to-day variation
        daily_variation = np.random.normal(1, 0.08)  # ±8% variation
        
        # Calculate final values
        systolic = baseline_systolic * trend_factor * condition_adjustment * circadian_factor * seasonal_factor * daily_variation
        diastolic = baseline_diastolic * trend_factor * condition_adjustment * circadian_factor * seasonal_factor * daily_variation
        
        # Add some correlation between systolic and diastolic
        if systolic > baseline_systolic * 1.2:  # High systolic
            diastolic *= 1.1  # Slightly increase diastolic too
        
        # Ensure realistic ranges
        systolic = max(80, min(250, int(systolic)))
        diastolic = max(50, min(130, int(diastolic)))
        
        # Ensure diastolic is not higher than systolic
        if diastolic >= systolic:
            diastolic = systolic - 20
        
        return {"systolic": systolic, "diastolic": diastolic}
    
    def _simulate_heart_rate(self, baseline_hr: int, progression: float, 
                           conditions: List[str], timestamp: datetime) -> int:
        """Simulate heart rate with realistic patterns."""
        
        # Condition-specific adjustments
        condition_adjustment = 1.0
        if "heart_failure" in [c.lower() for c in conditions]:
            condition_adjustment *= 1.15  # Higher HR for heart failure
        if "diabetes" in [c.lower() for c in conditions]:
            condition_adjustment *= 1.05  # Slightly higher for diabetes
        
        # Circadian rhythm
        hour = timestamp.hour
        circadian_factor = 1.0
        if 8 <= hour <= 12:  # Morning activity
            circadian_factor = 1.10
        elif 20 <= hour <= 23:  # Evening
            circadian_factor = 1.05
        elif 0 <= hour <= 6:  # Sleep
            circadian_factor = 0.85
        
        # Activity level simulation (some measurements during activity)
        activity_factor = 1.0
        if np.random.random() < 0.15:  # 15% chance of elevated HR due to activity
            activity_factor = np.random.uniform(1.2, 1.6)
        
        # Random variation
        daily_variation = np.random.normal(1, 0.10)
        
        heart_rate = baseline_hr * condition_adjustment * circadian_factor * activity_factor * daily_variation
        
        return max(40, min(180, int(heart_rate)))
    
    def _simulate_temperature(self, baseline_temp: float, progression: float,
                            conditions: List[str], timestamp: datetime) -> float:
        """Simulate body temperature with realistic patterns."""
        
        # Circadian rhythm (lower in early morning, higher in late afternoon)
        hour = timestamp.hour
        circadian_adjustment = 0.0
        if 4 <= hour <= 6:  # Early morning low
            circadian_adjustment = -0.8
        elif 16 <= hour <= 19:  # Late afternoon high
            circadian_adjustment = 0.6
        
        # Occasional fever episodes (5% chance)
        fever_adjustment = 0.0
        if np.random.random() < 0.05:
            fever_adjustment = np.random.uniform(1.0, 3.5)  # Mild to moderate fever
        
        # Random daily variation
        daily_variation = np.random.normal(0, 0.3)
        
        temperature = baseline_temp + circadian_adjustment + fever_adjustment + daily_variation
        
        return max(95.0, min(105.0, round(temperature, 1)))
    
    def _simulate_oxygen_saturation(self, baseline_o2: int, progression: float,
                                  conditions: List[str]) -> int:
        """Simulate oxygen saturation."""
        
        # Condition-specific adjustments
        condition_adjustment = 0
        if "copd" in [c.lower() for c in conditions]:
            condition_adjustment = -3  # Lower O2 sat for COPD
        if "heart_failure" in [c.lower() for c in conditions]:
            condition_adjustment = -1  # Slightly lower for heart failure
        
        # Random variation (usually small for O2 sat)
        daily_variation = np.random.normal(0, 1.5)
        
        o2_sat = baseline_o2 + condition_adjustment + daily_variation
        
        return max(85, min(100, int(o2_sat)))
    
    def _simulate_respiratory_rate(self, baseline_rr: int, progression: float,
                                 conditions: List[str]) -> int:
        """Simulate respiratory rate."""
        
        # Condition-specific adjustments
        condition_adjustment = 1.0
        if "copd" in [c.lower() for c in conditions]:
            condition_adjustment *= 1.20  # Higher RR for COPD
        if "heart_failure" in [c.lower() for c in conditions]:
            condition_adjustment *= 1.15  # Higher RR for heart failure
        
        # Random variation
        daily_variation = np.random.normal(1, 0.08)
        
        resp_rate = baseline_rr * condition_adjustment * daily_variation
        
        return max(8, min(35, int(resp_rate)))
    
    def _generate_quality_indicators(self) -> Dict[str, Any]:
        """Generate measurement quality indicators."""
        return {
            "measurement_confidence": np.random.choice(['High', 'Medium', 'Low'], p=[0.80, 0.15, 0.05]),
            "patient_position": np.random.choice(['Sitting', 'Standing', 'Lying'], p=[0.70, 0.20, 0.10]),
            "measurement_method": np.random.choice(['Automatic', 'Manual'], p=[0.85, 0.15]),
            "patient_state": np.random.choice(['Resting', 'Post-activity', 'Anxious'], p=[0.75, 0.15, 0.10])
        }
    
    def _generate_measurement_notes(self, bp_data: Dict, heart_rate: int, 
                                  temperature: float) -> Optional[str]:
        """Generate clinical notes for abnormal measurements."""
        notes = []
        
        if bp_data['systolic'] > 160 or bp_data['diastolic'] > 100:
            notes.append("Elevated blood pressure")
        if bp_data['systolic'] < 90 or bp_data['diastolic'] < 60:
            notes.append("Low blood pressure")
        
        if heart_rate > 100:
            notes.append("Tachycardia")
        elif heart_rate < 60:
            notes.append("Bradycardia")
        
        if temperature > 100.4:
            notes.append("Fever")
        elif temperature < 96.0:
            notes.append("Hypothermia")
        
        return "; ".join(notes) if notes else None
    
    def _calculate_vital_statistics(self, vitals_data: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for the vital signs data."""
        try:
            df = pd.DataFrame(vitals_data)
            
            return {
                "blood_pressure": {
                    "systolic_mean": float(df['systolic_bp'].mean()),
                    "systolic_std": float(df['systolic_bp'].std()),
                    "diastolic_mean": float(df['diastolic_bp'].mean()),
                    "diastolic_std": float(df['diastolic_bp'].std()),
                    "hypertensive_readings": int((df['systolic_bp'] > 140).sum() + (df['diastolic_bp'] > 90).sum())
                },
                "heart_rate": {
                    "mean": float(df['heart_rate'].mean()),
                    "std": float(df['heart_rate'].std()),
                    "tachycardia_episodes": int((df['heart_rate'] > 100).sum()),
                    "bradycardia_episodes": int((df['heart_rate'] < 60).sum())
                },
                "temperature": {
                    "mean": float(df['temperature_f'].mean()),
                    "std": float(df['temperature_f'].std()),
                    "fever_episodes": int((df['temperature_f'] > 100.4).sum())
                },
                "measurement_quality": {
                    "high_confidence_percentage": float((df['measurement_quality'].apply(lambda x: x['measurement_confidence']) == 'High').mean() * 100)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating vital statistics: {str(e)}")
            return {"error": "Unable to calculate vital statistics"}
