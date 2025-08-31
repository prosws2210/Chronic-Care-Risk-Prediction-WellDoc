"""
Synthetic Data Generation Task for creating realistic chronic care patient datasets.
Orchestrates the creation of diverse patient populations for model training.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from crewai import Task

logger = logging.getLogger(__name__)

class SyntheticDataTask:
    """Task for generating synthetic chronic care patient data."""
    
    def __init__(self, config: Any, agents: Dict[str, Any]):
        """Initialize the synthetic data generation task."""
        self.config = config
        self.agents = agents
        
        # Create the CrewAI task
        self.task = Task(
            description=self._get_task_description(),
            expected_output=self._get_expected_output(),
            agent=self.agents['data_generator'].agent,
            tools=self.agents['data_generator'].agent.tools
        )
        
        logger.info("SyntheticDataTask initialized")
    
    def _get_task_description(self) -> str:
        """Get comprehensive task description."""
        return f"""
        Generate a comprehensive synthetic dataset of {self.config.SYNTHETIC_PATIENTS_COUNT} chronic care patients 
        with realistic medical histories spanning {self.config.MIN_HISTORY_DAYS} to {self.config.MAX_HISTORY_DAYS} days.
        
        **PRIMARY OBJECTIVE:**
        Create a diverse, clinically accurate dataset for training 90-day deterioration risk prediction models.
        
        **REQUIRED PATIENT CHARACTERISTICS:**
        - Demographics: Age (18-90), gender, ethnicity, socioeconomic factors
        - Chronic Conditions: Diabetes (Type 1/2), Heart Failure, Obesity, Hypertension
        - Medical History: Comorbidities, family history, previous hospitalizations
        - Baseline Vitals: Blood pressure, heart rate, weight, BMI, temperature
        - Laboratory Values: HbA1c, glucose, lipids, kidney function markers
        - Medications: Current prescriptions, adherence patterns, side effects
        - Lifestyle Factors: Diet, exercise, smoking, alcohol consumption
        - Healthcare Utilization: Visits, procedures, emergency encounters
        - Social Determinants: Housing, transportation, food security, education
        
        **CLINICAL REALISM REQUIREMENTS:**
        - Physiologically plausible value ranges and correlations
        - Age and gender-appropriate medical patterns
        - Realistic disease progression and comorbidity associations
        - Appropriate medication effects on biomarkers
        - Seasonal variations in health metrics
        
        **DETERIORATION RISK MODELING:**
        - 15% high-risk patients (deterioration within 90 days)
        - 25% medium-risk patients (some concerning factors)
        - 60% low-risk patients (stable condition)
        - Clear risk factor patterns for model training
        
        **DATA QUALITY STANDARDS:**
        - 5-10% missing data (realistic clinical scenario)
        - Temporal consistency in measurements
        - No impossible or contradictory values
        - Proper data type handling and encoding
        
        **OUTPUT REQUIREMENTS:**
        Generate structured patient data in JSON format with:
        - Individual patient profiles with unique identifiers
        - Time-series medical data (vitals, labs, medications)
        - Ground truth deterioration outcomes for supervised learning
        - Comprehensive metadata for data understanding
        
        Collaborate with specialist agents (diabetes, cardiology, obesity) to ensure 
        condition-specific medical accuracy and clinical validity.
        """
    
    def _get_expected_output(self) -> str:
        """Get expected output specification."""
        return f"""
        SYNTHETIC CHRONIC CARE DATASET:
        
        **DATASET STRUCTURE:**
        ```
        {{
            "metadata": {{
                "generation_timestamp": "ISO timestamp",
                "total_patients": {self.config.SYNTHETIC_PATIENTS_COUNT},
                "data_version": "1.0.0",
                "clinical_conditions_included": ["diabetes", "heart_failure", "obesity"],
                "prediction_window_days": {self.config.PREDICTION_WINDOW_DAYS}
            }},
            "patients": [
                {{
                    "patient_id": "SYNTH_000001",
                    "demographics": {{
                        "age": 67,
                        "gender": "Female",
                        "race_ethnicity": "White",
                        "insurance_type": "Medicare",
                        "zip_code": "12345"
                    }},
                    "clinical_profile": {{
                        "primary_condition": "diabetes",
                        "condition_details": {{
                            "diabetes_type": "Type_2",
                            "years_since_diagnosis": 12,
                            "last_hba1c": 8.2,
                            "complications": ["retinopathy"]
                        }},
                        "comorbidities": ["hypertension", "dyslipidemia"],
                        "baseline_vitals": {{
                            "systolic_bp": 145,
                            "diastolic_bp": 88,
                            "heart_rate": 78,
                            "bmi": 31.2
                        }},
                        "medications": [
                            {{
                                "name": "metformin",
                                "dosage": "1000mg",
                                "frequency": "twice_daily",
                                "adherence_rate": 0.85
                            }}
                        ]
                    }},
                    "time_series_data": {{
                        "vitals": [
                            {{
                                "timestamp": "2024-01-01T08:00:00Z",
                                "systolic_bp": 142,
                                "diastolic_bp": 85,
                                "heart_rate": 76
                            }}
                        ],
                        "laboratory_results": [
                            {{
                                "test_date": "2024-01-15",
                                "hba1c": 8.1,
                                "glucose_fasting": 165,
                                "creatinine": 1.2
                            }}
                        ]
                    }},
                    "deterioration_risk": {{
                        "risk_probability": 45.2,
                        "risk_level": "Medium",
                        "deterioration_occurs": false,
                        "primary_risk_factors": ["poor_glycemic_control", "hypertension"]
                    }}
                }}
            ],
            "summary_statistics": {{
                "demographics": {{
                    "mean_age": 64.5,
                    "gender_distribution": {{"Male": 48, "Female": 52}},
                    "condition_prevalence": {{
                        "diabetes": 40,
                        "heart_failure": 25,
                        "obesity": 35
                    }}
                }},
                "risk_distribution": {{
                    "high_risk": 15,
                    "medium_risk": 25,
                    "low_risk": 60
                }}
            }}
        }}
        ```
        
        **QUALITY ASSURANCE CHECKLIST:**
        ✓ All patients have complete demographic information
        ✓ Medical values are within physiologically plausible ranges
        ✓ Comorbidity patterns reflect real-world associations
        ✓ Medication lists match diagnosed conditions
        ✓ Risk levels correlate with clinical factors
        ✓ Data types are consistent and properly formatted
        ✓ No duplicate patient identifiers
        ✓ Time-series data shows realistic temporal patterns
        
        **FILES GENERATED:**
        - `synthetic_patients.json`: Complete patient dataset
        - `data_dictionary.json`: Variable definitions and ranges
        - `generation_report.md`: Process documentation and statistics
        - `quality_validation.json`: Data quality assessment results
        """


class FeatureEngineeringTask:
    """Task for engineering features from synthetic patient data."""
    
    def __init__(self, config: Any, agents: Dict[str, Any]):
        """Initialize the feature engineering task."""
        self.config = config
        self.agents = agents
        
        # Create the CrewAI task
        self.task = Task(
            description=self._get_task_description(),
            expected_output=self._get_expected_output(),
            agent=self.agents['data_generator'].agent,
            tools=self.agents['data_generator'].agent.tools
        )
        
        logger.info("FeatureEngineeringTask initialized")
    
    def _get_task_description(self) -> str:
        """Get comprehensive task description."""
        return """
        Transform raw synthetic patient data into engineered features optimized for 
        90-day deterioration risk prediction in chronic care patients.
        
        **PRIMARY OBJECTIVE:**
        Create a comprehensive feature set that captures temporal patterns, clinical relationships, 
        and risk indicators from the synthetic patient dataset.
        
        **FEATURE ENGINEERING CATEGORIES:**
        
        1. **DEMOGRAPHIC FEATURES:**
           - Age categories (young_adult, middle_aged, elderly, very_elderly)
           - Gender encoding and interactions
           - Socioeconomic risk indicators
           - Geographic health factors
        
        2. **CLINICAL CONDITION FEATURES:**
           - Disease duration and severity indicators
           - Comorbidity burden scores
           - Condition-specific risk markers
           - Disease interaction effects
        
        3. **TEMPORAL FEATURES:**
           - Trend analysis (improving, stable, worsening)
           - Variability measures (standard deviation, coefficient of variation)
           - Rate of change calculations
           - Seasonal pattern indicators
        
        4. **VITAL SIGNS FEATURES:**
           - Moving averages (7-day, 30-day)
           - Abnormal value frequency
           - Control status indicators
           - Threshold crossing events
        
        5. **LABORATORY FEATURES:**
           - Test result trends and deltas
           - Abnormal flag frequencies
           - Clinical decision thresholds
           - Biomarker ratios and indices
        
        6. **MEDICATION FEATURES:**
           - Adherence pattern analysis
           - Polypharmacy indicators
           - Drug class combinations
           - Side effect risk scores
        
        7. **HEALTHCARE UTILIZATION FEATURES:**
           - Visit frequency patterns
           - Emergency encounter history
           - Hospitalization recency
           - Care gap indicators
        
        8. **DERIVED CLINICAL INDICES:**
           - Charlson Comorbidity Index
           - Frailty indicators
           - Risk prediction scores
           - Composite health metrics
        
        **FEATURE SELECTION CRITERIA:**
        - Clinical relevance and interpretability
        - Predictive power for deterioration risk
        - Low correlation with other features
        - Robust to missing data
        - Computationally efficient
        
        **DATA PREPROCESSING:**
        - Handle missing values appropriately
        - Scale numerical features consistently
        - Encode categorical variables properly
        - Create interaction terms for key relationships
        - Implement feature versioning for reproducibility
        
        Ensure all engineered features maintain clinical interpretability and can be 
        explained to healthcare providers for decision support.
        """
    
    def _get_expected_output(self) -> str:
        """Get expected output specification."""
        return """
        ENGINEERED FEATURE DATASET:
        
        **FEATURE MATRIX STRUCTURE:**
        ```
        patient_id | feature_1 | feature_2 | ... | feature_n | target
        SYNTH_001  | 0.45      | 1.2       | ... | 0.78      | 0
        SYNTH_002  | 0.67      | 0.9       | ... | 1.23      | 1
        ```
        
        **FEATURE CATEGORIES INCLUDED:**
        - Demographics: 15 features
        - Clinical Conditions: 25 features  
        - Temporal Patterns: 35 features
        - Vital Signs: 40 features
        - Laboratory Values: 30 features
        - Medications: 20 features
        - Healthcare Utilization: 15 features
        - Derived Indices: 20 features
        **Total: ~200 engineered features**
        
        **KEY ENGINEERED FEATURES:**
        - `age_risk_category`: Age-based risk stratification
        - `hba1c_trend_30d`: 30-day HbA1c trend indicator
        - `bp_control_status`: Blood pressure control assessment
        - `medication_adherence_score`: Composite adherence metric
        - `comorbidity_burden_index`: Weighted comorbidity score
        - `healthcare_intensity`: Utilization-based risk indicator
        - `deterioration_velocity`: Rate of clinical decline
        - `stability_index`: Overall health stability measure
        
        **OUTPUT FILES:**
        - `engineered_features.csv`: Complete feature matrix
        - `feature_dictionary.json`: Detailed feature documentation
        - `feature_importance_ranking.json`: Preliminary importance scores
        - `feature_correlation_matrix.csv`: Inter-feature relationships
        - `preprocessing_pipeline.pkl`: Reusable transformation pipeline
        - `feature_engineering_report.md`: Process documentation
        
        **DATA QUALITY METRICS:**
        - Missing value percentage per feature (<5%)
        - Feature distribution statistics
        - Correlation analysis results
        - Outlier detection summary
        - Feature stability assessment
        """
