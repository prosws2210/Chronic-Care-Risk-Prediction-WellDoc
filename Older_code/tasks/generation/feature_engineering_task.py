"""
Feature Engineering Task for transforming raw patient data into ML-ready features.
Creates temporal, clinical, and derived features optimized for risk prediction.
"""

import logging
from typing import Dict, Any
from crewai import Task

logger = logging.getLogger(__name__)

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
