"""
Dashboard Utility Functions
===========================

Helper functions for the Chronic Care Risk Prediction Dashboard.
Includes data loading, processing, and clinical calculation utilities.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging

# ---- Logging Setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Data Loading Functions ----
def load_patient_data(data_path: str = "data/processed/chronic_care_data_processed.csv") -> Optional[pd.DataFrame]:
    """
    Load processed patient data for dashboard
    
    Args:
        data_path: Path to processed patient data CSV
        
    Returns:
        DataFrame with patient data or None if loading fails
    """
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} patients from {data_path}")
            return df
        else:
            logger.warning(f"Data file not found: {data_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading patient data: {str(e)}")
        return None

def load_model_artifacts(model_path: str = "models/saved") -> Tuple[Any, Any]:
    """
    Load trained model and scaler
    
    Args:
        model_path: Directory containing model artifacts
        
    Returns:
        Tuple of (model, scaler) or (None, None) if loading fails
    """
    try:
        model_file = os.path.join(model_path, "risk_prediction_model.pkl")
        scaler_file = os.path.join(model_path, "feature_scaler.pkl")
        
        model = None
        scaler = None
        
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info("Model loaded successfully")
        
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            logger.info("Scaler loaded successfully")
        
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        return None, None

def generate_synthetic_patient_data(n_patients: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic patient data for testing/demo
    
    Args:
        n_patients: Number of synthetic patients to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic patient data
    """
    np.random.seed(seed)
    
    # Base demographics
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(65, 15, n_patients).astype(int),
        'gender': np.random.choice([0, 1], n_patients, p=[0.45, 0.55]),
    }
    
    # Ensure realistic age bounds
    data['age'] = np.clip(data['age'], 18, 95)
    
    # Chronic conditions (with age correlation)
    age_factor = (data['age'] - 18) / 77  # Normalized age factor
    
    diabetes_prob = 0.2 + age_factor * 0.3
    data['diabetes_type2'] = np.random.binomial(1, diabetes_prob, n_patients)
    
    obesity_prob = 0.3 + data['diabetes_type2'] * 0.2
    data['obesity'] = np.random.binomial(1, obesity_prob, n_patients)
    
    hf_prob = 0.1 + age_factor * 0.2 + data['diabetes_type2'] * 0.15
    data['heart_failure'] = np.random.binomial(1, hf_prob, n_patients)
    
    htn_prob = 0.3 + age_factor * 0.4 + data['obesity'] * 0.2
    data['hypertension'] = np.random.binomial(1, htn_prob, n_patients)
    
    copd_prob = 0.08 + age_factor * 0.12
    data['copd'] = np.random.binomial(1, copd_prob, n_patients)
    
    # Clinical measurements (condition-dependent)
    # BMI
    base_bmi = 26 + data['obesity'] * 8 + np.random.normal(0, 4, n_patients)
    data['bmi'] = np.clip(base_bmi, 16, 50)
    
    # Blood pressure (hypertension-dependent)
    base_systolic = 130 + data['hypertension'] * 25 + np.random.normal(0, 15, n_patients)
    data['systolic_bp'] = np.clip(base_systolic, 90, 200)
    
    base_diastolic = 80 + data['hypertension'] * 15 + np.random.normal(0, 10, n_patients)
    data['diastolic_bp'] = np.clip(base_diastolic, 60, 120)
    
    # Heart rate
    data['heart_rate'] = np.clip(np.random.normal(75, 12, n_patients), 50, 120)
    
    # Glucose and HbA1c (diabetes-dependent)
    base_glucose = 100 + data['diabetes_type2'] * 60 + np.random.normal(0, 25, n_patients)
    data['glucose_level'] = np.clip(base_glucose, 70, 400)
    
    base_hba1c = 5.5 + data['diabetes_type2'] * 2.5 + np.random.normal(0, 1, n_patients)
    data['hba1c'] = np.clip(base_hba1c, 4.0, 15.0)
    
    # Behavioral factors
    data['medication_adherence'] = np.clip(
        np.random.beta(3, 1, n_patients) - data['diabetes_type2'] * 0.1, 0.3, 1.0
    )
    
    data['exercise_frequency'] = np.clip(
        np.random.poisson(3, n_patients) - data['obesity'] * 1, 0, 7
    )
    
    # Smoking (age and COPD related)
    smoking_prob = 0.15 + data['copd'] * 0.3 - age_factor * 0.1
    data['smoking_status'] = np.random.binomial(1, smoking_prob, n_patients)
    
    data['alcohol_consumption'] = np.clip(np.random.poisson(2, n_patients), 0, 14)
    data['sleep_hours'] = np.clip(np.random.normal(7, 1.2, n_patients), 4, 12)
    
    # Create realistic deterioration outcome based on risk factors
    risk_score = (
        (data['age'] > 75) * 0.15 +
        (data['bmi'] > 35) * 0.12 +
        (data['systolic_bp'] > 160) * 0.18 +
        (data['hba1c'] > 9) * 0.20 +
        (data['medication_adherence'] < 0.7) * 0.15 +
        data['heart_failure'] * 0.25 +
        data['smoking_status'] * 0.10 +
        (data['exercise_frequency'] < 2) * 0.08
    )
    
    # Add noise and create binary outcome
    risk_score += np.random.normal(0, 0.08, n_patients)
    data['deterioration_90d'] = (risk_score > 0.35).astype(int)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    logger.info(f"Generated {n_patients} synthetic patients")
    logger.info(f"Deterioration rate: {df['deterioration_90d'].mean():.1%}")
    
    return df

# ---- Clinical Calculation Functions ----
def calculate_risk_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate population-level risk metrics
    
    Args:
        df: DataFrame with patient data and risk probabilities
        
    Returns:
        Dictionary of calculated metrics
    """
    if 'risk_probability' not in df.columns:
        return {}
    
    metrics = {
        'total_patients': len(df),
        'mean_risk': df['risk_probability'].mean(),
        'median_risk': df['risk_probability'].median(),
        'std_risk': df['risk_probability'].std(),
        'high_risk_count': len(df[df['risk_probability'] > 0.7]),
        'medium_risk_count': len(df[(df['risk_probability'] >= 0.3) & (df['risk_probability'] <= 0.7)]),
        'low_risk_count': len(df[df['risk_probability'] < 0.3]),
        'high_risk_percentage': len(df[df['risk_probability'] > 0.7]) / len(df) * 100,
        'urgent_cases': len(df[df['risk_probability'] > 0.8])
    }
    
    return metrics

def create_feature_explanation(feature_name: str, value: float, population_stats: pd.Series) -> Dict[str, Any]:
    """
    Create clinical explanation for a feature value
    
    Args:
        feature_name: Name of the clinical feature
        value: Patient's value for the feature
        population_stats: Population statistics for the feature
        
    Returns:
        Dictionary with feature explanation
    """
    mean_val = population_stats['mean']
    std_val = population_stats['std']
    z_score = (value - mean_val) / std_val if std_val > 0 else 0
    
    # Clinical interpretations
    clinical_interpretations = {
        'age': {
            'unit': 'years',
            'normal_range': '18-65',
            'high_risk_threshold': 75,
            'clinical_note': 'Advanced age is a non-modifiable risk factor'
        },
        'bmi': {
            'unit': 'kg/m²',
            'normal_range': '18.5-24.9',
            'high_risk_threshold': 30,
            'clinical_note': 'Obesity significantly increases chronic disease risk'
        },
        'systolic_bp': {
            'unit': 'mmHg',
            'normal_range': '<120',
            'high_risk_threshold': 140,
            'clinical_note': 'Elevated BP increases cardiovascular risk'
        },
        'hba1c': {
            'unit': '%',
            'normal_range': '<5.7',
            'high_risk_threshold': 7.0,
            'clinical_note': 'HbA1c reflects 3-month average glucose control'
        },
        'medication_adherence': {
            'unit': 'proportion',
            'normal_range': '>0.8',
            'high_risk_threshold': 0.7,
            'clinical_note': 'Poor adherence significantly impacts outcomes'
        }
    }
    
    interpretation = clinical_interpretations.get(feature_name, {
        'unit': 'units',
        'normal_range': 'varies',
        'high_risk_threshold': None,
        'clinical_note': 'Clinical significance varies'
    })
    
    # Determine risk level
    if interpretation['high_risk_threshold']:
        if isinstance(interpretation['high_risk_threshold'], (int, float)):
            if feature_name == 'medication_adherence':
                risk_level = 'High' if value < interpretation['high_risk_threshold'] else 'Normal'
            else:
                risk_level = 'High' if value > interpretation['high_risk_threshold'] else 'Normal'
        else:
            risk_level = 'Unknown'
    else:
        risk_level = 'High' if abs(z_score) > 2 else 'Normal'
    
    return {
        'feature': feature_name,
        'value': value,
        'unit': interpretation['unit'],
        'normal_range': interpretation['normal_range'],
        'z_score': z_score,
        'risk_level': risk_level,
        'clinical_note': interpretation['clinical_note'],
        'percentile': calculate_percentile(value, population_stats)
    }

def calculate_percentile(value: float, population_stats: pd.Series) -> float:
    """Calculate percentile of value within population"""
    try:
        from scipy import stats
        return stats.percentileofscore(population_stats, value)
    except ImportError:
        # Fallback approximation using z-score
        z_score = (value - population_stats['mean']) / population_stats['std']
        # Rough approximation: z-score to percentile
        if z_score < -2:
            return 2.5
        elif z_score < -1:
            return 16
        elif z_score < 0:
            return 50 - (abs(z_score) * 34)
        elif z_score < 1:
            return 50 + (z_score * 34)
        elif z_score < 2:
            return 84
        else:
            return 97.5

def generate_clinical_recommendations(patient_data: pd.Series, risk_factors: List[Dict]) -> List[Dict[str, str]]:
    """
    Generate clinical recommendations based on patient data and risk factors
    
    Args:
        patient_data: Patient's clinical data
        risk_factors: List of risk factors with importance scores
        
    Returns:
        List of clinical recommendations
    """
    recommendations = []
    
    # Diabetes management
    if patient_data.get('hba1c', 7) > 8.5:
        recommendations.append({
            'category': 'Diabetes Management',
            'priority': 'High',
            'action': 'Intensify diabetes therapy',
            'details': f'HbA1c {patient_data["hba1c"]:.1f}% indicates poor glycemic control',
            'timeline': '1-2 weeks',
            'specialist': 'Endocrinology'
        })
    elif patient_data.get('hba1c', 7) > 7.5:
        recommendations.append({
            'category': 'Diabetes Monitoring',
            'priority': 'Medium',
            'action': 'Enhanced glucose monitoring',
            'details': 'HbA1c above target - consider CGM or more frequent testing',
            'timeline': '2-4 weeks',
            'specialist': 'Primary Care'
        })
    
    # Hypertension management
    if patient_data.get('systolic_bp', 120) > 160:
        recommendations.append({
            'category': 'Hypertension Management',
            'priority': 'High',
            'action': 'Urgent BP control',
            'details': f'Systolic BP {patient_data["systolic_bp"]:.0f} mmHg requires immediate attention',
            'timeline': '1 week',
            'specialist': 'Cardiology'
        })
    elif patient_data.get('systolic_bp', 120) > 140:
        recommendations.append({
            'category': 'BP Monitoring',
            'priority': 'Medium',
            'action': 'Optimize antihypertensive therapy',
            'details': 'Consider medication adjustment or lifestyle modifications',
            'timeline': '2-3 weeks',
            'specialist': 'Primary Care'
        })
    
    # Medication adherence
    if patient_data.get('medication_adherence', 1.0) < 0.6:
        recommendations.append({
            'category': 'Medication Management',
            'priority': 'High',
            'action': 'Address medication adherence',
            'details': 'Poor adherence significantly impacts outcomes',
            'timeline': '1-2 weeks',
            'specialist': 'Pharmacy'
        })
    elif patient_data.get('medication_adherence', 1.0) < 0.8:
        recommendations.append({
            'category': 'Adherence Support',
            'priority': 'Medium',
            'action': 'Adherence counseling and support',
            'details': 'Consider pill organizers, reminder systems',
            'timeline': '2-4 weeks',
            'specialist': 'Primary Care'
        })
    
    # Weight management
    if patient_data.get('bmi', 25) > 35:
        recommendations.append({
            'category': 'Weight Management',
            'priority': 'Medium',
            'action': 'Structured weight loss program',
            'details': f'BMI {patient_data["bmi"]:.1f} - consider bariatric evaluation',
            'timeline': '1-2 months',
            'specialist': 'Nutrition/Bariatric'
        })
    elif patient_data.get('bmi', 25) > 30:
        recommendations.append({
            'category': 'Lifestyle Modification',
            'priority': 'Medium',
            'action': 'Nutritional counseling',
            'details': 'Focus on sustainable weight loss strategies',
            'timeline': '1 month',
            'specialist': 'Dietitian'
        })
    
    # Exercise/lifestyle
    if patient_data.get('exercise_frequency', 3) < 2:
        recommendations.append({
            'category': 'Exercise Program',
            'priority': 'Low',
            'action': 'Increase physical activity',
            'details': 'Start with low-impact activities, gradual progression',
            'timeline': '1-2 months',
            'specialist': 'Physical Therapy'
        })
    
    # Smoking cessation
    if patient_data.get('smoking_status', 0) == 1:
        recommendations.append({
            'category': 'Smoking Cessation',
            'priority': 'High',
            'action': 'Immediate smoking cessation',
            'details': 'Consider nicotine replacement, counseling, medications',
            'timeline': 'Immediate',
            'specialist': 'Smoking Cessation'
        })
    
    # Sort by priority
    priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    # Add default if no specific recommendations
    if not recommendations:
        recommendations.append({
            'category': 'Routine Care',
            'priority': 'Low',
            'action': 'Continue current management',
            'details': 'Patient appears stable, maintain regular monitoring',
            'timeline': '3-6 months',
            'specialist': 'Primary Care'
        })
    
    return recommendations[:5]  # Return top 5 recommendations

def format_clinical_value(value: float, feature_name: str) -> str:
    """
    Format clinical values with appropriate units and precision
    
    Args:
        value: Numeric value to format
        feature_name: Name of the clinical feature
        
    Returns:
        Formatted string with value and units
    """
    format_specs = {
        'age': {'decimal_places': 0, 'unit': 'years'},
        'bmi': {'decimal_places': 1, 'unit': 'kg/m²'},
        'systolic_bp': {'decimal_places': 0, 'unit': 'mmHg'},
        'diastolic_bp': {'decimal_places': 0, 'unit': 'mmHg'},
        'heart_rate': {'decimal_places': 0, 'unit': 'bpm'},
        'glucose_level': {'decimal_places': 0, 'unit': 'mg/dL'},
        'hba1c': {'decimal_places': 1, 'unit': '%'},
        'medication_adherence': {'decimal_places': 0, 'unit': '%', 'multiply': 100},
        'exercise_frequency': {'decimal_places': 0, 'unit': '/week'},
        'sleep_hours': {'decimal_places': 1, 'unit': 'hours'}
    }
    
    spec = format_specs.get(feature_name, {'decimal_places': 2, 'unit': ''})
    
    # Apply multiplier if specified (e.g., for percentages)
    display_value = value * spec.get('multiply', 1)
    
    # Format with appropriate decimal places
    if spec['decimal_places'] == 0:
        formatted_value = f"{display_value:.0f}"
    else:
        formatted_value = f"{display_value:.{spec['decimal_places']}f}"
    
    # Add unit
    return f"{formatted_value} {spec['unit']}".strip()

def export_patient_data(df: pd.DataFrame, filename: str = None) -> str:
    """
    Export patient data to CSV with timestamp
    
    Args:
        df: DataFrame to export
        filename: Optional filename (will add timestamp if not provided)
        
    Returns:
        Path to exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"patient_data_export_{timestamp}.csv"
    
    # Ensure exports directory exists
    export_dir = "outputs/exports"
    os.makedirs(export_dir, exist_ok=True)
    
    filepath = os.path.join(export_dir, filename)
    
    # Export with proper formatting
    df.to_csv(filepath, index=False)
    
    logger.info(f"Exported {len(df)} patient records to {filepath}")
    
    return filepath

def validate_clinical_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate clinical data for consistency and plausibility
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'summary': {}
    }
    
    # Define clinical ranges
    clinical_ranges = {
        'age': (0, 120),
        'bmi': (10, 60),
        'systolic_bp': (60, 250),
        'diastolic_bp': (30, 150),
        'heart_rate': (30, 200),
        'glucose_level': (40, 600),
        'hba1c': (3.0, 20.0),
        'medication_adherence': (0.0, 1.0)
    }
    
    # Check each feature
    for feature, (min_val, max_val) in clinical_ranges.items():
        if feature in df.columns:
            out_of_range = df[(df[feature] < min_val) | (df[feature] > max_val)]
            
            if len(out_of_range) > 0:
                validation_results['warnings'].append(
                    f"{feature}: {len(out_of_range)} values out of clinical range ({min_val}-{max_val})"
                )
                
                # Critical errors
                if feature in ['systolic_bp', 'glucose_level'] and len(out_of_range) > len(df) * 0.1:
                    validation_results['errors'].append(
                        f"{feature}: >10% of values out of range - data quality issue"
                    )
                    validation_results['valid'] = False
    
    # Check for impossible combinations
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        impossible_bp = df[df['systolic_bp'] <= df['diastolic_bp']]
        if len(impossible_bp) > 0:
            validation_results['warnings'].append(
                f"Blood pressure: {len(impossible_bp)} patients with systolic ≤ diastolic"
            )
    
    # Summary statistics
    validation_results['summary'] = {
        'total_patients': len(df),
        'complete_records': len(df.dropna()),
        'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    return validation_results

# ---- Caching and Performance Functions ----
def cache_dashboard_data(df: pd.DataFrame, cache_path: str = "cache/dashboard_data.pkl") -> bool:
    """
    Cache processed dashboard data for faster loading
    
    Args:
        df: DataFrame to cache
        cache_path: Path to save cached data
        
    Returns:
        Success status
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_pickle(cache_path)
        logger.info(f"Dashboard data cached to {cache_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to cache data: {str(e)}")
        return False

def load_cached_dashboard_data(cache_path: str = "cache/dashboard_data.pkl", 
                              max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    """
    Load cached dashboard data if fresh enough
    
    Args:
        cache_path: Path to cached data
        max_age_hours: Maximum age of cache in hours
        
    Returns:
        Cached DataFrame or None if expired/missing
    """
    try:
        if not os.path.exists(cache_path):
            return None
        
        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        if cache_age.total_seconds() > max_age_hours * 3600:
            logger.info("Cache expired, will reload data")
            return None
        
        df = pd.read_pickle(cache_path)
        logger.info(f"Loaded {len(df)} patients from cache")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load cached data: {str(e)}")
        return None

if __name__ == "__main__":
    # Test utility functions
    print("🧪 Testing dashboard utilities...")
    
    # Generate test data
    test_df = generate_synthetic_patient_data(100)
    print(f"Generated {len(test_df)} test patients")
    
    # Calculate metrics
    metrics = calculate_risk_metrics(test_df)
    print(f"Risk metrics: {metrics}")
    
    # Validate data
    validation = validate_clinical_data(test_df)
    print(f"Validation: {validation['valid']}, {len(validation['warnings'])} warnings")
    
    print("✅ Utility functions working correctly")
