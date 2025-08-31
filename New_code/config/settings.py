"""
Settings Configuration for AI-Driven Risk Prediction Engine
==========================================================

Comprehensive configuration management with support for multiple environments.
Handles all settings for model training, clinical thresholds, and system parameters.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# ---- Environment Detection ----
CURRENT_ENVIRONMENT = os.getenv('ENVIRONMENT', 'development').lower()

def get_current_environment() -> str:
    """Get the current environment setting"""
    return CURRENT_ENVIRONMENT

# ---- Base Configuration Class ----
class BaseConfig:
    """Base configuration with common settings"""
    
    # ---- Clinical Feature Configuration ----
    FEATURE_COLUMNS = [
        # Demographics
        'age', 'gender',
        
        # Vitals & Physical Measurements  
        'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'weight', 'height',
        
        # Laboratory Values
        'glucose_level', 'hba1c', 'creatinine', 'bun', 
        'total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides',
        
        # Behavioral & Lifestyle
        'medication_adherence', 'exercise_frequency', 'smoking_status', 
        'alcohol_consumption', 'sleep_hours', 'stress_level',
        
        # Clinical History
        'previous_hospitalizations', 'emergency_visits', 'specialist_visits'
    ]
    
    # ---- Chronic Conditions ----
    CHRONIC_CONDITIONS = [
        'diabetes_type2', 'diabetes_type1', 'heart_failure', 'obesity', 
        'hypertension', 'copd', 'ckd', 'depression', 'anxiety'
    ]
    
    # ---- Clinical Parameters ----
    PREDICTION_WINDOW = 90  # days - predict deterioration within 90 days
    LOOKBACK_WINDOW_MIN = 30  # days - minimum historical data required
    LOOKBACK_WINDOW_MAX = 180  # days - maximum lookback for feature engineering
    
    # ---- Clinical Risk Thresholds ----
    RISK_THRESHOLD_HIGH = 0.7    # High risk threshold for immediate intervention
    RISK_THRESHOLD_MEDIUM = 0.3  # Medium risk threshold for monitoring
    RISK_THRESHOLD_LOW = 0.1     # Low risk threshold
    
    # ---- Model Performance Thresholds ----
    MIN_AUROC = 0.75            # Minimum acceptable AUROC for clinical deployment
    MIN_AUPRC = 0.65            # Minimum acceptable AUPRC for imbalanced data
    MIN_SENSITIVITY = 0.80      # Minimum sensitivity for patient safety
    MIN_SPECIFICITY = 0.75      # Minimum specificity to reduce alert fatigue
    MIN_PPV = 0.30              # Minimum positive predictive value
    MIN_NPV = 0.95              # Minimum negative predictive value
    
    # ---- Clinical Data Validation Ranges ----
    CLINICAL_RANGES = {
        'systolic_bp': (60, 300),      # mmHg
        'diastolic_bp': (30, 200),     # mmHg  
        'heart_rate': (30, 200),       # bpm
        'glucose_level': (40, 800),    # mg/dL
        'hba1c': (3.0, 20.0),         # %
        'bmi': (10.0, 80.0),          # kg/m²
        'age': (0, 120),              # years
        'creatinine': (0.1, 30.0),    # mg/dL
        'medication_adherence': (0.0, 1.0)  # proportion
    }
    
    # ---- Feature Engineering Configuration ----
    ROLLING_WINDOWS = [7, 14, 30, 60, 90]  # days for rolling averages
    LAG_FEATURES = [1, 7, 14, 30]          # days for lag features
    
    # ---- Missing Data Handling ----
    MISSING_DATA_STRATEGY = {
        'vitals': 'linear_interpolation',    # For BP, HR, weight
        'labs': 'locf_90days',              # Last observation carried forward
        'medications': 'assume_discontinued', # If missing > 30 days
        'lifestyle': 'population_median'     # Use population median by age/gender
    }

class DevelopmentConfig(BaseConfig):
    """Development environment configuration"""
    
    # ---- Model Configuration ----
    RISK_MODEL_NAME = "qwen/qwen3-14b"
    EXPLANATION_MODEL_NAME = "qwen/qwen3-8b"
    BASE_URL = "http://127.0.0.1:1234/v1"
    API_KEY = "your_api_key"
    TIMEOUT = 3600
    MODEL_LOAD_TIMEOUT = 300
    
    # ---- File Paths ----
    DATA_RAW_PATH = "data/raw/"
    DATA_PROCESSED_PATH = "data/processed/"
    MODELS_PATH = "models/saved/"
    OUTPUTS_PATH = "outputs/"
    LOGS_PATH = "logs/"
    
    # ---- Development Settings ----
    DEBUG = True
    VERBOSE_LOGGING = True
    SAMPLE_DATA_SIZE = 1000      # Use smaller dataset for faster development
    CROSS_VALIDATION_FOLDS = 3   # Fewer folds for faster training
    
    # ---- Model Training Settings ----
    MAX_TRAINING_TIME = 1800     # 30 minutes max for development
    EARLY_STOPPING_PATIENCE = 10
    HYPERPARAMETER_TRIALS = 20   # Fewer trials for faster optimization

class ProductionConfig(BaseConfig):
    """Production environment configuration"""
    
    # ---- Model Configuration ----
    RISK_MODEL_NAME = os.getenv("PROD_RISK_MODEL", "production/chronic_risk_model_v1")
    EXPLANATION_MODEL_NAME = os.getenv("PROD_EXPLANATION_MODEL", "production/explanation_model_v1")
    BASE_URL = os.getenv("PROD_API_BASE_URL", "https://api.healthcare-ai.com/v1")
    API_KEY = os.getenv("PROD_API_KEY", "")
    TIMEOUT = 1800
    MODEL_LOAD_TIMEOUT = 600
    
    # ---- Production File Paths ----
    DATA_RAW_PATH = os.getenv("PROD_DATA_RAW_PATH", "/data/healthcare/raw/")
    DATA_PROCESSED_PATH = os.getenv("PROD_DATA_PROCESSED_PATH", "/data/healthcare/processed/")
    MODELS_PATH = os.getenv("PROD_MODELS_PATH", "/models/production/")
    OUTPUTS_PATH = os.getenv("PROD_OUTPUTS_PATH", "/outputs/production/")
    LOGS_PATH = os.getenv("PROD_LOGS_PATH", "/logs/production/")
    
    # ---- Production Settings ----
    DEBUG = False
    VERBOSE_LOGGING = False
    
    # ---- Enhanced Thresholds for Production ----
    MIN_AUROC = 0.80            # Higher threshold for production
    MIN_AUPRC = 0.70
    MIN_SENSITIVITY = 0.85      # Higher sensitivity for production safety
    RISK_THRESHOLD_HIGH = 0.8   # More conservative high-risk threshold
    
    # ---- Production Training Settings ----
    MAX_TRAINING_TIME = 7200    # 2 hours max for production training
    CROSS_VALIDATION_FOLDS = 5  # More robust validation
    HYPERPARAMETER_TRIALS = 100 # More thorough optimization
    
    # ---- Security Settings ----
    ENCRYPT_MODEL_ARTIFACTS = True
    AUDIT_LOGGING = True
    HIPAA_COMPLIANCE_MODE = True
    
    # ---- Performance Monitoring ----
    PERFORMANCE_MONITORING = True
    DRIFT_DETECTION_THRESHOLD = 0.05
    RETRAINING_TRIGGER_THRESHOLD = 0.1

class TestingConfig(BaseConfig):
    """Testing environment configuration"""
    
    # ---- Model Configuration ----
    RISK_MODEL_NAME = "test/mock_risk_model"
    EXPLANATION_MODEL_NAME = "test/mock_explanation_model"
    BASE_URL = "http://localhost:8080/v1"
    API_KEY = "test_api_key"
    TIMEOUT = 30
    MODEL_LOAD_TIMEOUT = 60
    
    # ---- Test File Paths ----
    DATA_RAW_PATH = "tests/data/raw/"
    DATA_PROCESSED_PATH = "tests/data/processed/"
    MODELS_PATH = "tests/models/"
    OUTPUTS_PATH = "tests/outputs/"
    LOGS_PATH = "tests/logs/"
    
    # ---- Testing Settings ----
    DEBUG = True
    VERBOSE_LOGGING = False
    SAMPLE_DATA_SIZE = 100      # Very small dataset for testing
    CROSS_VALIDATION_FOLDS = 2  # Minimal folds for testing
    MAX_TRAINING_TIME = 60      # 1 minute max for testing

# ---- Configuration Selection ----
config_mapping = {
    'development': DevelopmentConfig,
    'dev': DevelopmentConfig,
    'production': ProductionConfig,
    'prod': ProductionConfig, 
    'testing': TestingConfig,
    'test': TestingConfig
}

def get_config(environment: Optional[str] = None) -> Any:
    """
    Get configuration class based on environment
    
    Args:
        environment: Environment name (development, production, testing)
        
    Returns:
        Configuration class instance
    """
    env = environment or CURRENT_ENVIRONMENT
    config_class = config_mapping.get(env.lower(), DevelopmentConfig)
    return config_class()

# ---- Load Current Configuration ----
current_config = get_config()

# ---- Export Configuration Variables ----
# Model Configuration
RISK_MODEL_NAME = current_config.RISK_MODEL_NAME
EXPLANATION_MODEL_NAME = current_config.EXPLANATION_MODEL_NAME
BASE_URL = current_config.BASE_URL
API_KEY = current_config.API_KEY
TIMEOUT = current_config.TIMEOUT
MODEL_LOAD_TIMEOUT = current_config.MODEL_LOAD_TIMEOUT

# Clinical Configuration
FEATURE_COLUMNS = current_config.FEATURE_COLUMNS
CHRONIC_CONDITIONS = current_config.CHRONIC_CONDITIONS
PREDICTION_WINDOW = current_config.PREDICTION_WINDOW
LOOKBACK_WINDOW_MIN = current_config.LOOKBACK_WINDOW_MIN
LOOKBACK_WINDOW_MAX = current_config.LOOKBACK_WINDOW_MAX

# Thresholds
RISK_THRESHOLD_HIGH = current_config.RISK_THRESHOLD_HIGH
RISK_THRESHOLD_MEDIUM = current_config.RISK_THRESHOLD_MEDIUM
RISK_THRESHOLD_LOW = getattr(current_config, 'RISK_THRESHOLD_LOW', 0.1)
MIN_AUROC = current_config.MIN_AUROC
MIN_AUPRC = current_config.MIN_AUPRC
MIN_SENSITIVITY = current_config.MIN_SENSITIVITY
MIN_SPECIFICITY = current_config.MIN_SPECIFICITY

# Paths
DATA_RAW_PATH = current_config.DATA_RAW_PATH
DATA_PROCESSED_PATH = current_config.DATA_PROCESSED_PATH
MODELS_PATH = current_config.MODELS_PATH
OUTPUTS_PATH = current_config.OUTPUTS_PATH
LOGS_PATH = getattr(current_config, 'LOGS_PATH', 'logs/')

# Clinical Ranges and Validation
CLINICAL_RANGES = current_config.CLINICAL_RANGES
ROLLING_WINDOWS = current_config.ROLLING_WINDOWS
LAG_FEATURES = current_config.LAG_FEATURES
MISSING_DATA_STRATEGY = current_config.MISSING_DATA_STRATEGY

# ---- Environment Management Functions ----
def load_environment(environment: str) -> Dict[str, Any]:
    """
    Load configuration for specific environment
    
    Args:
        environment: Environment name to load
        
    Returns:
        Dictionary of configuration values
    """
    config = get_config(environment)
    
    # Convert config object to dictionary
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_'):
            config_dict[attr] = getattr(config, attr)
    
    return config_dict

def validate_configuration() -> Dict[str, Any]:
    """
    Validate current configuration settings
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required API settings
    if not API_KEY or API_KEY == "your_api_key":
        validation_results['warnings'].append("API_KEY not set or using default value")
    
    # Check file paths exist
    for path_name, path_value in [
        ('DATA_RAW_PATH', DATA_RAW_PATH),
        ('DATA_PROCESSED_PATH', DATA_PROCESSED_PATH),
        ('MODELS_PATH', MODELS_PATH),
        ('OUTPUTS_PATH', OUTPUTS_PATH)
    ]:
        if not os.path.exists(path_value):
            validation_results['warnings'].append(f"{path_name} directory does not exist: {path_value}")
    
    # Check clinical thresholds
    if RISK_THRESHOLD_HIGH <= RISK_THRESHOLD_MEDIUM:
        validation_results['errors'].append("RISK_THRESHOLD_HIGH must be greater than RISK_THRESHOLD_MEDIUM")
        validation_results['valid'] = False
    
    # Check model performance thresholds
    if MIN_AUROC < 0.5 or MIN_AUROC > 1.0:
        validation_results['errors'].append("MIN_AUROC must be between 0.5 and 1.0")
        validation_results['valid'] = False
    
    return validation_results

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_RAW_PATH,
        DATA_PROCESSED_PATH, 
        MODELS_PATH,
        OUTPUTS_PATH,
        LOGS_PATH,
        f"{OUTPUTS_PATH}/figures",
        f"{OUTPUTS_PATH}/reports",
        f"{LOGS_PATH}/training",
        f"{LOGS_PATH}/inference"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_clinical_context() -> Dict[str, Any]:
    """
    Get clinical context and reference information
    
    Returns:
        Dictionary with clinical guidelines and references
    """
    return {
        'clinical_guidelines': {
            'diabetes': {
                'hba1c_target': '<7.0% for most adults',
                'glucose_target': '80-130 mg/dL preprandial',
                'reference': 'ADA Standards of Medical Care 2024'
            },
            'hypertension': {
                'bp_target': '<130/80 mmHg for most adults', 
                'high_risk_threshold': '≥140/90 mmHg',
                'reference': 'ACC/AHA 2017 Guidelines'
            },
            'heart_failure': {
                'ejection_fraction': '<40% (HFrEF), 40-49% (HFmrEF), ≥50% (HFpEF)',
                'natriuretic_peptides': 'BNP >400 pg/mL or NT-proBNP >2000 pg/mL',
                'reference': 'AHA/ACC/HFSA 2022 Guidelines'
            }
        },
        'risk_factors': {
            'modifiable': ['blood_pressure', 'glucose_control', 'medication_adherence', 'lifestyle'],
            'non_modifiable': ['age', 'gender', 'genetic_factors', 'medical_history']
        },
        'intervention_priorities': [
            'medication_optimization',
            'lifestyle_modification', 
            'monitoring_intensification',
            'specialist_referral'
        ]
    }

# ---- Environment Variable Setup ----
# Set OpenAI API key for CrewAI compatibility
os.environ["OPENAI_API_KEY"] = API_KEY if API_KEY and API_KEY != "your_api_key" else "dummy-key"

# Create necessary directories on import
create_directories()

# ---- Configuration Summary ----
def print_config_summary():
    """Print summary of current configuration"""
    print(f"""
🏥 AI-DRIVEN RISK PREDICTION ENGINE - CONFIGURATION
{'='*60}
Environment: {CURRENT_ENVIRONMENT.upper()}
Model: {RISK_MODEL_NAME}
API Base: {BASE_URL}
Data Path: {DATA_PROCESSED_PATH}
Models Path: {MODELS_PATH}

Clinical Settings:
- Prediction Window: {PREDICTION_WINDOW} days
- Risk Thresholds: High={RISK_THRESHOLD_HIGH}, Medium={RISK_THRESHOLD_MEDIUM}
- Min Performance: AUROC≥{MIN_AUROC}, AUPRC≥{MIN_AUPRC}
- Features: {len(FEATURE_COLUMNS)} clinical features
- Conditions: {len(CHRONIC_CONDITIONS)} chronic conditions
{'='*60}
    """)

if __name__ == "__main__":
    print_config_summary()
    
    # Validate configuration
    validation = validate_configuration()
    if validation['valid']:
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed:")
        for error in validation['errors']:
            print(f"   - {error}")
    
    if validation['warnings']:
        print("⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")
