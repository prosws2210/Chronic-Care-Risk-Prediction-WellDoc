"""
Configuration management for the Chronic Care Risk Prediction Engine.
Handles environment variables, configuration files, and default settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class for the chronic care risk prediction system."""
    
    # Project paths
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # AI Model Configuration
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    LM_STUDIO_BASE_URL: str = field(default_factory=lambda: os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    LM_STUDIO_API_KEY: str = field(default_factory=lambda: os.getenv("LM_STUDIO_API_KEY", "lm-studio"))
    
    # Model Settings
    PRIMARY_MODEL: str = field(default_factory=lambda: os.getenv("PRIMARY_MODEL", "gpt-4"))
    FALLBACK_MODEL: str = field(default_factory=lambda: os.getenv("FALLBACK_MODEL", "gpt-3.5-turbo"))
    MAX_TOKENS: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4000")))
    TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    
    # Database Configuration
    DATABASE_URL: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///chronic_care.db"))
    MONGODB_URL: str = field(default_factory=lambda: os.getenv("MONGODB_URL", "mongodb://localhost:27017/chronic_care"))
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    
    # Dashboard Configuration
    DASHBOARD_HOST: str = field(default_factory=lambda: os.getenv("DASHBOARD_HOST", "0.0.0.0"))
    DASHBOARD_PORT: int = field(default_factory=lambda: int(os.getenv("DASHBOARD_PORT", "8080")))
    DEBUG_MODE: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "True").lower() == "true")
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key"))
    
    # Data Configuration
    SYNTHETIC_PATIENTS_COUNT: int = field(default_factory=lambda: int(os.getenv("SYNTHETIC_PATIENTS_COUNT", "10000")))
    DATA_GENERATION_SEED: int = field(default_factory=lambda: int(os.getenv("DATA_GENERATION_SEED", "42")))
    MIN_HISTORY_DAYS: int = field(default_factory=lambda: int(os.getenv("MIN_HISTORY_DAYS", "30")))
    MAX_HISTORY_DAYS: int = field(default_factory=lambda: int(os.getenv("MAX_HISTORY_DAYS", "180")))
    PREDICTION_WINDOW_DAYS: int = field(default_factory=lambda: int(os.getenv("PREDICTION_WINDOW_DAYS", "90")))
    
    # Model Training Configuration
    TRAIN_TEST_SPLIT: float = field(default_factory=lambda: float(os.getenv("TRAIN_TEST_SPLIT", "0.8")))
    VALIDATION_SPLIT: float = field(default_factory=lambda: float(os.getenv("VALIDATION_SPLIT", "0.2")))
    CROSS_VALIDATION_FOLDS: int = field(default_factory=lambda: int(os.getenv("CROSS_VALIDATION_FOLDS", "5")))
    EARLY_STOPPING_PATIENCE: int = field(default_factory=lambda: int(os.getenv("EARLY_STOPPING_PATIENCE", "10")))
    
    # Logging Configuration
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FILE_PATH: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "logs/chronic_care.log"))
    LOG_MAX_FILE_SIZE: str = field(default_factory=lambda: os.getenv("LOG_MAX_FILE_SIZE", "10MB"))
    LOG_BACKUP_COUNT: int = field(default_factory=lambda: int(os.getenv("LOG_BACKUP_COUNT", "5")))
    
    # Security Configuration
    ENCRYPTION_KEY: str = field(default_factory=lambda: os.getenv("ENCRYPTION_KEY", ""))
    SESSION_TIMEOUT_MINUTES: int = field(default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")))
    MAX_LOGIN_ATTEMPTS: int = field(default_factory=lambda: int(os.getenv("MAX_LOGIN_ATTEMPTS", "3")))
    
    # Feature Flags
    ENABLE_SYNTHETIC_DATA: bool = field(default_factory=lambda: os.getenv("ENABLE_SYNTHETIC_DATA", "True").lower() == "true")
    ENABLE_MODEL_EXPLANATIONS: bool = field(default_factory=lambda: os.getenv("ENABLE_MODEL_EXPLANATIONS", "True").lower() == "true")
    ENABLE_BIAS_DETECTION: bool = field(default_factory=lambda: os.getenv("ENABLE_BIAS_DETECTION", "True").lower() == "true")
    ENABLE_REAL_TIME_PREDICTIONS: bool = field(default_factory=lambda: os.getenv("ENABLE_REAL_TIME_PREDICTIONS", "True").lower() == "true")
    
    # Additional Configuration
    config_file: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Set up directory paths
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.OUTPUTS_DIR = self.PROJECT_ROOT / "outputs"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        self.CONFIG_DIR = self.PROJECT_ROOT / "config"
        
        # Load additional configuration from file if specified
        if self.config_file:
            self.load_config_file(self.config_file)
    
    def load_config_file(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            config_file_path = Path(config_path)
            if not config_file_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update configuration with file data
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {str(e)}")
    
    def save_config_file(self, config_path: str):
        """Save current configuration to YAML file."""
        try:
            config_data = {}
            for key, value in self.__dict__.items():
                if not key.startswith('_') and not isinstance(value, Path):
                    config_data[key] = value
            
            config_file_path = Path(config_path)
            config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            raise ValueError(f"Failed to save configuration file: {str(e)}")
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate required API keys
        if not self.OPENAI_API_KEY and not self.ANTHROPIC_API_KEY:
            validation_results["errors"].append("No AI API keys configured")
            validation_results["valid"] = False
        
        # Validate data generation parameters
        if self.SYNTHETIC_PATIENTS_COUNT < 100:
            validation_results["warnings"].append("Low patient count may affect model performance")
        
        if self.MIN_HISTORY_DAYS >= self.MAX_HISTORY_DAYS:
            validation_results["errors"].append("MIN_HISTORY_DAYS must be less than MAX_HISTORY_DAYS")
            validation_results["valid"] = False
        
        # Validate model training parameters
        if not 0 < self.TRAIN_TEST_SPLIT < 1:
            validation_results["errors"].append("TRAIN_TEST_SPLIT must be between 0 and 1")
            validation_results["valid"] = False
        
        if not 0 < self.VALIDATION_SPLIT < 1:
            validation_results["errors"].append("VALIDATION_SPLIT must be between 0 and 1")
            validation_results["valid"] = False
        
        # Validate directory paths
        required_dirs = [self.DATA_DIR, self.MODELS_DIR, self.OUTPUTS_DIR, self.LOGS_DIR]
        for dir_path in required_dirs:
            if not dir_path.exists():
                validation_results["warnings"].append(f"Directory does not exist: {dir_path}")
        
        return validation_results
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            "primary_model": self.PRIMARY_MODEL,
            "fallback_model": self.FALLBACK_MODEL,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
            "api_keys": {
                "openai": bool(self.OPENAI_API_KEY),
                "anthropic": bool(self.ANTHROPIC_API_KEY)
            }
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-specific configuration."""
        return {
            "synthetic_patients": self.SYNTHETIC_PATIENTS_COUNT,
            "history_range": (self.MIN_HISTORY_DAYS, self.MAX_HISTORY_DAYS),
            "prediction_window": self.PREDICTION_WINDOW_DAYS,
            "seed": self.DATA_GENERATION_SEED
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            "train_test_split": self.TRAIN_TEST_SPLIT,
            "validation_split": self.VALIDATION_SPLIT,
            "cv_folds": self.CROSS_VALIDATION_FOLDS,
            "early_stopping_patience": self.EARLY_STOPPING_PATIENCE
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ChronicCareConfig(patients={self.SYNTHETIC_PATIENTS_COUNT}, model={self.PRIMARY_MODEL})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()


# Global configuration instance
config = Config()

# Export commonly used paths
DATA_DIR = config.DATA_DIR
MODELS_DIR = config.MODELS_DIR
OUTPUTS_DIR = config.OUTPUTS_DIR
LOGS_DIR = config.LOGS_DIR
