"""
Configuration Module for AI-Driven Risk Prediction Engine
========================================================

This module handles all configuration settings for the chronic care risk prediction system.
Supports multiple environments (development, production) with appropriate defaults.

Usage:
    from config import settings
    from config.settings import RISK_MODEL_NAME, BASE_URL
"""

from .settings import (
    # Model Configuration
    RISK_MODEL_NAME,
    EXPLANATION_MODEL_NAME,
    BASE_URL,
    API_KEY,
    TIMEOUT,
    MODEL_LOAD_TIMEOUT,
    
    # Clinical Configuration
    FEATURE_COLUMNS,
    CHRONIC_CONDITIONS,
    PREDICTION_WINDOW,
    LOOKBACK_WINDOW_MIN,
    LOOKBACK_WINDOW_MAX,
    
    # Thresholds
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
    MIN_AUROC,
    MIN_AUPRC,
    
    # Paths
    DATA_RAW_PATH,
    DATA_PROCESSED_PATH,
    MODELS_PATH,
    OUTPUTS_PATH,
    
    # Environment Management
    get_config,
    load_environment,
    get_current_environment
)

__all__ = [
    'RISK_MODEL_NAME',
    'EXPLANATION_MODEL_NAME', 
    'BASE_URL',
    'API_KEY',
    'TIMEOUT',
    'MODEL_LOAD_TIMEOUT',
    'FEATURE_COLUMNS',
    'CHRONIC_CONDITIONS',
    'PREDICTION_WINDOW',
    'LOOKBACK_WINDOW_MIN',
    'LOOKBACK_WINDOW_MAX',
    'RISK_THRESHOLD_HIGH',
    'RISK_THRESHOLD_MEDIUM',
    'MIN_AUROC',
    'MIN_AUPRC',
    'DATA_RAW_PATH',
    'DATA_PROCESSED_PATH',
    'MODELS_PATH',
    'OUTPUTS_PATH',
    'get_config',
    'load_environment',
    'get_current_environment'
]
