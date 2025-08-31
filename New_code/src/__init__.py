"""
AI-Driven Risk Prediction Engine for Chronic Care Patients
=========================================================

This package contains the main execution pipeline for predicting patient 
deterioration risk using CrewAI agents, machine learning models, and 
Streamlit dashboard.

Author: Healthcare AI Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"

# Import main components
from .main import RiskPredictionPipeline, main

__all__ = [
    'RiskPredictionPipeline',
    'main'
]
