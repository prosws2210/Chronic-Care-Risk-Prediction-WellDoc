"""
Dashboard package for Chronic Care Risk Prediction Engine.
Web-based dashboard for monitoring patient cohorts, risk assessments, and clinical insights.
"""

from .app import create_app
from .cohort_view import CohortView
from .patient_detail_view import PatientDetailView
from .risk_dashboard import RiskDashboard

__all__ = [
    'create_app',
    'CohortView',
    'PatientDetailView', 
    'RiskDashboard'
]

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"
