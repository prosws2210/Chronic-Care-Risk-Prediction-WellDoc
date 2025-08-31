"""
Specialist agents for domain-specific medical expertise.
Provides condition-specific knowledge for chronic care management.
"""

from .diabetes_specialist import DiabetesSpecialistAgent
from .cardiology_specialist import CardiologySpecialistAgent
from .obesity_specialist import ObesitySpecialistAgent

__all__ = [
    'DiabetesSpecialistAgent',
    'CardiologySpecialistAgent', 
    'ObesitySpecialistAgent'
]
