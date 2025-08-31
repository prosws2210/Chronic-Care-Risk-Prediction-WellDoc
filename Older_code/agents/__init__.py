"""
Agents package for the Chronic Care Risk Prediction Engine.
Contains all CrewAI agent implementations for healthcare AI.
"""

from .data_generator_agent import DataGeneratorAgent
from .model_trainer_agent import ModelTrainerAgent
from .model_evaluator_agent import ModelEvaluatorAgent
from .explainability_agent import ExplainabilityAgent
from .clinical_validator_agent import ClinicalValidatorAgent
from .risk_assessor_agent import RiskAssessorAgent

# Specialist agents
from .specialists.diabetes_specialist import DiabetesSpecialistAgent
from .specialists.cardiology_specialist import CardiologySpecialistAgent
from .specialists.obesity_specialist import ObesitySpecialistAgent

# Evaluator agents
from .evaluators.performance_evaluator import PerformanceEvaluatorAgent
from .evaluators.bias_detector import BiasDetectorAgent
from .evaluators.clinical_safety_evaluator import ClinicalSafetyEvaluatorAgent

__all__ = [
    # Core agents
    'DataGeneratorAgent',
    'ModelTrainerAgent',
    'ModelEvaluatorAgent',
    'ExplainabilityAgent',
    'ClinicalValidatorAgent',
    'RiskAssessorAgent',
    
    # Specialist agents
    'DiabetesSpecialistAgent',
    'CardiologySpecialistAgent',
    'ObesitySpecialistAgent',
    
    # Evaluator agents
    'PerformanceEvaluatorAgent',
    'BiasDetectorAgent',
    'ClinicalSafetyEvaluatorAgent'
]

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"
