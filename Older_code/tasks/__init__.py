"""
Tasks package for the Chronic Care Risk Prediction Engine.
Contains all CrewAI task definitions for healthcare AI workflows.
"""

# Generation tasks
from .generation.synthetic_data_task import SyntheticDataTask
from .generation.feature_engineering_task import FeatureEngineeringTask

# Training tasks
from .training.model_training_task import ModelTrainingTask
from .training.hyperparameter_tuning_task import HyperparameterTuningTask

# Evaluation tasks
from .evaluation.model_evaluation_task import ModelEvaluationTask
from .evaluation.clinical_validation_task import ClinicalValidationTask

# Explanation tasks
from .explanation.global_explanation_task import GlobalExplanationTask
from .explanation.local_explanation_task import LocalExplanationTask

__all__ = [
    # Generation tasks
    'SyntheticDataTask',
    'FeatureEngineeringTask',
    
    # Training tasks
    'ModelTrainingTask',
    'HyperparameterTuningTask',
    
    # Evaluation tasks
    'ModelEvaluationTask',
    'ClinicalValidationTask',
    
    # Explanation tasks
    'GlobalExplanationTask',
    'LocalExplanationTask'
]

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"
