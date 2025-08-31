"""
Tasks Module for AI-Driven Risk Prediction Engine
=================================================

This module contains all CrewAI Task definitions for the chronic care risk prediction system.
Tasks are organized by functionality and designed to work with specialized healthcare agents.

Task Categories:
- Data Processing: Preprocessing, feature engineering, validation
- Model Development: Training, optimization, evaluation  
- Clinical Analysis: Explanation, validation, recommendations
- Visualization: Dashboard components, clinical charts

Usage:
    from tasks.prediction_tasks import create_prediction_tasks
    
    tasks = create_prediction_tasks(agents)
"""

from .prediction_tasks import (
    create_prediction_tasks,
    create_data_preprocessing_task,
    create_model_training_task,
    create_model_evaluation_task,
    create_explanation_task,
    create_clinical_validation_task,
    create_visualization_task
)

__all__ = [
    'create_prediction_tasks',
    'create_data_preprocessing_task', 
    'create_model_training_task',
    'create_model_evaluation_task',
    'create_explanation_task',
    'create_clinical_validation_task',
    'create_visualization_task'
]
