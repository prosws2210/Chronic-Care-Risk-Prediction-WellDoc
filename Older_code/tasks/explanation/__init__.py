"""
Explanation tasks for model interpretability and clinical insight generation.
Provides global and local explanations for chronic care risk prediction models.
"""

from .global_explanation_task import GlobalExplanationTask
from .local_explanation_task import LocalExplanationTask

__all__ = [
    'GlobalExplanationTask',
    'LocalExplanationTask'
]

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"
