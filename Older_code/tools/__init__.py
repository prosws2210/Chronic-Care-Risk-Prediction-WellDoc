"""
Tools package for the Chronic Care Risk Prediction Engine.
Contains specialized CrewAI tools for healthcare data processing and ML operations.
"""

# Data tools
from .data_tools.patient_data_generator import PatientDataGeneratorTool
from .data_tools.vital_simulator import VitalSimulatorTool
from .data_tools.lab_result_simulator import LabResultSimulatorTool
from .data_tools.medication_adherence_tool import MedicationAdherenceTool

# ML tools
from .ml_tools.risk_prediction_model import RiskPredictionModelTool
from .ml_tools.feature_selector import FeatureSelectorTool
from .ml_tools.model_evaluator import ModelEvaluatorTool

# Health tools
from .health_tools.clinical_calculator import ClinicalCalculatorTool
from .health_tools.risk_scorer import RiskScorerTool
from .health_tools.deterioration_detector import DeteriorationDetectorTool

__all__ = [
    # Data tools
    'PatientDataGeneratorTool',
    'VitalSimulatorTool',
    'LabResultSimulatorTool',
    'MedicationAdherenceTool',
    
    # ML tools
    'RiskPredictionModelTool',
    'FeatureSelectorTool',
    'ModelEvaluatorTool',
    
    # Health tools
    'ClinicalCalculatorTool',
    'RiskScorerTool',
    'DeteriorationDetectorTool'
]

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"
