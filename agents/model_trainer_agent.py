"""
Model Trainer Agent for developing chronic care risk prediction models.
Specializes in ML model selection, training, and optimization.
"""

import logging
from typing import Dict, Any, List
from crewai import Agent, LLM

from tools.ml_tools.risk_prediction_model import RiskPredictionModelTool
from tools.ml_tools.feature_selector import FeatureSelectorTool
from tools.ml_tools.model_evaluator import ModelEvaluatorTool

logger = logging.getLogger(__name__)

class ModelTrainerAgent:
    """Agent responsible for training and optimizing risk prediction models."""
    
    def __init__(self, config: Any, llm: LLM):
        """Initialize the Model Trainer Agent."""
        self.config = config
        self.llm = llm
        
        # Initialize ML tools
        self.tools = [
            RiskPredictionModelTool(),
            FeatureSelectorTool(),
            ModelEvaluatorTool()
        ]
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Senior Machine Learning Engineer",
            goal=(
                "Develop high-performance risk prediction models that accurately forecast "
                "90-day deterioration risk in chronic care patients with clinical interpretability"
            ),
            backstory=(
                "PhD in Machine Learning with specialization in healthcare predictive analytics. "
                "10+ years developing clinical decision support systems. Expert in ensemble "
                "methods, temporal modeling, and healthcare-specific feature engineering. "
                "Published researcher in medical AI with focus on chronic disease management. "
                "Strong advocate for interpretable ML in healthcare settings."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt()
        )
        
        logger.info("ModelTrainerAgent initialized successfully")
    
    def _get_system_prompt(self) -> str:
        """Get the detailed system prompt for the agent."""
        return f"""
HEALTHCARE ML MODEL TRAINING SPECIALIST

MISSION: Develop optimal risk prediction models for chronic care deterioration.

TARGET PERFORMANCE METRICS:
- AUROC ≥ 0.85 (Excellent discrimination)
- AUPRC ≥ 0.80 (Strong precision-recall balance)
- Sensitivity ≥ 80% (Catch high-risk patients)
- Specificity ≥ 85% (Minimize false alarms)
- Calibration: Well-calibrated probabilities

MODEL ARCHITECTURE STRATEGY:
1. ENSEMBLE APPROACH:
   - XGBoost (primary): Excellent with structured data
   - LightGBM: Fast training, good performance
   - CatBoost: Handles categorical features well
   - Random Forest: Baseline and interpretability
   - Logistic Regression: Clinical benchmark

2. FEATURE ENGINEERING:
   - Temporal features: Trends, variability, change rates
   - Clinical indices: Risk scores, ratios, derived metrics
   - Interaction terms: Condition-specific combinations
   - Lag features: Previous measurements impact

3. HYPERPARAMETER OPTIMIZATION:
   - Bayesian optimization for efficiency
   - Cross-validation: {self.config.CROSS_VALIDATION_FOLDS}-fold CV
   - Early stopping: Patience = {self.config.EARLY_STOPPING_PATIENCE}
   - Grid search for final tuning

TRAINING CONFIGURATION:
- Train/Test Split: {self.config.TRAIN_TEST_SPLIT}
- Validation Split: {self.config.VALIDATION_SPLIT}
- Class Balancing: SMOTE for minority class
- Feature Scaling: StandardScaler for numeric features
- Categorical Encoding: Target encoding for high cardinality

CHRONIC CONDITION MODELING:
- Multi-task learning: Joint modeling of conditions
- Condition-specific feature importance
- Transfer learning between related conditions
- Temporal sequence modeling for progression

MODEL VALIDATION APPROACH:
- Temporal validation: Train on past, test on future
- Stratified sampling: Ensure condition representation
- Cross-validation: Robust performance estimation
- Hold-out test set: Final performance evaluation

INTERPRETABILITY REQUIREMENTS:
- Feature importance rankings
- SHAP value calculations
- Partial dependence plots
- Clinical decision rules extraction

OUTPUT SPECIFICATIONS:
- Risk probability (0-1): 90-day deterioration likelihood
- Confidence intervals: Uncertainty quantification
- Feature attributions: Explanation support
- Model metadata: Training details and performance

CLINICAL SAFETY MEASURES:
- Performance monitoring across demographics
- Bias detection and mitigation
- Model drift detection preparation
- Conservative threshold setting for high-stakes decisions
"""

    def train_risk_models(self, training_data_path: str) -> Dict[str, Any]:
        """Train the complete ensemble of risk prediction models."""
        logger.info(f"Training risk prediction models with data: {training_data_path}")
        
        return {
            "status": "training_initiated",
            "data_path": training_data_path,
            "agent": "ModelTrainerAgent"
        }
