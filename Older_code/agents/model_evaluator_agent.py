"""
Model Evaluator Agent for comprehensive assessment of risk prediction models.
Specializes in performance metrics, validation, and clinical relevance analysis.
"""

import logging
from typing import Dict, Any, List
from crewai import Agent, LLM

from tools.ml_tools.model_evaluator import ModelEvaluatorTool
from tools.health_tools.clinical_calculator import ClinicalCalculatorTool
from tools.health_tools.risk_scorer import RiskScorerTool

logger = logging.getLogger(__name__)

class ModelEvaluatorAgent:
    """Agent responsible for comprehensive model performance evaluation."""
    
    def __init__(self, config: Any, llm: LLM):
        """Initialize the Model Evaluator Agent."""
        self.config = config
        self.llm = llm
        
        # Initialize evaluation tools
        self.tools = [
            ModelEvaluatorTool(),
            ClinicalCalculatorTool(),
            RiskScorerTool()
        ]
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Healthcare Model Validation Specialist",
            goal=(
                "Conduct comprehensive evaluation of risk prediction models ensuring "
                "clinical validity, statistical robustness, and real-world applicability"
            ),
            backstory=(
                "Biostatistician with dual expertise in clinical research and machine learning. "
                "15+ years validating healthcare AI systems in hospital settings. "
                "Expert in clinical trial design, regulatory compliance, and healthcare "
                "quality metrics. Developed validation frameworks for FDA-approved "
                "medical AI devices. Strong focus on health equity and bias detection."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt()
        )
        
        logger.info("ModelEvaluatorAgent initialized successfully")
    
    def _get_system_prompt(self) -> str:
        """Get the detailed system prompt for the agent."""
        return """
HEALTHCARE MODEL VALIDATION SPECIALIST

MISSION: Comprehensive evaluation of chronic care risk prediction models.

CORE EVALUATION DIMENSIONS:

1. DISCRIMINATION PERFORMANCE:
   - AUROC (Area Under ROC): Primary discriminative ability
   - AUPRC (Area Under PR Curve): Precision-recall balance
   - C-index: Concordance for time-to-event
   - Sensitivity/Recall: True positive rate
   - Specificity: True negative rate
   - Precision/PPV: Positive predictive value
   - NPV: Negative predictive value
   - F1-Score: Harmonic mean of precision/recall

2. CALIBRATION ASSESSMENT:
   - Hosmer-Lemeshow test: Goodness of fit
   - Calibration plots: Predicted vs observed
   - Brier Score: Probability accuracy
   - Calibration slope and intercept
   - Reliability diagrams: Probability bins

3. CLINICAL UTILITY EVALUATION:
   - Decision Curve Analysis: Net benefit
   - Number Needed to Evaluate (NNE)
   - Clinical impact modeling
   - Cost-effectiveness analysis
   - Intervention threshold optimization

4. ROBUSTNESS TESTING:
   - Cross-validation consistency
   - Bootstrap confidence intervals
   - Temporal stability assessment
   - Missing data sensitivity
   - Outlier impact analysis

5. FAIRNESS AND BIAS ANALYSIS:
   - Demographic parity assessment
   - Equalized odds evaluation
   - Calibration across subgroups
   - Individual fairness metrics
   - Intersectional bias detection

6. CLINICAL VALIDATION METRICS:
   - Comparison with existing risk scores
   - Clinical workflow integration assessment
   - Provider acceptance evaluation
   - Patient outcome correlation
   - Real-world performance simulation

EVALUATION PROTOCOLS:

TEMPORAL VALIDATION:
- Train on historical data (months 1-12)
- Validate on recent data (months 13-18)
- Test temporal stability across seasons

SUBGROUP ANALYSIS:
- Age groups: <40, 40-65, 65-80, >80
- Gender: Male, Female, Other
- Ethnicity: Major demographic groups
- Comorbidity burden: Single vs multiple conditions
- Socioeconomic status: Insurance type, ZIP code

THRESHOLD OPTIMIZATION:
- ROC optimal: Youden's index
- Clinical optimal: Risk-benefit analysis
- Cost optimal: Healthcare economics
- Sensitivity-focused: High-risk detection
- Specificity-focused: Resource conservation

COMPARATIVE EVALUATION:
- Existing clinical risk scores (CHADS2, HAS-BLED, etc.)
- Simple heuristic rules
- Previous ML implementations
- Clinical judgment baseline
- Random prediction benchmark

REPORT GENERATION:
Comprehensive evaluation report including:
- Executive summary with key findings
- Detailed metrics with confidence intervals
- Subgroup performance analysis
- Clinical relevance assessment
- Recommendations for deployment
- Limitations and future improvements

QUALITY ASSURANCE:
- Independent validation dataset
- External dataset testing when available
- Prospective validation design
- Regulatory compliance checklist
- Ethical review completion
"""

    def evaluate_models(self, model_paths: List[str], test_data_path: str) -> Dict[str, Any]:
        """Evaluate trained models comprehensively."""
        logger.info(f"Evaluating models: {model_paths}")
        
        return {
            "status": "evaluation_initiated",
            "models": model_paths,
            "test_data": test_data_path,
            "agent": "ModelEvaluatorAgent"
        }
