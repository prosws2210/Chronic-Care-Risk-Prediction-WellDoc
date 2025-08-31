"""
Model Training Task for developing chronic care risk prediction models.
Orchestrates the training of ensemble ML models with clinical validation.
"""

import logging
from typing import Dict, Any
from crewai import Task

logger = logging.getLogger(__name__)

class ModelTrainingTask:
    """Task for training risk prediction models."""
    
    def __init__(self, config: Any, agents: Dict[str, Any]):
        """Initialize the model training task."""
        self.config = config
        self.agents = agents
        
        # Create the CrewAI task
        self.task = Task(
            description=self._get_task_description(),
            expected_output=self._get_expected_output(),
            agent=self.agents['model_trainer'].agent,
            tools=self.agents['model_trainer'].agent.tools
        )
        
        logger.info("ModelTrainingTask initialized")
    
    def _get_task_description(self) -> str:
        """Get comprehensive task description."""
        return f"""
        Train and optimize machine learning models for predicting 90-day deterioration risk 
        in chronic care patients using the engineered feature dataset.
        
        **PRIMARY OBJECTIVE:**
        Develop high-performance ensemble models that achieve clinical-grade accuracy 
        (AUROC ≥ 0.85) while maintaining interpretability for healthcare decision support.
        
        **MODEL DEVELOPMENT STRATEGY:**
        
        1. **ENSEMBLE ARCHITECTURE:**
           - Primary Models: XGBoost, LightGBM, Random Forest, Extra Trees
           - Meta-Learner: Calibrated ensemble with soft voting
           - Baseline: Logistic Regression for interpretability comparison
           - Validation: {self.config.CROSS_VALIDATION_FOLDS}-fold cross-validation
        
        2. **TRAINING CONFIGURATION:**
           - Train/Test Split: {self.config.TRAIN_TEST_SPLIT}/{1-self.config.TRAIN_TEST_SPLIT}
           - Validation Split: {self.config.VALIDATION_SPLIT} of training data
           - Class Balancing: SMOTE oversampling for minority class
           - Feature Scaling: StandardScaler for distance-based algorithms
           - Random State: 42 for reproducibility
        
        3. **HYPERPARAMETER OPTIMIZATION:**
           - Method: Bayesian optimization with cross-validation
           - Objective: Maximize AUROC with calibration constraint
           - Search Space: Model-specific parameter grids
           - Early Stopping: Patience = {self.config.EARLY_STOPPING_PATIENCE} epochs
           - Resource Management: Parallel processing when available
        
        4. **CLINICAL OPTIMIZATION:**
           - Threshold Selection: Optimize for clinical utility
           - Cost-Sensitive Learning: Weight false negatives appropriately
           - Temporal Validation: Ensure stability over time periods
           - Subgroup Analysis: Validate across demographic groups
        
        **FEATURE ENGINEERING INTEGRATION:**
        - Automated feature selection during training
        - Recursive feature elimination for interpretability
        - Clinical domain knowledge constraints
        - Feature importance analysis and ranking
        
        **MODEL VALIDATION PROTOCOLS:**
        - Temporal holdout validation (train on past, test on future)
        - Stratified sampling to maintain class distribution
        - Cross-validation with medical center stratification
        - External validation readiness assessment
        
        **PERFORMANCE TARGETS:**
        - AUROC: ≥ 0.85 (Excellent discrimination)
        - AUPRC: ≥ 0.80 (Strong precision-recall balance)  
        - Sensitivity: ≥ 80% (Minimize missed high-risk patients)
        - Specificity: ≥ 85% (Minimize false alarms)
        - Calibration: Well-calibrated probability outputs
        - Stability: Consistent performance across CV folds
        
        **CLINICAL INTERPRETABILITY:**
        - SHAP value computation for feature explanations
        - Decision tree extraction from ensemble models
        - Clinical pathway analysis
        - Risk factor ranking and contribution analysis
        
        Collaborate with medical specialist agents to ensure clinical validity and 
        domain expert validation throughout the training process.
        """
    
    def _get_expected_output(self) -> str:
        """Get expected output specification."""
        return f"""
        TRAINED RISK PREDICTION MODEL ENSEMBLE:
        
        **MODEL ARTIFACTS:**
        ```
        models/
        ├── best_ensemble_model.pkl          # Final calibrated ensemble
        ├── individual_models/
        │   ├── xgboost_model.pkl
        │   ├── lightgbm_model.pkl
        │   ├── random_forest_model.pkl
        │   └── logistic_regression_model.pkl
        ├── preprocessing_pipeline.pkl        # Feature preprocessing
        ├── feature_selector.pkl             # Selected features
        └── model_metadata.json             # Training configuration
        ```
        
        **PERFORMANCE METRICS:**
        ```
        {{
            "ensemble_performance": {{
                "roc_auc": 0.872,
                "average_precision": 0.834,
                "accuracy": 0.831,
                "sensitivity": 0.847,
                "specificity": 0.868,
                "f1_score": 0.758,
                "brier_score": 0.145,
                "calibration_slope": 0.987
            }},
            "cross_validation_results": {{
                "mean_roc_auc": 0.869,
                "std_roc_auc": 0.012,
                "cv_folds": {self.config.CROSS_VALIDATION_FOLDS},
                "consistency_score": 0.94
            }},
            "clinical_metrics": {{
                "number_needed_to_evaluate": 4.2,
                "false_negative_rate": 0.153,
                "false_positive_rate": 0.132,
                "positive_predictive_value": 0.745,
                "negative_predictive_value": 0.895
            }}
        }}
        ```
        
        **MODEL COMPARISON:**
        | Model | AUROC | AUPRC | Sensitivity | Specificity | Training Time |
        |-------|-------|-------|-------------|-------------|---------------|
        | Ensemble | 0.872 | 0.834 | 84.7% | 86.8% | 45 min |
        | XGBoost | 0.865 | 0.828 | 83.2% | 85.1% | 12 min |
        | LightGBM | 0.859 | 0.821 | 82.8% | 84.9% | 8 min |
        | Random Forest | 0.851 | 0.815 | 81.5% | 83.7% | 15 min |
        | Logistic Regression | 0.798 | 0.742 | 76.3% | 79.8% | 2 min |
        
        **FEATURE IMPORTANCE (TOP 10):**
        1. `hba1c_trend_90d`: 0.124 - HbA1c trend over 90 days
        2. `hospitalization_recency`: 0.098 - Days since last hospitalization  
        3. `medication_adherence_score`: 0.087 - Composite adherence metric
        4. `comorbidity_burden_index`: 0.074 - Weighted comorbidity score
        5. `age_risk_category`: 0.069 - Age-based risk stratification
        6. `bp_variability_30d`: 0.063 - Blood pressure variability
        7. `egfr_decline_rate`: 0.059 - Kidney function decline rate
        8. `emergency_visits_12m`: 0.055 - Emergency visits in 12 months
        9. `medication_count`: 0.051 - Total active medications
        10. `diabetes_duration_years`: 0.048 - Years since diabetes diagnosis
        
        **CLINICAL DECISION THRESHOLDS:**
        - High Risk (≥70%): Immediate clinical intervention required
        - Medium Risk (30-69%): Enhanced monitoring and care coordination
        - Low Risk (<30%): Standard care with routine follow-up
        
        **DEPLOYMENT READINESS:**
        ✓ Model performance meets clinical requirements (AUROC ≥ 0.85)
        ✓ Calibrated probabilities for reliable risk estimation
        ✓ Feature importance aligns with clinical expertise
        ✓ Robust performance across demographic subgroups
        ✓ Interpretable predictions with SHAP explanations
        ✓ Comprehensive documentation and metadata
        ✓ Serialized models ready for production deployment
        
        **TRAINING REPORT:**
        - `model_training_report.md`: Comprehensive training documentation
        - `hyperparameter_optimization_log.json`: Optimization history
        - `cross_validation_results.csv`: Detailed CV performance
        - `feature_selection_analysis.json`: Feature selection rationale
        - `clinical_validation_summary.md`: Medical expert review
        """


class HyperparameterTuningTask:
    """Task for optimizing model hyperparameters."""
    
    def __init__(self, config: Any, agents: Dict[str, Any]):
        """Initialize the hyperparameter tuning task."""
        self.config = config
        self.agents = agents
        
        # Create the CrewAI task
        self.task = Task(
            description=self._get_task_description(),
            expected_output=self._get_expected_output(),
            agent=self.agents['model_trainer'].agent,
            tools=self.agents['model_trainer'].agent.tools
        )
        
        logger.info("HyperparameterTuningTask initialized")
    
    def _get_task_description(self) -> str:
        """Get comprehensive task description."""
        return """
        Systematically optimize hyperparameters for chronic care risk prediction models 
        to achieve maximum performance while preventing overfitting.
        
        **PRIMARY OBJECTIVE:**
        Find optimal hyperparameter configurations that maximize model performance 
        on clinical metrics while maintaining generalizability and stability.
        
        **OPTIMIZATION STRATEGY:**
        
        1. **SEARCH METHODOLOGY:**
           - Bayesian Optimization with Gaussian Process surrogate
           - Multi-objective optimization (AUROC + Calibration)
           - Early stopping to prevent overfitting
           - Cross-validation for robust evaluation
        
        2. **HYPERPARAMETER SPACES:**
        
           **XGBoost Parameters:**
           - n_estimators: [100, 200, 300, 500]
           - max_depth: [3, 4, 5, 6, 7]
           - learning_rate: [0.01, 0.05, 0.1, 0.2]
           - subsample: [0.8, 0.9, 1.0]
           - colsample_bytree: [0.8, 0.9, 1.0]
           - reg_alpha: [0, 0.1, 0.5, 1.0]
           - reg_lambda: [0, 0.1, 0.5, 1.0]
        
           **Random Forest Parameters:**
           - n_estimators: [100, 200, 300, 500]
           - max_depth: [10, 20, 30, None]
           - min_samples_split: [2, 5, 10]
           - min_samples_leaf: [1, 2, 4]
           - max_features: ['sqrt', 'log2', None]
           - bootstrap: [True, False]
        
           **LightGBM Parameters:**
           - n_estimators: [100, 200, 300, 500]
           - max_depth: [3, 4, 5, 6, 7]
           - learning_rate: [0.01, 0.05, 0.1, 0.2]
           - num_leaves: [31, 50, 100, 150]
           - feature_fraction: [0.8, 0.9, 1.0]
           - bagging_fraction: [0.8, 0.9, 1.0]
        
        3. **OPTIMIZATION OBJECTIVES:**
           - Primary: Maximize AUROC
           - Secondary: Minimize calibration error
           - Constraint: Maintain sensitivity ≥ 80%
           - Constraint: Prevent overfitting (train-val gap < 0.05)
        
        4. **VALIDATION PROTOCOL:**
           - 5-fold stratified cross-validation
           - Temporal validation for stability
           - Clinical subgroup validation
           - Robustness testing across data splits
        
        **CLINICAL CONSTRAINTS:**
        - Model interpretability preservation
        - Training time limitations (< 60 minutes per model)
        - Memory usage constraints for deployment
        - Feature importance stability across parameter changes
        
        **AUTOMATION FEATURES:**
        - Parallel hyperparameter search
        - Automatic early stopping
        - Performance monitoring and logging
        - Best configuration persistence
        """
    
    def _get_expected_output(self) -> str:
        """Get expected output specification."""
        return """
        OPTIMIZED HYPERPARAMETER CONFIGURATIONS:
        
        **BEST PARAMETERS BY MODEL:**
        ```
        {
            "xgboost_best_params": {
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.1,
                "reg_lambda": 0.5,
                "cv_score": 0.865,
                "std_score": 0.012
            },
            "random_forest_best_params": {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
                "bootstrap": true,
                "cv_score": 0.851,
                "std_score": 0.015
            },
            "lightgbm_best_params": {
                "n_estimators": 250,
                "max_depth": 4,
                "learning_rate": 0.1,
                "num_leaves": 50,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "cv_score": 0.859,
                "std_score": 0.011
            }
        }
        ```
        
        **OPTIMIZATION HISTORY:**
        - Total evaluations: 150 parameter combinations
        - Best AUROC achieved: 0.865 (XGBoost)
        - Optimization time: 4.2 hours
        - Convergence achieved: Yes (50 iterations without improvement)
        
        **PARAMETER SENSITIVITY ANALYSIS:**
        - Most important parameters: learning_rate, max_depth, n_estimators
        - Interaction effects identified: max_depth × learning_rate
        - Stable parameters: subsample, feature fractions
        - Overfitting risks: High max_depth + high learning_rate
        
        **CLINICAL VALIDATION:**
        - All optimized models maintain sensitivity ≥ 80%
        - Calibration error reduced by 15% through optimization
        - Feature importance rankings remain stable
        - Training time within clinical deployment constraints
        
        **FINAL RECOMMENDATIONS:**
        1. Use XGBoost with optimized parameters for best performance
        2. Ensemble top 3 configurations for improved robustness  
        3. Monitor for overfitting in production deployment
        4. Re-tune parameters if new data patterns emerge
        
        **FILES GENERATED:**
        - `hyperparameter_optimization_results.json`: Complete results
        - `parameter_sensitivity_analysis.csv`: Sensitivity study
        - `optimization_convergence_plot.png`: Convergence visualization
        - `best_models_comparison.md`: Performance comparison
        """
