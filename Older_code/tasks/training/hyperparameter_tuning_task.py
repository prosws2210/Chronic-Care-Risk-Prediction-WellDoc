"""
Hyperparameter Tuning Task for optimizing model performance.
Uses Bayesian optimization to find optimal parameter configurations.
"""

import logging
from typing import Dict, Any
from crewai import Task

logger = logging.getLogger(__name__)

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
