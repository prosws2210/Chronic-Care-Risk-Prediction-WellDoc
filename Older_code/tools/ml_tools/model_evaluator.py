"""
Model Evaluator Tool for comprehensive ML model assessment.
Provides clinical-grade evaluation metrics and validation protocols.
"""

import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path

# Evaluation Libraries
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    brier_score_loss, log_loss, matthews_corrcoef
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score, StratifiedKFold
import scipy.stats as stats

logger = logging.getLogger(__name__)

class ModelEvaluatorInput(BaseModel):
    """Input schema for model evaluation."""
    y_true_path: str = Field(description="Path to true labels file")
    y_pred_path: str = Field(description="Path to predicted probabilities file")
    model_name: Optional[str] = Field(default="model", description="Name of the model being evaluated")
    threshold: float = Field(default=0.5, description="Classification threshold")
    clinical_focus: bool = Field(default=True, description="Include clinical evaluation metrics")

class ModelEvaluatorTool(BaseTool):
    """Tool for comprehensive evaluation of ML models with clinical focus."""
    
    name: str = "Model Evaluator"
    description: str = "Comprehensive ML model evaluation with healthcare-specific metrics"
    args_schema: type[BaseModel] = ModelEvaluatorInput
    
    def __init__(self):
        super().__init__()
        logger.info("ModelEvaluatorTool initialized")
    
    def _run(self, y_true_path: str, y_pred_path: str, model_name: str = "model",
             threshold: float = 0.5, clinical_focus: bool = True) -> str:
        """Evaluate model performance comprehensively."""
        try:
            logger.info(f"Evaluating model: {model_name}")
            
            # Load predictions and true labels
            y_true, y_pred_proba, y_pred = self._load_predictions(
                y_true_path, y_pred_path, threshold
            )
            
            # Basic classification metrics
            basic_metrics = self._calculate_basic_metrics(y_true, y_pred_proba, y_pred)
            
            # Advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(y_true, y_pred_proba, y_pred)
            
            # Clinical evaluation metrics
            clinical_metrics = {}
            if clinical_focus:
                clinical_metrics = self._calculate_clinical_metrics(y_true, y_pred_proba, y_pred)
            
            # Calibration analysis
            calibration_results = self._analyze_calibration(y_true, y_pred_proba)
            
            # Threshold analysis
            threshold_analysis = self._analyze_thresholds(y_true, y_pred_proba)
            
            # Statistical tests
            statistical_tests = self._perform_statistical_tests(y_true, y_pred_proba)
            
            # Subgroup analysis (if demographic data available)
            subgroup_analysis = self._perform_subgroup_analysis(y_true, y_pred_proba)
            
            # Generate confusion matrix analysis
            confusion_analysis = self._analyze_confusion_matrix(y_true, y_pred)
            
            # Performance interpretation
            interpretation = self._interpret_performance(basic_metrics, advanced_metrics, clinical_metrics)
            
            # Create comprehensive report
            evaluation_report = {
                "model_name": model_name,
                "evaluation_timestamp": pd.Timestamp.now().isoformat(),
                "data_summary": {
                    "total_samples": len(y_true),
                    "positive_cases": int(np.sum(y_true)),
                    "negative_cases": int(len(y_true) - np.sum(y_true)),
                    "class_balance": float(np.mean(y_true)),
                    "classification_threshold": threshold
                },
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "clinical_metrics": clinical_metrics,
                "calibration_analysis": calibration_results,
                "threshold_analysis": threshold_analysis,
                "statistical_tests": statistical_tests,
                "confusion_matrix_analysis": confusion_analysis,
                "subgroup_analysis": subgroup_analysis,
                "performance_interpretation": interpretation,
                "recommendations": self._generate_recommendations(
                    basic_metrics, advanced_metrics, clinical_metrics, calibration_results
                )
            }
            
            logger.info(f"Model evaluation completed. AUROC: {basic_metrics['roc_auc']:.3f}")
            return json.dumps(evaluation_report, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _load_predictions(self, y_true_path: str, y_pred_path: str, 
                         threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load true labels and predictions."""
        
        # Load true labels
        if y_true_path.endswith('.csv'):
            y_true = pd.read_csv(y_true_path).values.flatten()
        elif y_true_path.endswith('.json'):
            y_true = np.array(pd.read_json(y_true_path).iloc[:, 0])
        else:
            y_true = np.loadtxt(y_true_path)
        
        # Load predicted probabilities
        if y_pred_path.endswith('.csv'):
            y_pred_proba = pd.read_csv(y_pred_path).values.flatten()
        elif y_pred_path.endswith('.json'):
            y_pred_proba = np.array(pd.read_json(y_pred_path).iloc[:, 0])
        else:
            y_pred_proba = np.loadtxt(y_pred_path)
        
        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Validate data
        if len(y_true) != len(y_pred_proba):
            raise ValueError("Mismatch in number of true labels and predictions")
        
        if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
            logger.warning("Some predicted values are outside [0,1] range")
            y_pred_proba = np.clip(y_pred_proba, 0, 1)
        
        return y_true.astype(int), y_pred_proba, y_pred
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
        }
        
        return {k: round(v, 4) for k, v in metrics.items()}
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate advanced evaluation metrics."""
        
        # Probabilistic metrics
        brier_score = brier_score_loss(y_true, y_pred_proba)
        try:
            logloss = log_loss(y_true, y_pred_proba)
        except:
            logloss = np.nan
        
        # Confidence intervals for AUROC
        auroc_ci = self._calculate_auroc_confidence_interval(y_true, y_pred_proba)
        
        # Positive and Negative Predictive Values
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Likelihood ratios
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        lr_positive = sensitivity / (1 - specificity) if specificity < 1 else np.inf
        lr_negative = (1 - sensitivity) / specificity if specificity > 0 else np.inf
        
        # Diagnostic odds ratio
        dor = lr_positive / lr_negative if lr_negative > 0 else np.inf
        
        metrics = {
            'brier_score': brier_score,
            'log_loss': logloss,
            'auroc_ci_lower': auroc_ci[0],
            'auroc_ci_upper': auroc_ci[1],
            'positive_predictive_value': ppv,
            'negative_predictive_value': npv,
            'likelihood_ratio_positive': lr_positive,
            'likelihood_ratio_negative': lr_negative,
            'diagnostic_odds_ratio': dor,
            'youden_index': sensitivity + specificity - 1
        }
        
        return {k: round(v, 4) if not np.isinf(v) else v for k, v in metrics.items()}
    
    def _calculate_clinical_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate clinical evaluation metrics."""
        
        # Number needed to screen/evaluate
        prevalence = np.mean(y_true)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        
        # Calculate NNE (Number Needed to Evaluate)
        nne = 1 / (sensitivity * prevalence) if sensitivity > 0 and prevalence > 0 else np.inf
        
        # Clinical decision metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Missed cases (false negatives) - critical in healthcare
        missed_cases_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        # False alarm rate (false positives) - resource impact
        false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        # Net benefit analysis (simplified)
        net_benefit = self._calculate_net_benefit(y_true, y_pred_proba, threshold=0.5)
        
        # Risk stratification analysis
        risk_strata = self._analyze_risk_stratification(y_true, y_pred_proba)
        
        clinical_metrics = {
            'number_needed_to_evaluate': nne,
            'missed_cases_rate': missed_cases_rate,
            'false_alarm_rate': false_alarm_rate,
            'net_benefit': net_benefit,
            'risk_stratification': risk_strata,
            'clinical_impact': {
                'high_risk_correctly_identified': tp,
                'high_risk_missed': fn,
                'low_risk_incorrectly_flagged': fp,
                'low_risk_correctly_identified': tn
            }
        }
        
        return {k: round(v, 4) if isinstance(v, (int, float)) and not np.isinf(v) else v 
                for k, v in clinical_metrics.items()}
    
    def _analyze_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration."""
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Hosmer-Lemeshow test approximation
        hl_stat, hl_pvalue = self._hosmer_lemeshow_test(y_true, y_pred_proba)
        
        # Expected Calibration Error
        ece = self._calculate_expected_calibration_error(y_true, y_pred_proba)
        
        # Calibration slope and intercept
        cal_slope, cal_intercept = self._calculate_calibration_slope_intercept(y_true, y_pred_proba)
        
        calibration_results = {
            'calibration_curve': {
                'bin_boundaries': mean_predicted_value.tolist(),
                'observed_frequencies': fraction_of_positives.tolist()
            },
            'hosmer_lemeshow_statistic': hl_stat,
            'hosmer_lemeshow_pvalue': hl_pvalue,
            'expected_calibration_error': ece,
            'calibration_slope': cal_slope,
            'calibration_intercept': cal_intercept,
            'calibration_quality': self._assess_calibration_quality(ece, hl_pvalue)
        }
        
        return calibration_results
    
    def _analyze_thresholds(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze performance across different classification thresholds."""
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Calculate precision-recall curve  
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find optimal thresholds
        youden_threshold = roc_thresholds[np.argmax(tpr - fpr)]
        f1_optimal_idx = np.argmax([2 * p * r / (p + r) if p + r > 0 else 0 
                                   for p, r in zip(precision[:-1], recall[:-1])])
        f1_threshold = pr_thresholds[f1_optimal_idx] if f1_optimal_idx < len(pr_thresholds) else 0.5
        
        # Sensitivity-Specificity at different thresholds
        threshold_analysis = []
        test_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        for thresh in test_thresholds:
            y_pred_thresh = (y_pred_proba >= thresh).astype(int)
            sens = recall_score(y_true, y_pred_thresh, zero_division=0)
            spec = self._calculate_specificity(y_true, y_pred_thresh)
            ppv = precision_score(y_true, y_pred_thresh, zero_division=0)
            
            threshold_analysis.append({
                'threshold': thresh,
                'sensitivity': round(sens, 4),
                'specificity': round(spec, 4),
                'ppv': round(ppv, 4),
                'youden_index': round(sens + spec - 1, 4)
            })
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'precision_recall_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            },
            'optimal_thresholds': {
                'youden_optimal': round(youden_threshold, 4),
                'f1_optimal': round(f1_threshold, 4)
            },
            'threshold_analysis': threshold_analysis
        }
    
    def _perform_statistical_tests(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        
        # DeLong test for comparing AUROC to 0.5 (no discrimination)
        auroc = roc_auc_score(y_true, y_pred_proba)
        
        # Bootstrap confidence interval for AUROC
        n_bootstrap = 1000
        bootstrap_aucs = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[indices])) < 2:
                continue
            bootstrap_aucs.append(roc_auc_score(y_true[indices], y_pred_proba[indices]))
        
        auroc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
        auroc_ci_upper = np.percentile(bootstrap_aucs, 97.5)
        
        # Test if AUROC is significantly different from 0.5
        auroc_pvalue = 2 * min(
            stats.norm.cdf((0.5 - auroc) / np.std(bootstrap_aucs)),
            1 - stats.norm.cdf((0.5 - auroc) / np.std(bootstrap_aucs))
        )
        
        return {
            'auroc_bootstrap_ci': [round(auroc_ci_lower, 4), round(auroc_ci_upper, 4)],
            'auroc_vs_random_pvalue': round(auroc_pvalue, 4),
            'auroc_significantly_better_than_random': auroc_pvalue < 0.05
        }
    
    def _perform_subgroup_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Perform subgroup analysis if demographic data is available."""
        
        # This is a placeholder for subgroup analysis
        # In practice, you would need demographic data to perform this analysis
        
        return {
            'note': 'Subgroup analysis requires demographic data not provided in this evaluation',
            'recommendations': [
                'Consider evaluating model performance across age groups',
                'Assess performance across gender categories', 
                'Evaluate performance across racial/ethnic groups',
                'Check for performance differences by socioeconomic status'
            ]
        }
    
    def _analyze_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix in detail."""
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total = tn + fp + fn + tp
        
        analysis = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'total_samples': int(total),
            'error_analysis': {
                'false_positive_rate': round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0,
                'false_negative_rate': round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0,
                'false_discovery_rate': round(fp / (fp + tp), 4) if (fp + tp) > 0 else 0,
                'false_omission_rate': round(fn / (fn + tn), 4) if (fn + tn) > 0 else 0
            }
        }
        
        return analysis
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_auroc_confidence_interval(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                           confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for AUROC using DeLong method approximation."""
        
        auroc = roc_auc_score(y_true, y_pred_proba)
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        # Approximation for AUROC variance
        q1 = auroc / (2 - auroc)
        q2 = 2 * auroc**2 / (1 + auroc)
        
        se_auroc = np.sqrt((auroc * (1 - auroc) + (n_pos - 1) * (q1 - auroc**2) + 
                          (n_neg - 1) * (q2 - auroc**2)) / (n_pos * n_neg))
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        ci_lower = max(0, auroc - z_score * se_auroc)
        ci_upper = min(1, auroc + z_score * se_auroc)
        
        return ci_lower, ci_upper
    
    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             n_bins: int = 10) -> Tuple[float, float]:
        """Perform Hosmer-Lemeshow goodness of fit test."""
        
        # Create bins based on predicted probabilities
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        hl_stat = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find observations in this bin
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)
            if bin_upper == 1:  # Include upper boundary for last bin
                in_bin = in_bin | (y_pred_proba == 1)
            
            if np.sum(in_bin) > 0:
                observed_pos = np.sum(y_true[in_bin])
                observed_neg = np.sum(in_bin) - observed_pos
                expected_pos = np.sum(y_pred_proba[in_bin])
                expected_neg = np.sum(in_bin) - expected_pos
                
                if expected_pos > 0 and expected_neg > 0:
                    hl_stat += ((observed_pos - expected_pos)**2 / expected_pos + 
                              (observed_neg - expected_neg)**2 / expected_neg)
        
        # Degrees of freedom = number of bins - 2
        df = n_bins - 2
        p_value = 1 - stats.chi2.cdf(hl_stat, df)
        
        return hl_stat, p_value
    
    def _calculate_expected_calibration_error(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                            n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        total_samples = len(y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_calibration_slope_intercept(self, y_true: np.ndarray, 
                                             y_pred_proba: np.ndarray) -> Tuple[float, float]:
        """Calculate calibration slope and intercept using logistic regression."""
        
        from sklearn.linear_model import LogisticRegression
        
        # Convert probabilities to logits
        epsilon = 1e-15
        y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        logits = np.log(y_pred_proba_clipped / (1 - y_pred_proba_clipped))
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), y_true)
        
        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]
        
        return slope, intercept
    
    def _assess_calibration_quality(self, ece: float, hl_pvalue: float) -> str:
        """Assess calibration quality based on metrics."""
        
        if ece < 0.05 and hl_pvalue > 0.05:
            return "Excellent"
        elif ece < 0.10 and hl_pvalue > 0.01:
            return "Good"
        elif ece < 0.15:
            return "Acceptable"
        else:
            return "Poor"
    
    def _calculate_net_benefit(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             threshold: float) -> float:
        """Calculate net benefit for decision curve analysis."""
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # True positives and false positives
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        # Total samples
        n = len(y_true)
        
        # Net benefit calculation
        net_benefit = (tp - fp * (threshold / (1 - threshold))) / n
        
        return net_benefit
    
    def _analyze_risk_stratification(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze risk stratification performance."""
        
        # Define risk groups based on predicted probabilities
        low_risk = y_pred_proba < 0.3
        medium_risk = (y_pred_proba >= 0.3) & (y_pred_proba < 0.7)
        high_risk = y_pred_proba >= 0.7
        
        risk_groups = {
            'low_risk': {
                'count': int(np.sum(low_risk)),
                'positive_rate': float(np.mean(y_true[low_risk])) if np.sum(low_risk) > 0 else 0,
                'avg_predicted_prob': float(np.mean(y_pred_proba[low_risk])) if np.sum(low_risk) > 0 else 0
            },
            'medium_risk': {
                'count': int(np.sum(medium_risk)),
                'positive_rate': float(np.mean(y_true[medium_risk])) if np.sum(medium_risk) > 0 else 0,
                'avg_predicted_prob': float(np.mean(y_pred_proba[medium_risk])) if np.sum(medium_risk) > 0 else 0
            },
            'high_risk': {
                'count': int(np.sum(high_risk)),
                'positive_rate': float(np.mean(y_true[high_risk])) if np.sum(high_risk) > 0 else 0,
                'avg_predicted_prob': float(np.mean(y_pred_proba[high_risk])) if np.sum(high_risk) > 0 else 0
            }
        }
        
        return risk_groups
    
    def _interpret_performance(self, basic_metrics: Dict, advanced_metrics: Dict, 
                             clinical_metrics: Dict) -> Dict[str, str]:
        """Interpret model performance in clinical context."""
        
        auroc = basic_metrics['roc_auc']
        
        # AUROC interpretation
        if auroc >= 0.90:
            auroc_interpretation = "Outstanding discrimination"
        elif auroc >= 0.80:
            auroc_interpretation = "Excellent discrimination"
        elif auroc >= 0.70:
            auroc_interpretation = "Acceptable discrimination"
        elif auroc >= 0.60:
            auroc_interpretation = "Poor discrimination"
        else:
            auroc_interpretation = "Fail - no better than random"
        
        # Clinical utility interpretation
        sensitivity = basic_metrics['recall']
        specificity = basic_metrics['specificity']
        
        if sensitivity >= 0.80 and specificity >= 0.80:
            clinical_utility = "High clinical utility - good balance of sensitivity and specificity"
        elif sensitivity >= 0.80:
            clinical_utility = "High sensitivity - good for screening, but may have false positives"
        elif specificity >= 0.80:
            clinical_utility = "High specificity - good for confirmation, but may miss cases"
        else:
            clinical_utility = "Limited clinical utility - consider threshold adjustment"
        
        return {
            'auroc_interpretation': auroc_interpretation,
            'clinical_utility': clinical_utility,
            'overall_assessment': self._get_overall_assessment(basic_metrics, advanced_metrics)
        }
    
    def _get_overall_assessment(self, basic_metrics: Dict, advanced_metrics: Dict) -> str:
        """Get overall model assessment."""
        
        auroc = basic_metrics['roc_auc']
        f1 = basic_metrics['f1_score']
        
        if auroc >= 0.85 and f1 >= 0.70:
            return "Excellent model ready for clinical deployment"
        elif auroc >= 0.75 and f1 >= 0.60:
            return "Good model suitable for clinical use with monitoring"
        elif auroc >= 0.70:
            return "Acceptable model - consider improvements before deployment"
        else:
            return "Model needs significant improvement before clinical use"
    
    def _generate_recommendations(self, basic_metrics: Dict, advanced_metrics: Dict, 
                                clinical_metrics: Dict, calibration_results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results."""
        
        recommendations = []
        
        # Performance-based recommendations
        if basic_metrics['roc_auc'] < 0.75:
            recommendations.append("Consider feature engineering or alternative algorithms to improve discrimination")
        
        if basic_metrics['recall'] < 0.80:
            recommendations.append("Consider lowering classification threshold to improve sensitivity (reduce missed cases)")
        
        if basic_metrics['precision'] < 0.70:
            recommendations.append("Consider raising classification threshold to reduce false positives")
        
        # Calibration-based recommendations
        calibration_quality = calibration_results.get('calibration_quality', 'Unknown')
        if calibration_quality in ['Poor', 'Acceptable']:
            recommendations.append("Consider calibrating model probabilities using Platt scaling or isotonic regression")
        
        # Clinical recommendations
        if clinical_metrics and clinical_metrics.get('missed_cases_rate', 0) > 0.20:
            recommendations.append("High missed case rate - consider clinical review of false negatives")
        
        if clinical_metrics and clinical_metrics.get('false_alarm_rate', 0) > 0.20:
            recommendations.append("High false alarm rate - consider resource impact and threshold optimization")
        
        # General recommendations
        recommendations.append("Validate model performance on external dataset before deployment")
        recommendations.append("Establish continuous monitoring protocols for model drift")
        recommendations.append("Consider subgroup analysis to ensure equitable performance")
        
        return recommendations
