"""
Feature Selector Tool for identifying important predictive features.
Uses multiple feature selection techniques optimized for healthcare data.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Feature Selection Libraries
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    RFE, RFECV, f_classif, chi2, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb

logger = logging.getLogger(__name__)

class FeatureSelectorInput(BaseModel):
    """Input schema for feature selection."""
    data_path: str = Field(description="Path to dataset for feature selection")
    target_column: str = Field(description="Name of target column")
    selection_method: str = Field(default="comprehensive", description="Feature selection method")
    max_features: Optional[int] = Field(default=None, description="Maximum number of features to select")
    importance_threshold: float = Field(default=0.01, description="Importance threshold for selection")

class FeatureSelectorTool(BaseTool):
    """Tool for selecting important features for chronic care risk prediction."""
    
    name: str = "Feature Selector"
    description: str = "Selects optimal features using multiple statistical and ML-based methods"
    args_schema: type[BaseModel] = FeatureSelectorInput
    
    def __init__(self):
        super().__init__()
        logger.info("FeatureSelectorTool initialized")
    
    def _run(self, data_path: str, target_column: str, selection_method: str = "comprehensive",
             max_features: Optional[int] = None, importance_threshold: float = 0.01) -> str:
        """Select optimal features for model training."""
        try:
            logger.info(f"Starting feature selection with method: {selection_method}")
            
            # Load and preprocess data
            X, y, feature_names = self._load_and_preprocess_data(data_path, target_column)
            
            # Apply feature selection method
            if selection_method == "comprehensive":
                results = self._comprehensive_feature_selection(
                    X, y, feature_names, max_features, importance_threshold
                )
            elif selection_method == "statistical":
                results = self._statistical_feature_selection(
                    X, y, feature_names, max_features
                )
            elif selection_method == "model_based":
                results = self._model_based_feature_selection(
                    X, y, feature_names, max_features, importance_threshold
                )
            elif selection_method == "recursive":
                results = self._recursive_feature_elimination(
                    X, y, feature_names, max_features
                )
            elif selection_method == "stability":
                results = self._stability_based_selection(
                    X, y, feature_names, max_features, importance_threshold
                )
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            
            # Validate selected features
            validation_results = self._validate_feature_selection(
                X, y, results["selected_features"], feature_names
            )
            
            # Generate feature selection report
            report = self._generate_selection_report(
                results, validation_results, selection_method
            )
            
            final_result = {
                "selection_method": selection_method,
                "original_features_count": len(feature_names),
                "selected_features_count": len(results["selected_features"]),
                "selected_features": results["selected_features"],
                "feature_scores": results.get("feature_scores", {}),
                "selection_rationale": results.get("rationale", {}),
                "validation_results": validation_results,
                "report": report,
                "data_info": {
                    "samples": X.shape[0],
                    "original_features": X.shape[1],
                    "target_distribution": {
                        "positive_class": int(np.sum(y)),
                        "negative_class": int(len(y) - np.sum(y)),
                        "positive_ratio": float(np.mean(y))
                    }
                }
            }
            
            logger.info(f"Feature selection completed: {len(results['selected_features'])} features selected")
            return json.dumps(final_result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _load_and_preprocess_data(self, data_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess data for feature selection."""
        # Load data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        y = df[target_column].values
        X_df = df.drop(columns=[target_column])
        
        # Handle missing values
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        X_df = X_df.fillna(X_df.mode().iloc[0])
        
        # Encode categorical variables
        for col in X_df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        
        feature_names = X_df.columns.tolist()
        X = X_df.values
        
        # Encode target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        return X, y, feature_names
    
    def _comprehensive_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                       feature_names: List[str], max_features: Optional[int], 
                                       importance_threshold: float) -> Dict[str, Any]:
        """Comprehensive feature selection using multiple methods."""
        
        # 1. Statistical methods
        statistical_features = self._get_statistical_features(X, y, feature_names)
        
        # 2. Model-based methods
        model_features = self._get_model_based_features(X, y, feature_names, importance_threshold)
        
        # 3. Recursive feature elimination
        rfe_features = self._get_rfe_features(X, y, feature_names)
        
        # 4. Stability selection
        stability_features = self._get_stability_features(X, y, feature_names, importance_threshold)
        
        # Combine results using voting
        feature_votes = {}
        for feature in feature_names:
            votes = 0
            if feature in statistical_features["selected"]: votes += 1
            if feature in model_features["selected"]: votes += 1
            if feature in rfe_features["selected"]: votes += 1
            if feature in stability_features["selected"]: votes += 1
            
            feature_votes[feature] = votes
        
        # Select features with majority votes (at least 2/4 methods)
        selected_features = [f for f, votes in feature_votes.items() if votes >= 2]
        
        # Apply max_features constraint if specified
        if max_features and len(selected_features) > max_features:
            # Sort by total score and select top features
            feature_scores = {}
            for feature in selected_features:
                total_score = 0
                total_score += statistical_features["scores"].get(feature, 0)
                total_score += model_features["scores"].get(feature, 0)
                total_score += rfe_features["scores"].get(feature, 0)
                total_score += stability_features["scores"].get(feature, 0)
                feature_scores[feature] = total_score
            
            selected_features = sorted(feature_scores.keys(), 
                                     key=lambda x: feature_scores[x], reverse=True)[:max_features]
        
        return {
            "selected_features": selected_features,
            "feature_scores": feature_votes,
            "method_results": {
                "statistical": statistical_features,
                "model_based": model_features,
                "rfe": rfe_features,
                "stability": stability_features
            },
            "rationale": {
                "approach": "Ensemble voting with majority consensus",
                "methods_used": ["statistical", "model_based", "rfe", "stability"],
                "voting_threshold": "2 out of 4 methods"
            }
        }
    
    def _get_statistical_features(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str]) -> Dict[str, Any]:
        """Get features using statistical methods."""
        
        # F-score for continuous features
        f_scores = f_classif(X, y)[0]
        f_scores = np.nan_to_num(f_scores, 0)
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Combine scores
        combined_scores = (f_scores + mi_scores) / 2
        
        # Select top 50% of features
        n_select = max(1, len(feature_names) // 2)
        top_indices = np.argsort(combined_scores)[::-1][:n_select]
        selected_features = [feature_names[i] for i in top_indices]
        
        feature_scores = dict(zip(feature_names, combined_scores))
        
        return {
            "selected": selected_features,
            "scores": feature_scores,
            "method": "f_score + mutual_information"
        }
    
    def _get_model_based_features(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str], threshold: float) -> Dict[str, Any]:
        """Get features using model-based importance."""
        
        # Train multiple models and average importance
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)
        }
        
        all_importances = []
        
        for name, model in models.items():
            model.fit(X, y)
            if hasattr(model, 'feature_importances_'):
                all_importances.append(model.feature_importances_)
        
        # Average importance across models
        avg_importance = np.mean(all_importances, axis=0)
        
        # Select features above threshold
        selected_indices = np.where(avg_importance >= threshold)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # If no features selected, select top 20%
        if not selected_features:
            n_select = max(1, len(feature_names) // 5)
            top_indices = np.argsort(avg_importance)[::-1][:n_select]
            selected_features = [feature_names[i] for i in top_indices]
        
        feature_scores = dict(zip(feature_names, avg_importance))
        
        return {
            "selected": selected_features,
            "scores": feature_scores,
            "method": "averaged_model_importance"
        }
    
    def _get_rfe_features(self, X: np.ndarray, y: np.ndarray, 
                         feature_names: List[str]) -> Dict[str, Any]:
        """Get features using recursive feature elimination."""
        
        # Use RandomForest for RFE
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Determine optimal number of features using cross-validation
        n_features_range = range(max(1, len(feature_names) // 4), 
                               min(len(feature_names), len(feature_names) // 2))
        
        best_score = 0
        best_n_features = len(feature_names) // 3
        
        for n_features in n_features_range:
            rfe = RFE(estimator, n_features_to_select=n_features)
            scores = cross_val_score(rfe, X, y, cv=3, scoring='roc_auc')
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_n_features = n_features
        
        # Apply RFE with optimal number of features
        rfe = RFE(estimator, n_features_to_select=best_n_features)
        rfe.fit(X, y)
        
        selected_indices = np.where(rfe.support_)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Create ranking scores (lower rank = higher importance)
        feature_scores = {}
        for i, feature in enumerate(feature_names):
            feature_scores[feature] = 1.0 / rfe.ranking_[i]  # Inverse ranking
        
        return {
            "selected": selected_features,
            "scores": feature_scores,
            "method": "recursive_feature_elimination",
            "optimal_n_features": best_n_features,
            "cv_score": best_score
        }
    
    def _get_stability_features(self, X: np.ndarray, y: np.ndarray, 
                              feature_names: List[str], threshold: float) -> Dict[str, Any]:
        """Get features using stability selection."""
        
        n_bootstrap = 20
        n_features = len(feature_names)
        selection_counts = np.zeros(n_features)
        
        # Bootstrap sampling and feature selection
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model and get feature importance
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_boot, y_boot)
            
            # Select features above threshold
            importances = model.feature_importances_
            selected = importances >= threshold
            selection_counts += selected
        
        # Calculate selection probability
        selection_prob = selection_counts / n_bootstrap
        
        # Select stable features (selected in >50% of bootstraps)
        stable_threshold = 0.5
        selected_indices = np.where(selection_prob >= stable_threshold)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # If no features selected, select top 30%
        if not selected_features:
            n_select = max(1, len(feature_names) // 3)
            top_indices = np.argsort(selection_prob)[::-1][:n_select]
            selected_features = [feature_names[i] for i in top_indices]
        
        feature_scores = dict(zip(feature_names, selection_prob))
        
        return {
            "selected": selected_features,
            "scores": feature_scores,
            "method": "stability_selection",
            "bootstrap_iterations": n_bootstrap,
            "stability_threshold": stable_threshold
        }
    
    def _statistical_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str], max_features: Optional[int]) -> Dict[str, Any]:
        """Statistical-only feature selection."""
        return self._get_statistical_features(X, y, feature_names)
    
    def _model_based_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str], max_features: Optional[int], 
                                     threshold: float) -> Dict[str, Any]:
        """Model-based feature selection."""
        return self._get_model_based_features(X, y, feature_names, threshold)
    
    def _recursive_feature_elimination(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str], max_features: Optional[int]) -> Dict[str, Any]:
        """Recursive feature elimination."""
        return self._get_rfe_features(X, y, feature_names)
    
    def _stability_based_selection(self, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str], max_features: Optional[int], 
                                 threshold: float) -> Dict[str, Any]:
        """Stability-based feature selection."""
        return self._get_stability_features(X, y, feature_names, threshold)
    
    def _validate_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                                  selected_features: List[str], all_features: List[str]) -> Dict[str, Any]:
        """Validate the quality of selected features."""
        
        # Get feature indices
        selected_indices = [all_features.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]
        
        # Compare model performance with all features vs selected features
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Performance with all features
        scores_all = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        
        # Performance with selected features
        scores_selected = cross_val_score(model, X_selected, y, cv=5, scoring='roc_auc')
        
        # Calculate feature reduction ratio
        reduction_ratio = len(selected_features) / len(all_features)
        
        # Performance retention
        performance_retention = np.mean(scores_selected) / np.mean(scores_all)
        
        return {
            "all_features_performance": {
                "mean_roc_auc": float(np.mean(scores_all)),
                "std_roc_auc": float(np.std(scores_all))
            },
            "selected_features_performance": {
                "mean_roc_auc": float(np.mean(scores_selected)),
                "std_roc_auc": float(np.std(scores_selected))
            },
            "feature_reduction_ratio": float(reduction_ratio),
            "performance_retention": float(performance_retention),
            "improvement": float(np.mean(scores_selected) - np.mean(scores_all))
        }
    
    def _generate_selection_report(self, results: Dict[str, Any], 
                                 validation: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Generate comprehensive feature selection report."""
        
        report = {
            "summary": {
                "method_used": method,
                "features_selected": len(results["selected_features"]),
                "selection_quality": "excellent" if validation["performance_retention"] >= 0.95 else
                                  "good" if validation["performance_retention"] >= 0.90 else "acceptable"
            },
            "performance_impact": {
                "feature_reduction": f"{(1 - validation['feature_reduction_ratio']) * 100:.1f}%",
                "performance_change": f"{validation['improvement'] * 100:+.2f}% ROC-AUC",
                "performance_retention": f"{validation['performance_retention'] * 100:.1f}%"
            },
            "top_selected_features": results["selected_features"][:10],
            "recommendations": self._generate_recommendations(validation, results)
        }
        
        return report
    
    def _generate_recommendations(self, validation: Dict[str, Any], 
                                results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on selection results."""
        recommendations = []
        
        if validation["performance_retention"] >= 0.95:
            recommendations.append("Selected features maintain excellent model performance")
        elif validation["performance_retention"] >= 0.90:
            recommendations.append("Selected features maintain good model performance")
        else:
            recommendations.append("Consider including additional features to improve performance")
        
        if validation["feature_reduction_ratio"] < 0.3:
            recommendations.append("Significant feature reduction achieved - good for model interpretability")
        elif validation["feature_reduction_ratio"] < 0.6:
            recommendations.append("Moderate feature reduction - balanced approach")
        else:
            recommendations.append("Limited feature reduction - consider more aggressive selection")
        
        if validation["improvement"] > 0:
            recommendations.append("Feature selection improved model performance")
        else:
            recommendations.append("Monitor model performance with selected features")
        
        return recommendations
