"""
Risk Prediction Model Tool for building and training chronic care deterioration models.
Implements ensemble ML models optimized for healthcare risk prediction.
"""

import json
import logging
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# ML Libraries
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class RiskPredictionModelInput(BaseModel):
    """Input schema for risk prediction model training."""
    training_data_path: str = Field(description="Path to training dataset")
    model_type: str = Field(default="ensemble", description="Type of model to train")
    target_column: str = Field(default="deterioration_risk", description="Target column name")
    hyperparameter_tuning: bool = Field(default=True, description="Enable hyperparameter tuning")
    cross_validation_folds: int = Field(default=5, description="Number of CV folds")
    test_size: float = Field(default=0.2, description="Test set proportion")

class RiskPredictionModelTool(BaseTool):
    """Tool for training risk prediction models for chronic care patients."""
    
    name: str = "Risk Prediction Model"
    description: str = "Builds and trains ML models for 90-day deterioration risk prediction"
    args_schema: type[BaseModel] = RiskPredictionModelInput
    
    def __init__(self):
        super().__init__()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_metrics = {}
        logger.info("RiskPredictionModelTool initialized")
    
    def _run(self, training_data_path: str, model_type: str = "ensemble",
             target_column: str = "deterioration_risk", hyperparameter_tuning: bool = True,
             cross_validation_folds: int = 5, test_size: float = 0.2) -> str:
        """Train risk prediction models."""
        try:
            logger.info(f"Training {model_type} model with data: {training_data_path}")
            
            # Load and preprocess data
            X, y, feature_names = self._load_and_preprocess_data(
                training_data_path, target_column
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Train models based on type
            if model_type == "ensemble":
                model_results = self._train_ensemble_models(
                    X_train, X_test, y_train, y_test, hyperparameter_tuning, cross_validation_folds
                )
            elif model_type == "xgboost":
                model_results = self._train_xgboost_model(
                    X_train, X_test, y_train, y_test, hyperparameter_tuning
                )
            elif model_type == "random_forest":
                model_results = self._train_random_forest(
                    X_train, X_test, y_train, y_test, hyperparameter_tuning
                )
            elif model_type == "logistic_regression":
                model_results = self._train_logistic_regression(
                    X_train, X_test, y_train, y_test, hyperparameter_tuning
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Generate feature importance analysis
            feature_importance = self._analyze_feature_importance(
                model_results["best_model"], feature_names
            )
            
            # Calibrate probabilities
            calibrated_model = self._calibrate_model(
                model_results["best_model"], X_train, y_train
            )
            
            # Save models and artifacts
            model_artifacts = self._save_model_artifacts(
                calibrated_model, model_results, feature_names, feature_importance
            )
            
            result = {
                "model_type": model_type,
                "training_completed": True,
                "model_performance": model_results["performance"],
                "feature_importance": feature_importance,
                "model_artifacts": model_artifacts,
                "data_info": {
                    "total_samples": len(X),
                    "features_count": len(feature_names),
                    "positive_class_ratio": y.mean(),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test)
                },
                "training_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Model training completed successfully. AUROC: {model_results['performance']['roc_auc']:.3f}")
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def _load_and_preprocess_data(self, data_path: str, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess training data."""
        try:
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Separate features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            y = df[target_column].values
            X_df = df.drop(columns=[target_column])
            
            # Handle missing values
            X_df = self._handle_missing_values(X_df)
            
            # Encode categorical variables
            X_df = self._encode_categorical_features(X_df)
            
            # Feature engineering for healthcare data
            X_df = self._engineer_healthcare_features(X_df)
            
            feature_names = X_df.columns.tolist()
            X = X_df.values
            
            # Handle target variable
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            logger.info(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Numeric columns: fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Categorical columns: fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            # Use label encoding for binary categories, one-hot for others
            if df[col].nunique() == 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                # One-hot encoding with max categories limit
                if df[col].nunique() <= 10:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                else:
                    # For high cardinality, keep top categories
                    top_categories = df[col].value_counts().head(5).index
                    df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        
        return df
    
    def _engineer_healthcare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer healthcare-specific features."""
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], bins=[0, 40, 65, 80, 100], 
                                   labels=['young', 'middle', 'elderly', 'very_elderly'])
            df = pd.get_dummies(df, columns=['age_group'], prefix='age_group')
        
        # BMI categories
        if 'bmi' in df.columns:
            df['bmi_category'] = pd.cut(df['bmi'], 
                                      bins=[0, 18.5, 25, 30, 35, 100],
                                      labels=['underweight', 'normal', 'overweight', 'obese_1', 'obese_2'])
            df = pd.get_dummies(df, columns=['bmi_category'], prefix='bmi')
        
        # Blood pressure categories
        if 'systolic_bp' in df.columns:
            df['bp_category'] = pd.cut(df['systolic_bp'],
                                     bins=[0, 120, 140, 160, 300],
                                     labels=['normal', 'elevated', 'stage1', 'stage2'])
            df = pd.get_dummies(df, columns=['bp_category'], prefix='bp')
        
        # HbA1c control categories for diabetes
        if 'hba1c' in df.columns:
            df['diabetes_control'] = pd.cut(df['hba1c'],
                                          bins=[0, 7, 8, 9, 20],
                                          labels=['good', 'fair', 'poor', 'very_poor'])
            df = pd.get_dummies(df, columns=['diabetes_control'], prefix='diabetes')
        
        # Comorbidity burden
        comorbidity_cols = [col for col in df.columns if 'comorbidity' in col.lower() or 'condition' in col.lower()]
        if comorbidity_cols:
            df['comorbidity_count'] = df[comorbidity_cols].sum(axis=1)
            df['high_comorbidity_burden'] = (df['comorbidity_count'] >= 3).astype(int)
        
        return df
    
    def _train_ensemble_models(self, X_train: np.ndarray, X_test: np.ndarray,
                              y_train: np.ndarray, y_test: np.ndarray,
                              tune_hyperparameters: bool, cv_folds: int) -> Dict[str, Any]:
        """Train ensemble of multiple models."""
        models = {}
        performances = {}
        
        # Define base models
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'extra_trees': ExtraTreesClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Train individual models
        for name, model in base_models.items():
            logger.info(f"Training {name} model...")
            
            if tune_hyperparameters:
                tuned_model = self._tune_hyperparameters(model, X_train, y_train, cv_folds)
                models[name] = tuned_model
            else:
                model.fit(X_train, y_train)
                models[name] = model
            
            # Evaluate model
            y_pred_proba = models[name].predict_proba(X_test)[:, 1]
            performances[name] = self._calculate_metrics(y_test, y_pred_proba)
            
            logger.info(f"{name} AUROC: {performances[name]['roc_auc']:.3f}")
        
        # Create voting ensemble
        voting_models = [(name, model) for name, model in models.items()]
        ensemble = VotingClassifier(estimators=voting_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
        ensemble_performance = self._calculate_metrics(y_test, y_pred_proba_ensemble)
        
        # Find best performing model
        best_model_name = max(performances.keys(), key=lambda x: performances[x]['roc_auc'])
        best_individual_model = models[best_model_name]
        
        # Choose between best individual model and ensemble
        if ensemble_performance['roc_auc'] > performances[best_model_name]['roc_auc']:
            best_model = ensemble
            best_performance = ensemble_performance
            best_model_name = 'ensemble'
        else:
            best_model = best_individual_model
            best_performance = performances[best_model_name]
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'performance': best_performance,
            'all_performances': performances,
            'individual_models': models
        }
    
    def _train_xgboost_model(self, X_train: np.ndarray, X_test: np.ndarray,
                            y_train: np.ndarray, y_test: np.ndarray,
                            tune_hyperparameters: bool) -> Dict[str, Any]:
        """Train XGBoost model."""
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        performance = self._calculate_metrics(y_test, y_pred_proba)
        
        return {
            'best_model': model,
            'best_model_name': 'xgboost',
            'performance': performance
        }
    
    def _train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray, y_test: np.ndarray,
                           tune_hyperparameters: bool) -> Dict[str, Any]:
        """Train Random Forest model."""
        model = RandomForestClassifier(random_state=42)
        
        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        performance = self._calculate_metrics(y_test, y_pred_proba)
        
        return {
            'best_model': model,
            'best_model_name': 'random_forest',
            'performance': performance
        }
    
    def _train_logistic_regression(self, X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray,
                                 tune_hyperparameters: bool) -> Dict[str, Any]:
        """Train Logistic Regression model."""
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        
        if tune_hyperparameters:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            }
            
            # Filter combinations for compatibility
            param_grid = [
                {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear', 'saga']},
                {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear', 'saga']},
                {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.5]}
            ]
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='roc_auc',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            model = grid_search.best_estimator_
        else:
            model.fit(X_train_scaled, y_train)
        
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        performance = self._calculate_metrics(y_test, y_pred_proba)
        
        # Store scaler for later use
        self.scalers['logistic_regression'] = scaler
        
        return {
            'best_model': model,
            'best_model_name': 'logistic_regression',
            'performance': performance,
            'scaler': scaler
        }
    
    def _tune_hyperparameters(self, model, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int):
        """Tune hyperparameters for a given model."""
        param_grids = {
            'RandomForestClassifier': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'XGBClassifier': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'LGBMClassifier': {
                'n_estimators': [100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        }
        
        model_name = type(model).__name__
        param_grid = param_grids.get(model_name, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=cv_folds, scoring='roc_auc',
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            brier_score_loss, log_loss
        )
        
        # Convert probabilities to predictions
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'average_precision': average_precision_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }
        
        return {k: round(v, 4) for k, v in metrics.items()}
    
    def _analyze_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance from the trained model."""
        try:
            # Get feature importance based on model type
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = abs(model.coef_[0])
            else:
                # For voting classifier, use average importance
                if hasattr(model, 'estimators_'):
                    importances_list = []
                    for estimator in model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            importances_list.append(estimator.feature_importances_)
                    importances = np.mean(importances_list, axis=0) if importances_list else np.zeros(len(feature_names))
                else:
                    importances = np.zeros(len(feature_names))
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': feature_importance,
                'top_10_features': dict(sorted_features[:10]),
                'feature_rankings': [f[0] for f in sorted_features]
            }
            
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {str(e)}")
            return {'feature_importance': {}, 'top_10_features': {}, 'feature_rankings': []}
    
    def _calibrate_model(self, model, X_train: np.ndarray, y_train: np.ndarray):
        """Calibrate model probabilities using Platt scaling."""
        try:
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            calibrated_model.fit(X_train, y_train)
            return calibrated_model
        except Exception as e:
            logger.warning(f"Model calibration failed: {str(e)}")
            return model
    
    def _save_model_artifacts(self, model, model_results: Dict, feature_names: List[str], 
                             feature_importance: Dict) -> Dict[str, str]:
        """Save trained model and associated artifacts."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artifacts_dir = Path("models/trained")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main model
            model_path = artifacts_dir / f"risk_prediction_model_{timestamp}.pkl"
            joblib.dump(model, model_path)
            
            # Save feature names
            features_path = artifacts_dir / f"feature_names_{timestamp}.pkl"
            joblib.dump(feature_names, features_path)
            
            # Save feature importance
            importance_path = artifacts_dir / f"feature_importance_{timestamp}.json"
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2, default=str)
            
            # Save model metadata
            metadata = {
                'model_type': model_results.get('best_model_name', 'unknown'),
                'performance_metrics': model_results.get('performance', {}),
                'feature_count': len(feature_names),
                'training_timestamp': timestamp
            }
            
            metadata_path = artifacts_dir / f"model_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'model_path': str(model_path),
                'features_path': str(features_path),
                'importance_path': str(importance_path),
                'metadata_path': str(metadata_path)
            }
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {str(e)}")
            return {}
