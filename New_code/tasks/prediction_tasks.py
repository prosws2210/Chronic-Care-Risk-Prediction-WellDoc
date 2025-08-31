"""
Chronic Care Risk Prediction Task Definitions
=============================================

Comprehensive task definitions for the AI-driven risk prediction engine.
Each task is designed with clinical expertise and follows healthcare best practices.
"""

import os
import sys
from datetime import datetime
from crewai import Task
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.settings import PREDICTION_WINDOW, MIN_AUROC, MIN_AUPRC

# ---- Task Creation Functions ----

def create_prediction_tasks(agents: Dict) -> Dict[str, Task]:
    """
    Create all tasks for the chronic care risk prediction pipeline
    
    Args:
        agents: Dictionary of CrewAI agents
        
    Returns:
        Dictionary of initialized Task objects
    """
    
    # Ensure output directory exists
    os.makedirs("outputs/reports", exist_ok=True)
    
    tasks = {
        'data_preprocessing_task': create_data_preprocessing_task(agents.get('data_processor')),
        'model_training_task': create_model_training_task(agents.get('risk_assessor')),
        'model_evaluation_task': create_model_evaluation_task(agents.get('evaluator')),
        'explanation_task': create_explanation_task(agents.get('explainer')),
        'clinical_validation_task': create_clinical_validation_task(agents.get('clinical_validator')),
        'visualization_task': create_visualization_task(agents.get('visualizer'))
    }
    
    return tasks

def create_data_preprocessing_task(agent) -> Task:
    """Create comprehensive data preprocessing task"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "🔬 COMPREHENSIVE CHRONIC CARE DATA PREPROCESSING\n\n"
            
            "📋 PRIMARY OBJECTIVES:\n"
            "- Transform raw chronic care patient data into ML-ready features\n"
            "- Ensure clinical validity and data quality throughout the pipeline\n"
            "- Create meaningful temporal features for 90-day deterioration prediction\n"
            "- Generate comprehensive data quality and patient cohort reports\n\n"
            
            "📊 DATA SOURCES & VALIDATION:\n"
            "- Load patient data from specified path or generate synthetic dataset (1000+ patients)\n"
            "- Validate clinical ranges: BP (60/40-250/150), Glucose (50-600), HbA1c (4-15)\n"
            "- Check temporal consistency across longitudinal measurements\n"
            "- Identify and flag biologically implausible value combinations\n"
            "- Generate patient data completeness scorecard\n\n"
            
            "⚕️ CLINICAL FEATURE ENGINEERING:\n"
            "- Temporal Features:\n"
            "  • 7/14/30/60/90-day rolling averages for all vitals\n"
            "  • Trend analysis: slopes and variability metrics\n"
            "  • Last observation carried forward (LOCF) for missing labs\n"
            "- Glucose Management Features:\n"
            "  • Glycemic variability: Coefficient of variation, MAGE\n"
            "  • Time-in-range calculations (70-180 mg/dL)\n"
            "  • HbA1c trajectory and rate of change\n"
            "- Blood Pressure Control:\n"
            "  • BP variability and control consistency\n"
            "  • Hypertensive crisis episodes identification\n"
            "  • Medication effectiveness indicators\n"
            "- Medication Adherence Patterns:\n"
            "  • PDC (Proportion of Days Covered) calculations\n"
            "  • Gap analysis and adherence trend identification\n"
            "  • Polypharmacy complexity scoring\n"
            "- Lifestyle & Behavioral Features:\n"
            "  • BMI trajectory and weight change velocity\n"
            "  • Exercise frequency patterns and consistency\n"
            "  • Sleep quality and duration trends\n\n"
            
            "🔧 ADVANCED DATA PROCESSING:\n"
            "- Missing Data Strategy Implementation:\n"
            "  • Vitals: Linear interpolation for gaps <7 days\n"
            "  • Labs: LOCF up to 90 days, then median imputation\n"
            "  • Medications: Assume discontinued if missing >30 days\n"
            "  • Lifestyle: Population stratified means (age/gender/condition)\n"
            "- Outlier Detection & Clinical Validation:\n"
            "  • Statistical outliers (3-sigma rule with clinical bounds)\n"
            "  • Clinical impossibility detection (systolic ≤ diastolic)\n"
            "  • Extreme value investigation and documentation\n"
            "- Feature Scaling & Normalization:\n"
            "  • StandardScaler for continuous variables\n"
            "  • Robust scaling for variables with outliers\n"
            "  • Categorical encoding with clinical hierarchy\n\n"
            
            "🎯 CHRONIC CONDITION INTERACTIONS:\n"
            "- Comorbidity Risk Multipliers:\n"
            "  • Diabetes + Hypertension interaction features\n"
            "  • Heart Failure + COPD severity combinations\n"
            "  • Obesity impact on diabetes/hypertension control\n"
            "- Condition-Specific Features:\n"
            "  • Diabetes: Insulin requirements, hypoglycemic episodes\n"
            "  • Heart Failure: Fluid retention indicators, exercise tolerance\n"
            "  • COPD: Exacerbation frequency, oxygen requirements\n\n"
            
            "📈 QUALITY ASSURANCE & REPORTING:\n"
            "- Data Quality Metrics:\n"
            "  • Completeness rates by feature and patient\n"
            "  • Clinical range adherence percentages\n"
            "  • Temporal consistency validation results\n"
            "- Patient Cohort Characterization:\n"
            "  • Demographics and comorbidity distributions\n"
            "  • Baseline risk stratification\n"
            "  • Care complexity scoring\n"
            "- Processing Pipeline Documentation:\n"
            "  • Transformation steps and clinical rationales\n"
            "  • Feature dictionary with clinical interpretations\n"
            "  • Data lineage and processing timestamps\n\n"
            
            "🎪 SUCCESS CRITERIA:\n"
            "- Clean, ML-ready dataset with >95% data quality score\n"
            "- Clinically meaningful features aligned with chronic care guidelines\n"
            "- Comprehensive documentation suitable for clinical review\n"
            "- Processing pipeline ready for production deployment\n\n"
            
            "OUTPUT DELIVERABLES:\n"
            "1. Processed dataset (CSV) with engineered features\n"
            "2. Data quality report with clinical validation results\n"
            "3. Feature dictionary with clinical interpretations\n"
            "4. Patient cohort summary statistics\n"
            "5. Processing pipeline documentation"
        ),
        expected_output=(
            "Comprehensive preprocessing package including:\n"
            "• Processed dataset location and feature summary\n"
            "• Data quality validation report with clinical ranges\n"
            "• Feature engineering documentation\n"
            "• Patient cohort characteristics\n"
            "• Pipeline performance metrics and recommendations"
        ),
        agent=agent,
        output_file=f"outputs/reports/preprocessing_report_{timestamp}.json",
        tools=None,  # Tools will be assigned via agent
        dependencies=[]
    )

def create_model_training_task(agent) -> Task:
    """Create comprehensive model training and optimization task"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "🤖 ADVANCED CLINICAL RISK PREDICTION MODEL TRAINING\n\n"
            
            "🎯 MISSION OBJECTIVES:\n"
            "- Develop high-performance ML model for 90-day deterioration prediction\n"
            "- Achieve clinical-grade performance (AUROC ≥0.75, AUPRC ≥0.65)\n"
            "- Optimize for patient safety (Sensitivity ≥80%) and workflow efficiency\n"
            "- Create interpretable, calibrated probability predictions for clinical use\n\n"
            
            "📊 DATA PREPARATION & SPLITTING:\n"
            "- Load preprocessed chronic care dataset with engineered features\n"
            "- Implement stratified train/validation/test splits (70/15/15)\n"
            "- Ensure temporal integrity: train on earlier data, test on recent\n"
            "- Balance classes using clinical risk-aware sampling strategies\n"
            "- Create separate holdout set for final clinical validation\n\n"
            
            "⚗️ MULTI-ALGORITHM ENSEMBLE APPROACH:\n"
            "- Primary Models to Train:\n"
            "  • Random Forest: Interpretable baseline with feature importance\n"
            "  • XGBoost: High-performance gradient boosting\n"
            "  • LightGBM: Efficient gradient boosting for large datasets\n"
            "  • Neural Network: Deep learning for complex pattern recognition\n"
            "  • Logistic Regression: Linear baseline for comparison\n"
            "- Advanced Ensemble Methods:\n"
            "  • Stacking with meta-learner for optimal combination\n"
            "  • Weighted voting based on validation performance\n"
            "  • Bayesian model averaging for uncertainty quantification\n\n"
            
            "🔧 HYPERPARAMETER OPTIMIZATION:\n"
            "- Optimization Framework:\n"
            "  • Bayesian optimization (Optuna/Hyperopt) for efficiency\n"
            "  • Multi-objective optimization: accuracy vs. interpretability\n"
            "  • Clinical constraint integration (minimum sensitivity requirements)\n"
            "- Model-Specific Tuning:\n"
            "  • Random Forest: n_estimators, max_depth, min_samples_split\n"
            "  • XGBoost: learning_rate, max_depth, subsample, colsample_bytree\n"
            "  • Neural Network: architecture, dropout, learning_rate, batch_size\n"
            "- Cross-Validation Strategy:\n"
            "  • Stratified K-Fold (5 folds) with temporal awareness\n"
            "  • Time series cross-validation for longitudinal data\n"
            "  • Blocked cross-validation to prevent data leakage\n\n"
            
            "🩺 CLINICAL OPTIMIZATION PRIORITIES:\n"
            "- Patient Safety Metrics (Primary):\n"
            "  • Sensitivity (Recall) ≥80% - minimize dangerous false negatives\n"
            "  • Negative Predictive Value ≥95% - safe rule-out capability\n"
            "  • High-risk precision - reliable positive predictions\n"
            "- Operational Efficiency Metrics:\n"
            "  • Specificity ≥75% - reduce alert fatigue and false alarms\n"
            "  • Positive Predictive Value - actionable positive predictions\n"
            "  • F1-Score optimization for imbalanced clinical data\n"
            "- Clinical Utility Optimization:\n"
            "  • AUROC ≥0.75 - overall discrimination capability\n"
            "  • AUPRC ≥0.65 - performance on imbalanced deterioration events\n"
            "  • Brier Score ≤0.15 - well-calibrated probability predictions\n\n"
            
            "📊 ADVANCED MODEL TECHNIQUES:\n"
            "- Class Imbalance Handling:\n"
            "  • SMOTE/ADASYN for synthetic minority class generation\n"
            "  • Class weights based on clinical cost-benefit analysis\n"
            "  • Focal loss for hard example mining\n"
            "- Model Calibration:\n"
            "  • Platt scaling for probability calibration\n"
            "  • Isotonic regression for non-parametric calibration\n"
            "  • Temperature scaling for neural network calibration\n"
            "- Uncertainty Quantification:\n"
            "  • Bootstrap aggregation for prediction intervals\n"
            "  • Monte Carlo dropout for neural network uncertainty\n"
            "  • Bayesian methods for model uncertainty estimation\n\n"
            
            "🔍 FEATURE IMPORTANCE & SELECTION:\n"
            "- Global Feature Importance:\n"
            "  • Permutation importance for model-agnostic insights\n"
            "  • SHAP values for consistent feature attribution\n"
            "  • Recursive feature elimination with cross-validation\n"
            "- Clinical Feature Validation:\n"
            "  • Alignment with established clinical risk factors\n"
            "  • Correlation analysis with known biomarkers\n"
            "  • Clinical expert review of feature rankings\n"
            "- Feature Stability Analysis:\n"
            "  • Cross-validation feature importance consistency\n"
            "  • Bootstrap feature importance confidence intervals\n"
            "  • Temporal stability across different time periods\n\n"
            
            "💾 MODEL ARTIFACTS & DEPLOYMENT:\n"
            "- Model Persistence:\n"
            "  • Save best performing model with versioning\n"
            "  • Export preprocessing pipelines and scalers\n"
            "  • Create ONNX models for production deployment\n"
            "- Documentation Package:\n"
            "  • Model architecture and hyperparameter documentation\n"
            "  • Training procedure and validation methodology\n"
            "  • Performance benchmarks and clinical interpretation\n"
            "- Production Readiness:\n"
            "  • Inference time optimization (<100ms per prediction)\n"
            "  • Memory usage profiling and optimization\n"
            "  • API-ready model serving preparation\n\n"
            
            f"🎯 PERFORMANCE TARGETS:\n"
            "- Minimum Acceptable Performance:\n"
            f"  • AUROC ≥ {MIN_AUROC} (Area Under ROC Curve)\n"
            f"  • AUPRC ≥ {MIN_AUPRC} (Area Under Precision-Recall Curve)\n"
            "  • Sensitivity ≥ 0.80 (True Positive Rate)\n"
            "  • Specificity ≥ 0.75 (True Negative Rate)\n"
            "- Excellence Targets:\n"
            "  • AUROC ≥ 0.85 (Excellent discrimination)\n"
            "  • AUPRC ≥ 0.75 (Strong precision-recall performance)\n"
            "  • Brier Score ≤ 0.10 (Excellent calibration)\n"
            "  • Calibration slope: 0.9-1.1 (Well-calibrated probabilities)\n\n"
            
            "DELIVERABLES:\n"
            "1. Trained ensemble model with optimal hyperparameters\n"
            "2. Comprehensive performance evaluation on validation set\n"
            "3. Feature importance rankings with clinical interpretations\n"
            "4. Model artifacts (pickled models, scalers, pipelines)\n"
            "5. Training documentation and reproducibility package"
        ),
        expected_output=(
            "Complete model training package including:\n"
            "• Best performing model with performance metrics exceeding clinical thresholds\n"
            "• Cross-validation results demonstrating model stability\n"
            "• Feature importance analysis with clinical validation\n"
            "• Calibration assessment and probability reliability analysis\n"
            "• Production-ready model artifacts and deployment documentation"
        ),
        agent=agent,
        output_file=f"outputs/reports/model_training_report_{timestamp}.json",
        tools=None,
        dependencies=[]
    )

def create_model_evaluation_task(agent) -> Task:
    """Create comprehensive model evaluation and validation task"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "📊 COMPREHENSIVE CLINICAL MODEL EVALUATION & VALIDATION\n\n"
            
            "🎯 EVALUATION MISSION:\n"
            "- Conduct rigorous clinical and statistical evaluation of risk prediction model\n"
            "- Assess performance across diverse patient populations and scenarios\n"
            "- Validate clinical utility and real-world deployment readiness\n"
            "- Generate regulatory-compliant validation documentation\n\n"
            
            "📈 PRIMARY PERFORMANCE METRICS:\n"
            "- Discrimination Metrics:\n"
            "  • AUROC (Area Under ROC Curve): Overall discrimination capability\n"
            "  • AUPRC (Area Under PR Curve): Performance on imbalanced data\n"
            "  • C-Index: Concordance index for risk ranking\n"
            "  • Gini Coefficient: Alternative discrimination measure\n"
            "- Calibration Metrics:\n"
            "  • Brier Score: Overall prediction accuracy and calibration\n"
            "  • Hosmer-Lemeshow test: Goodness-of-fit assessment\n"
            "  • Calibration slope and intercept: Probability reliability\n"
            "  • Integrated Calibration Index (ICI): Modern calibration metric\n"
            "- Clinical Performance Metrics:\n"
            "  • Sensitivity (True Positive Rate): Patient safety focus\n"
            "  • Specificity (True Negative Rate): Alert fatigue prevention\n"
            "  • Positive Predictive Value: Actionable positive predictions\n"
            "  • Negative Predictive Value: Safe rule-out capability\n"
            "  • F1-Score: Balanced performance measure\n\n"
            
            "🏥 CLINICAL UTILITY ASSESSMENT:\n"
            "- Decision Curve Analysis (DCA):\n"
            "  • Net benefit calculation across risk thresholds\n"
            "  • Clinical intervention threshold optimization\n"
            "  • Comparison with standard care pathways\n"
            "- Number Needed to Screen/Treat:\n"
            "  • Cost-effectiveness analysis for interventions\n"
            "  • Resource allocation optimization\n"
            "  • Healthcare economics impact assessment\n"
            "- Clinical Impact Modeling:\n"
            "  • Hospital readmission reduction potential\n"
            "  • Emergency department visit prevention\n"
            "  • Quality-adjusted life years (QALY) improvement\n\n"
            
            "👥 SUBGROUP ANALYSIS & FAIRNESS:\n"
            "- Demographic Subgroups:\n"
            "  • Age stratification: <50, 50-65, 65-80, >80 years\n"
            "  • Gender performance: Male vs Female prediction accuracy\n"
            "  • Racial/ethnic groups: Bias detection and mitigation\n"
            "  • Socioeconomic status: Insurance, geographic, access factors\n"
            "- Clinical Subgroups:\n"
            "  • Primary chronic conditions: Diabetes, CHF, COPD, obesity\n"
            "  • Comorbidity combinations: Multi-condition interactions\n"
            "  • Disease severity levels: Mild, moderate, severe presentations\n"
            "  • Treatment complexity: Medication burden, specialist care\n"
            "- Algorithmic Fairness Metrics:\n"
            "  • Equalized odds: Equal TPR and FPR across groups\n"
            "  • Demographic parity: Equal positive prediction rates\n"
            "  • Individual fairness: Similar predictions for similar patients\n\n"
            
            "⏰ TEMPORAL VALIDATION ANALYSIS:\n"
            "- Cross-Temporal Performance:\n"
            "  • Train on historical data, test on recent data\n"
            "  • Seasonal variation analysis: Account for healthcare patterns\n"
            "  • Long-term stability assessment: Performance drift detection\n"
            "- Prediction Horizon Analysis:\n"
            f"  • Primary target: {PREDICTION_WINDOW}-day deterioration prediction\n"
            "  • Alternative horizons: 30, 60, 120-day predictions\n"
            "  • Time-to-event analysis: Survival curve validation\n"
            "- Concept Drift Detection:\n"
            "  • Statistical tests for distribution changes\n"
            "  • Performance degradation monitoring\n"
            "  • Retraining trigger identification\n\n"
            
            "🔬 ADVANCED STATISTICAL ANALYSIS:\n"
            "- Bootstrap Confidence Intervals:\n"
            "  • Metric uncertainty quantification\n"
            "  • Robust performance estimates\n"
            "  • Statistical significance testing\n"
            "- Cross-Validation Robustness:\n"
            "  • Stratified k-fold performance consistency\n"
            "  • Leave-one-group-out validation\n"
            "  • Nested cross-validation for unbiased estimates\n"
            "- Sensitivity Analysis:\n"
            "  • Performance under different prevalence rates\n"
            "  • Threshold sensitivity for clinical decisions\n"
            "  • Robustness to missing data scenarios\n\n"
            
            "📊 VISUALIZATION SUITE GENERATION:\n"
            "- Performance Plots:\n"
            "  • ROC curves with confidence bands\n"
            "  • Precision-Recall curves with baselines\n"
            "  • Calibration plots with perfect calibration reference\n"
            "  • Confusion matrices with clinical annotations\n"
            "- Clinical Decision Plots:\n"
            "  • Decision curve analysis visualizations\n"
            "  • Risk threshold optimization curves\n"
            "  • Clinical utility comparison charts\n"
            "- Fairness and Bias Visualizations:\n"
            "  • Subgroup performance comparison charts\n"
            "  • Bias detection heatmaps\n"
            "  • Demographic parity assessment plots\n\n"
            
            "⚖️ REGULATORY & COMPLIANCE VALIDATION:\n"
            "- FDA Software as Medical Device (SaMD) Assessment:\n"
            "  • Risk categorization and validation requirements\n"
            "  • Clinical evidence documentation\n"
            "  • Predicate device comparison analysis\n"
            "- Clinical Trial Design Recommendations:\n"
            "  • Prospective validation study protocol\n"
            "  • Sample size calculations for clinical endpoints\n"
            "  • Statistical analysis plan for regulatory submission\n"
            "- Quality Management System Documentation:\n"
            "  • Validation protocols and acceptance criteria\n"
            "  • Risk management and mitigation strategies\n"
            "  • Change control and version management procedures\n\n"
            
            "🎯 PERFORMANCE BENCHMARKING:\n"
            "- Literature Comparison:\n"
            "  • Comparison with published chronic care prediction models\n"
            "  • Benchmark against clinical risk scores (Framingham, ASCVD)\n"
            "  • Performance relative to existing clinical decision support tools\n"
            "- Internal Benchmarking:\n"
            "  • Comparison with simpler baseline models\n"
            "  • Expert clinician prediction comparison\n"
            "  • Current standard of care performance assessment\n\n"
            
            "DELIVERABLES:\n"
            "1. Comprehensive evaluation report with all performance metrics\n"
            "2. Subgroup analysis and fairness assessment\n"
            "3. Clinical utility and decision curve analysis\n"
            "4. Regulatory compliance documentation\n"
            "5. Performance visualization suite\n"
            "6. Recommendations for clinical deployment"
        ),
        expected_output=(
            "Comprehensive evaluation package including:\n"
            "• Primary performance metrics with confidence intervals\n"
            "• Subgroup analysis revealing any performance disparities\n"
            "• Clinical utility assessment with decision curve analysis\n"
            "• Calibration evaluation and probability reliability assessment\n"
            "• Regulatory compliance documentation for clinical deployment\n"
            "• Visualization suite supporting clinical interpretation"
        ),
        agent=agent,
        output_file=f"outputs/reports/evaluation_report_{timestamp}.json",
        tools=None,
        dependencies=[]
    )

def create_explanation_task(agent) -> Task:
    """Create comprehensive AI explanation and interpretation task"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "🧠 CLINICAL AI EXPLAINABILITY & INTERPRETATION SYSTEM\n\n"
            
            "💡 EXPLAINABILITY MISSION:\n"
            "- Transform complex ML predictions into clear, actionable clinical insights\n"
            "- Generate evidence-based explanations that support clinical decision-making\n"
            "- Create both population-level and patient-specific interpretations\n"
            "- Ensure explanations align with clinical reasoning and medical evidence\n\n"
            
            "🌍 GLOBAL EXPLANATIONS (Population-Level Insights):\n"
            "- Feature Importance Analysis:\n"
            "  • SHAP feature importance across entire patient population\n"
            "  • Permutation importance for model-agnostic insights\n"
            "  • Recursive feature elimination rankings\n"
            "  • Stability analysis across different data subsets\n"
            "- Clinical Risk Factor Hierarchy:\n"
            "  • Top 15 most influential clinical features\n"
            "  • Modifiable vs. non-modifiable risk factor classification\n"
            "  • Risk factor clustering and interaction analysis\n"
            "  • Evidence-based clinical context for each factor\n"
            "- Population Risk Patterns:\n"
            "  • Common risk profiles and patient archetypes\n"
            "  • Seasonal and temporal risk factor variations\n"
            "  • Chronic condition interaction patterns\n"
            "  • Healthcare utilization impact analysis\n\n"
            
            "👤 LOCAL EXPLANATIONS (Patient-Specific Insights):\n"
            "- Individual Risk Attribution:\n"
            "  • SHAP waterfall plots for each high-risk patient\n"
            "  • Feature contribution calculations with confidence intervals\n"
            "  • Patient vs. population comparison analysis\n"
            "  • Risk factor deviation quantification (z-scores)\n"
            "- Personalized Clinical Narratives:\n"
            "  • Auto-generated clinical summaries for each patient\n"
            "  • Risk factor prioritization based on modifiability\n"
            "  • Timeline analysis: How risk factors evolved over time\n"
            "  • Intervention opportunity identification\n"
            "- Counterfactual Analysis:\n"
            "  • 'What-if' scenarios for risk reduction strategies\n"
            "  • Minimum changes needed to reduce risk category\n"
            "  • Impact of specific interventions on risk score\n"
            "  • Alternative care pathway risk projections\n\n"
            
            "🩺 CLINICAL TRANSLATION & INTERPRETATION:\n"
            "- Medical Terminology Translation:\n"
            "  • Convert technical features to clinical language:\n"
            "    - 'glucose_cv' → 'Blood glucose variability'\n"
            "    - 'bp_variability' → 'Blood pressure control consistency'\n"
            "    - 'medication_adherence' → 'Treatment compliance patterns'\n"
            "    - 'hba1c_trend' → 'Long-term diabetes control trajectory'\n"
            "- Clinical Context Integration:\n"
            "  • ADA diabetes management guidelines integration\n"
            "  • ACC/AHA heart failure recommendations alignment\n"
            "  • Hypertension treatment pathway references\n"
            "  • Evidence-based medicine citations for risk factors\n"
            "- Risk Factor Clinical Significance:\n"
            "  • Clinical thresholds and target ranges explanation\n"
            "  • Prognostic implications of each risk factor\n"
            "  • Interaction effects between multiple conditions\n"
            "  • Long-term outcome predictions based on current risk\n\n"
            
            "🎯 ACTIONABLE INTERVENTION RECOMMENDATIONS:\n"
            "- High-Priority Interventions:\n"
            "  • Medication optimization opportunities:\n"
            "    - Diabetes: Insulin adjustment, CGM implementation\n"
            "    - Hypertension: ACE inhibitor titration, DASH diet\n"
            "    - Heart failure: Diuretic optimization, fluid management\n"
            "  • Lifestyle modification targets:\n"
            "    - Weight management: Caloric restriction, exercise prescription\n"
            "    - Smoking cessation: NRT, counseling, pharmacotherapy\n"
            "    - Sleep hygiene: Sleep study referral, CPAP therapy\n"
            "- Care Team Coordination:\n"
            "  • Specialist referral recommendations with urgency levels\n"
            "  • Care team communication templates\n"
            "  • Patient education material suggestions\n"
            "  • Monitoring frequency optimization\n"
            "- Timeline and Priority Matrix:\n"
            "  • Immediate interventions (<1 week)\n"
            "  • Short-term goals (1-4 weeks)\n"
            "  • Medium-term objectives (1-3 months)\n"
            "  • Long-term management strategies (>3 months)\n\n"
            
            "📚 EVIDENCE-BASED CONTEXTUALIZATION:\n"
            "- Clinical Guideline Integration:\n"
            "  • ADA Standards of Medical Care alignment\n"
            "  • ACC/AHA Heart Failure guidelines compliance\n"
            "  • JNC 8 Hypertension management recommendations\n"
            "  • CDC Chronic Disease Prevention guidelines\n"
            "- Literature Evidence Support:\n"
            "  • Relevant clinical trial citations for interventions\n"
            "  • Meta-analysis support for risk factor importance\n"
            "  • Real-world evidence for intervention effectiveness\n"
            "  • Cost-effectiveness analysis references\n"
            "- Clinical Decision Support Integration:\n"
            "  • EHR-compatible explanation formats\n"
            "  • Clinical alert integration recommendations\n"
            "  • Workflow-optimized explanation delivery\n"
            "  • Decision fatigue minimization strategies\n\n"
            
            "👨‍⚕️ MULTI-AUDIENCE EXPLANATION FORMATS:\n"
            "- For Physicians:\n"
            "  • Technical accuracy with statistical confidence measures\n"
            "  • Clinical guideline references and evidence citations\n"
            "  • Differential diagnosis considerations\n"
            "  • Treatment algorithm recommendations\n"
            "- For Nurses and Care Coordinators:\n"
            "  • Practical monitoring instructions and red flags\n"
            "  • Patient education talking points\n"
            "  • Care coordination workflow integration\n"
            "  • Family communication guidance\n"
            "- For Patients and Families:\n"
            "  • Simple, jargon-free language explanations\n"
            "  • Visual aids and analogies for complex concepts\n"
            "  • Actionable self-management steps\n"
            "  • Motivation and empowerment messaging\n\n"
            
            "📊 EXPLANATION VISUALIZATION SUITE:\n"
            "- Clinical Decision Support Visuals:\n"
            "  • Risk factor contribution bar charts\n"
            "  • Patient trajectory timelines\n"
            "  • Intervention impact projections\n"
            "  • Comparative risk scenario displays\n"
            "- Interactive Explanation Interfaces:\n"
            "  • Drill-down feature importance exploration\n"
            "  • Dynamic 'what-if' scenario calculators\n"
            "  • Patient comparison tools\n"
            "  • Risk factor trend visualizations\n"
            "- Print-Ready Clinical Reports:\n"
            "  • One-page patient risk summaries\n"
            "  • Intervention recommendation sheets\n"
            "  • Clinical documentation templates\n"
            "  • Patient education handouts\n\n"
            
            "🔍 EXPLANATION QUALITY ASSURANCE:\n"
            "- Clinical Validation:\n"
            "  • Expert clinician review of explanation accuracy\n"
            "  • Alignment with established clinical reasoning\n"
            "  • Bias detection in explanation generation\n"
            "  • Cultural sensitivity assessment\n"
            "- Explanation Consistency:\n"
            "  • Cross-validation of explanation stability\n"
            "  • Patient similarity explanation comparison\n"
            "  • Temporal consistency across data updates\n"
            "  • Multi-model explanation agreement\n\n"
            
            "DELIVERABLES:\n"
            "1. Global feature importance report with clinical interpretations\n"
            "2. Patient-level SHAP explanation package\n"
            "3. Clinical intervention recommendation engine\n"
            "4. Multi-audience explanation templates\n"
            "5. Interactive explanation dashboard components\n"
            "6. Evidence-based clinical context database"
        ),
        expected_output=(
            "Comprehensive explanation system including:\n"
            "• Global feature importance with clinical significance rankings\n"
            "• Patient-specific SHAP explanations with clinical narratives\n"
            "• Actionable intervention recommendations prioritized by impact\n"
            "• Multi-audience explanation formats (physicians, nurses, patients)\n"
            "• Evidence-based clinical context and guideline alignment\n"
            "• Interactive visualization components for clinical decision support"
        ),
        agent=agent,
        output_file=f"outputs/reports/explanation_report_{timestamp}.json",
        tools=None,
        dependencies=[]
    )

def create_clinical_validation_task(agent) -> Task:
    """Create comprehensive clinical validation and safety assessment task"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "🩺 COMPREHENSIVE CLINICAL VALIDATION & SAFETY ASSESSMENT\n\n"
            
            "⚖️ CLINICAL VALIDATION MISSION:\n"
            "- Ensure AI predictions align with established clinical guidelines and evidence\n"
            "- Validate safety and appropriateness of recommendations for patient care\n"
            "- Identify and mitigate potential biases in AI-driven clinical decisions\n"
            "- Confirm clinical workflow integration feasibility and safety\n\n"
            
            "📋 CLINICAL GUIDELINE ALIGNMENT VERIFICATION:\n"
            "- Diabetes Management (ADA Guidelines):\n"
            "  • HbA1c target validation: <7% for most adults, individualized targets\n"
            "  • Glucose monitoring recommendations: CGM vs SMBG appropriateness\n"
            "  • Medication intensification thresholds: Metformin → insulin pathway\n"
            "  • Hypoglycemia risk assessment and mitigation strategies\n"
            "  • Diabetic complication screening schedule compliance\n"
            "- Heart Failure Management (ACC/AHA Guidelines):\n"
            "  • NYHA class progression prediction accuracy\n"
            "  • Ejection fraction considerations: HFrEF vs HFpEF management\n"
            "  • Guideline-directed medical therapy (GDMT) optimization\n"
            "  • Fluid management and diuretic titration recommendations\n"
            "  • Device therapy candidacy assessment (ICD, CRT)\n"
            "- Hypertension Control (AHA/ACC 2017):\n"
            "  • Blood pressure targets: <130/80 for most, <140/90 for elderly\n"
            "  • Medication stepwise therapy: ACE-I → ARB → CCB → diuretic\n"
            "  • Home BP monitoring recommendations\n"
            "  • Lifestyle modification priority ranking\n"
            "  • Resistant hypertension identification and management\n"
            "- Obesity Management (AHA/ACC/TOS Guidelines):\n"
            "  • BMI and waist circumference risk stratification\n"
            "  • Weight loss goal setting: 5-10% initial target\n"
            "  • Bariatric surgery candidacy criteria\n"
            "  • Pharmacotherapy initiation thresholds\n"
            "  • Comorbidity-focused treatment prioritization\n\n"
            
            "⚠️ CLINICAL SAFETY VALIDATION:\n"
            "- Biologically Implausible Prediction Detection:\n"
            "  • Glucose levels: Flag predictions <50 or >600 mg/dL without DKA context\n"
            "  • Blood pressure: Identify systolic <70 or >250, diastolic <40 or >150\n"
            "  • Heart rate: Detect rates <40 or >150 without clinical context\n"
            "  • BMI: Flag extreme values <15 or >60 without documented conditions\n"
            "  • Age-inappropriate predictions: Pediatric thresholds in adults\n"
            "- Dangerous False Negative Analysis:\n"
            "  • High-risk patients incorrectly classified as low-risk\n"
            "  • Critical symptom combinations missed by the model\n"
            "  • Emergency department presentation risk underestimation\n"
            "  • Medication contraindication oversight detection\n"
            "- Drug Interaction and Contraindication Validation:\n"
            "  • Major drug-drug interaction screening\n"
            "  • Allergy and adverse reaction history consideration\n"
            "  • Renal/hepatic dose adjustment requirements\n"
            "  • Age-appropriate medication recommendations\n"
            "  • Polypharmacy risk assessment and optimization\n\n"
            
            "🔬 EVIDENCE-BASED MEDICINE VALIDATION:\n"
            "- Risk Factor Clinical Evidence Review:\n"
            "  • Framingham Risk Score component alignment\n"
            "  • ASCVD Risk Calculator consistency\n"
            "  • WHO/CDC chronic disease risk factor validation\n"
            "  • Landmark clinical trial outcome correlation\n"
            "- Intervention Recommendation Evidence Support:\n"
            "  • Cochrane systematic review alignment\n"
            "  • Meta-analysis support for recommended interventions\n"
            "  • Number-needed-to-treat (NNT) consideration\n"
            "  • Cost-effectiveness analysis integration\n"
            "- Clinical Prediction Rule Comparison:\n"
            "  • Comparison with established clinical scores:\n"
            "    - CHADS2-VASc for stroke risk\n"
            "    - GRACE score for ACS outcomes\n"
            "    - STOP-BANG for sleep apnea\n"
            "    - Framingham for cardiovascular risk\n\n"
            
            "⚖️ ALGORITHMIC FAIRNESS & BIAS ASSESSMENT:\n"
            "- Demographic Bias Analysis:\n"
            "  • Age bias: Ensure appropriate risk assessment across age groups\n"
            "  • Gender bias: Validate equitable performance for male/female patients\n"
            "  • Racial/ethnic bias: Screen for systematic disparities\n"
            "  • Socioeconomic bias: Address insurance and access-related disparities\n"
            "- Clinical Bias Detection:\n"
            "  • Disease severity bias: Validate performance across severity spectrums\n"
            "  • Comorbidity bias: Ensure fair treatment of multi-condition patients\n"
            "  • Treatment complexity bias: Address care intensity variations\n"
            "  • Geographic bias: Rural vs urban care access considerations\n"
            "- Historical Bias Mitigation:\n"
            "  • Legacy healthcare disparities correction\n"
            "  • Systematic exclusion pattern identification\n"
            "  • Representation adequacy assessment\n"
            "  • Bias amplification prevention strategies\n\n"
            
            "🏥 CLINICAL WORKFLOW INTEGRATION VALIDATION:\n"
            "- Care Team Workflow Assessment:\n"
            "  • Primary care physician workflow integration\n"
            "  • Specialist referral pathway optimization\n"
            "  • Nursing care plan integration feasibility\n"
            "  • Care coordinator task assignment appropriateness\n"
            "- Clinical Decision Support Integration:\n"
            "  • EHR system compatibility validation\n"
            "  • Clinical alert optimization: Reduce alert fatigue\n"
            "  • Documentation burden assessment\n"
            "  • Quality metric alignment (HEDIS, CMS measures)\n"
            "- Patient Care Pathway Validation:\n"
            "  • Care transition risk assessment accuracy\n"
            "  • Discharge planning optimization\n"
            "  • Follow-up scheduling appropriateness\n"
            "  • Self-management support alignment\n\n"
            
            "🎯 CLINICAL ACTIONABILITY ASSESSMENT:\n"
            "- Intervention Feasibility Analysis:\n"
            "  • Resource availability assessment: Can recommendations be implemented?\n"
            "  • Time constraint evaluation: Realistic within clinical encounters?\n"
            "  • Cost consideration: Insurance coverage and patient affordability\n"
            "  • Specialist availability: Referral capacity and wait times\n"
            "- Patient Engagement and Compliance:\n"
            "  • Health literacy level appropriateness\n"
            "  • Cultural sensitivity assessment\n"
            "  • Language barrier consideration\n"
            "  • Shared decision-making framework alignment\n"
            "- Quality Improvement Integration:\n"
            "  • Population health management alignment\n"
            "  • Value-based care metric contribution\n"
            "  • Clinical outcome improvement potential\n"
            "  • Healthcare cost reduction opportunity assessment\n\n"
            
            "🔍 REGULATORY COMPLIANCE & LIABILITY ASSESSMENT:\n"
            "- Medical Device Regulation Compliance:\n"
            "  • FDA Software as Medical Device (SaMD) classification\n"
            "  • Clinical validation requirements fulfillment\n"
            "  • Quality management system alignment\n"
            "  • Post-market surveillance plan adequacy\n"
            "- Clinical Liability Risk Assessment:\n"
            "  • Malpractice risk evaluation for AI recommendations\n"
            "  • Standard of care compliance verification\n"
            "  • Informed consent integration requirements\n"
            "  • Professional liability coverage considerations\n"
            "- Data Privacy and Security Validation:\n"
            "  • HIPAA compliance verification\n"
            "  • Patient data protection adequacy\n"
            "  • Consent management appropriateness\n"
            "  • Data breach risk mitigation assessment\n\n"
            
            "⚕️ CLINICAL EXPERT REVIEW PROCESS:\n"
            "- Multi-Disciplinary Expert Panel:\n"
            "  • Board-certified physicians: Internal medicine, endocrinology, cardiology\n"
            "  • Clinical pharmacists: Medication optimization experts\n"
            "  • Nurse practitioners: Primary care and chronic disease management\n"
            "  • Health informaticians: Clinical workflow and EHR integration\n"
            "- Systematic Clinical Case Review:\n"
            "  • High-risk patient prediction validation\n"
            "  • Edge case scenario evaluation\n"
            "  • Unusual presentation assessment\n"
            "  • Clinical reasoning alignment verification\n"
            "- Continuous Quality Improvement:\n"
            "  • Clinical feedback integration mechanism\n"
            "  • Model performance monitoring in clinical practice\n"
            "  • Adverse event reporting and analysis\n"
            "  • Clinical outcome tracking and validation\n\n"
            
            "DELIVERABLES:\n"
            "1. Clinical guideline compliance assessment report\n"
            "2. Safety validation and risk mitigation documentation\n"
            "3. Bias detection and fairness analysis\n"
            "4. Clinical workflow integration feasibility study\n"
            "5. Regulatory compliance and liability assessment\n"
            "6. Clinical expert review summary and recommendations"
        ),
        expected_output=(
            "Comprehensive clinical validation package including:\n"
            "• Guideline compliance verification with specific recommendations\n"
            "• Safety assessment identifying potential risks and mitigation strategies\n"
            "• Bias analysis with fairness improvement recommendations\n"
            "• Clinical workflow integration feasibility assessment\n"
            "• Regulatory compliance documentation for deployment approval\n"
            "• Multi-disciplinary clinical expert review summary"
        ),
        agent=agent,
        output_file=f"outputs/reports/clinical_validation_report_{timestamp}.json",
        tools=None,
        dependencies=[]
    )

def create_visualization_task(agent) -> Task:
    """Create comprehensive visualization and dashboard component task"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "📊 COMPREHENSIVE CLINICAL VISUALIZATION & DASHBOARD DEVELOPMENT\n\n"
            
            "🎨 VISUALIZATION MISSION:\n"
            "- Create intuitive, clinically-relevant visualizations for healthcare workflows\n"
            "- Design dashboard components optimized for rapid clinical decision-making\n"
            "- Generate publication-quality performance and analysis charts\n"
            "- Ensure accessibility and usability across diverse clinical environments\n\n"
            
            "🏥 CLINICAL DASHBOARD COMPONENTS:\n"
            "- Risk Assessment Display:\n"
            "  • Large, color-coded risk probability indicators\n"
            "  • Risk category badges: HIGH/MEDIUM/LOW with clinical context\n"
            "  • Confidence interval visualization with uncertainty representation\n"
            "  • Time-to-predicted-event countdown displays\n"
            "  • Risk trajectory trends with directional arrows\n"
            "- Patient Cohort Overview:\n"
            "  • Population risk distribution histograms\n"
            "  • Sortable, filterable patient lists with risk stratification\n"
            "  • Alert summary dashboards: Urgent cases requiring attention\n"
            "  • Population trend analysis: Risk changes over time\n"
            "  • Care team workload distribution visualizations\n"
            "- Individual Patient Deep-Dive:\n"
            "  • Comprehensive patient risk profiles with clinical context\n"
            "  • Feature contribution waterfall charts (SHAP visualizations)\n"
            "  • Vital signs trend analysis with normal range overlays\n"
            "  • Medication adherence pattern visualizations\n"
            "  • Intervention impact projections and recommendations\n\n"
            
            "📈 CLINICAL PERFORMANCE VISUALIZATIONS:\n"
            "- Model Performance Metrics:\n"
            "  • ROC curves with confidence bands and clinical thresholds\n"
            "  • Precision-Recall curves with baseline comparisons\n"
            "  • Calibration plots with perfect calibration reference lines\n"
            "  • Confusion matrices with clinical metric annotations\n"
            "  • Performance heatmaps across different patient subgroups\n"
            "- Clinical Decision Analysis:\n"
            "  • Decision curve analysis with net benefit visualization\n"
            "  • Risk threshold optimization curves\n"
            "  • Clinical utility comparison charts vs standard care\n"
            "  • Cost-effectiveness scatter plots\n"
            "  • Number-needed-to-treat visualizations\n"
            "- Temporal Analysis Charts:\n"
            "  • Model performance stability over time\n"
            "  • Concept drift detection visualizations\n"
            "  • Seasonal pattern analysis in risk predictions\n"
            "  • Performance degradation monitoring dashboards\n\n"
            
            "🧠 EXPLAINABILITY VISUALIZATIONS:\n"
            "- Global Feature Importance:\n"
            "  • Horizontal bar charts with clinical feature names\n"
            "  • Feature interaction network graphs\n"
            "  • Correlation heatmaps with clinical clustering\n"
            "  • Stability analysis charts across different datasets\n"
            "- Patient-Specific Explanations:\n"
            "  • SHAP waterfall plots with clinical annotations\n"
            "  • Feature contribution radar charts\n"
            "  • Patient vs population comparison charts\n"
            "  • Counterfactual analysis: 'what-if' scenario visualizations\n"
            "- Clinical Context Integration:\n"
            "  • Evidence-based guideline overlay charts\n"
            "  • Risk factor clinical threshold indicators\n"
            "  • Treatment pathway flow diagrams\n"
            "  • Intervention timeline and priority matrices\n\n"
            
            "⏱️ TEMPORAL DATA VISUALIZATIONS:\n"
            "- Patient Vital Signs Trends:\n"
            "  • Multi-parameter time series with normal range bands\n"
            "  • Interactive zoom and pan for detailed analysis\n"
            "  • Medication timing overlays and adherence indicators\n"
            "  • Clinical event markers (hospitalizations, procedures)\n"
            "- Risk Evolution Analysis:\n"
            "  • 90-day prediction window visualization with uncertainty\n"
            "  • Historical risk score progression\n"
            "  • Intervention impact timeline visualization\n"
            "  • Comparative risk trajectory analysis\n"
            "- Population Trend Analysis:\n"
            "  • Cohort risk distribution changes over time\n"
            "  • Seasonal healthcare pattern visualizations\n"
            "  • Care quality metric improvements\n"
            "  • Population health outcome tracking\n\n"
            
            "🎯 CLINICAL WORKFLOW OPTIMIZED DESIGN:\n"
            "- Healthcare UI/UX Best Practices:\n"
            "  • 10-second interpretation optimization for busy clinical settings\n"
            "  • Healthcare-standard color schemes (red=critical, yellow=caution, green=normal)\n"
            "  • High-contrast design for various lighting conditions\n"
            "  • Touch-friendly interface for tablet-based bedside use\n"
            "- Clinical Cognitive Load Reduction:\n"
            "  • Information hierarchy: Most critical information prominently displayed\n"
            "  • Progressive disclosure: Drill-down for detailed analysis\n"
            "  • Contextual tooltips with clinical explanations\n"
            "  • Alert fatigue prevention: Intelligent notification design\n"
            "- Multi-Device Responsiveness:\n"
            "  • Desktop workstation optimization for detailed analysis\n"
            "  • Tablet interface for bedside patient care\n"
            "  • Mobile phone compatibility for quick consultations\n"
            "  • Print-optimized formats for clinical documentation\n\n"
            
            "📊 ADVANCED VISUALIZATION TECHNIQUES:\n"
            "- Interactive Dashboard Elements:\n"
            "  • Real-time filtering and sorting capabilities\n"
            "  • Dynamic threshold adjustment with live updates\n"
            "  • Cross-filtering between different visualization components\n"
            "  • Drill-down analysis from population to individual patients\n"
            "- Statistical Visualization Enhancements:\n"
            "  • Confidence interval bands on time series\n"
            "  • Bootstrap distribution visualizations\n"
            "  • Bayesian posterior probability displays\n"
            "  • Monte Carlo simulation result presentations\n"
            "- Clinical Animation and Transitions:\n"
            "  • Smooth transitions between different time periods\n"
            "  • Animated risk score changes during what-if scenarios\n"
            "  • Progressive data loading for large patient cohorts\n"
            "  • Guided tour animations for new user onboarding\n\n"
            
            "🌈 ACCESSIBILITY & INCLUSIVITY DESIGN:\n"
            "- Visual Accessibility:\n"
            "  • Color-blind friendly palettes (tested with Coblis)\n"
            "  • High contrast mode for visual impairments\n"
            "  • Scalable text and UI elements\n"
            "  • Screen reader compatible annotations\n"
            "- Cultural and Linguistic Accessibility:\n"
            "  • Multi-language support for diverse patient populations\n"
            "  • Cultural sensitivity in visual metaphors and examples\n"
            "  • Health literacy appropriate explanatory text\n"
            "  • Gender-neutral and inclusive design patterns\n"
            "- Technical Accessibility:\n"
            "  • Keyboard navigation support\n"
            "  • Voice control compatibility\n"
            "  • Assistive technology integration\n"
            "  • Offline functionality for resource-constrained environments\n\n"
            
            "🖥️ DASHBOARD INTEGRATION & DEPLOYMENT:\n"
            "- EHR System Integration:\n"
            "  • FHIR-compliant data visualization components\n"
            "  • Epic MyChart and Cerner PowerChart integration patterns\n"
            "  • HL7 message format compatibility\n"
            "  • Single sign-on (SSO) authentication support\n"
            "- Clinical Workflow Integration:\n"
            "  • Embedded visualization widgets for clinical applications\n"
            "  • API endpoints for real-time dashboard updates\n"
            "  • Clinical alert system integration\n"
            "  • Quality reporting dashboard connections\n"
            "- Performance Optimization:\n"
            "  • Large dataset visualization optimization\n"
            "  • Lazy loading for improved response times\n"
            "  • Caching strategies for frequently accessed visualizations\n"
            "  • Progressive web app (PWA) functionality\n\n"
            
            "📋 CLINICAL REPORTING & DOCUMENTATION:\n"
            "- Automated Report Generation:\n"
            "  • One-page patient risk summary reports\n"
            "  • Population health management dashboards\n"
            "  • Quality improvement metric visualizations\n"
            "  • Regulatory compliance reporting charts\n"
            "- Customizable Clinical Templates:\n"
            "  • Physician consultation note templates\n"
            "  • Nursing care plan visualization templates\n"
            "  • Patient education handout designs\n"
            "  • Clinical research presentation templates\n"
            "- Export and Sharing Capabilities:\n"
            "  • High-resolution image exports (PNG, SVG, PDF)\n"
            "  • Interactive HTML exports for presentations\n"
            "  • Data table exports (CSV, Excel) for analysis\n"
            "  • Secure sharing links with access controls\n\n"
            
            "DELIVERABLES:\n"
            "1. Complete Streamlit dashboard with clinical workflow integration\n"
            "2. Interactive visualization component library\n"
            "3. Performance metrics visualization suite\n"
            "4. Patient-specific explanation visualizations\n"
            "5. Clinical reporting templates and automated generators\n"
            "6. Mobile-responsive design with accessibility compliance"
        ),
        expected_output=(
            "Comprehensive visualization package including:\n"
            "• Complete clinical dashboard with cohort overview and patient details\n"
            "• Interactive visualization component library for various clinical needs\n"
            "• Performance evaluation charts with clinical interpretation\n"
            "• Patient-specific explanation visualizations (SHAP, feature contributions)\n"
            "• Clinical workflow-optimized design with accessibility compliance\n"
            "• Export-ready clinical reports and documentation templates"
        ),
        agent=agent,
        output_file=f"outputs/reports/visualization_report_{timestamp}.json",
        tools=None,
        dependencies=[]
    )

# ---- Individual Task Creation Functions ----
def create_comprehensive_pipeline_task(agents: Dict) -> Task:
    """Create a single comprehensive task that orchestrates the entire pipeline"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return Task(
        description=(
            "🏥 EXECUTE COMPLETE AI-DRIVEN CHRONIC CARE RISK PREDICTION PIPELINE\n\n"
            "Orchestrate the full end-to-end pipeline including data preprocessing, "
            "model training, evaluation, explanation, clinical validation, and visualization. "
            "Ensure each step meets clinical standards and integrates seamlessly for "
            "deployment in healthcare environments.\n\n"
            "PIPELINE STEPS:\n"
            "1. Clinical data preprocessing and feature engineering\n"
            "2. Multi-algorithm model training and optimization\n"
            "3. Comprehensive performance evaluation\n"
            "4. AI explainability and clinical interpretation\n"
            "5. Clinical guideline validation and safety assessment\n"
            "6. Dashboard and visualization development\n\n"
            "SUCCESS CRITERIA:\n"
            "- Model performance: AUROC ≥0.75, AUPRC ≥0.65, Sensitivity ≥80%\n"
            "- Clinical validation: Guideline compliance and safety verification\n"
            "- Deployment readiness: Dashboard and documentation complete"
        ),
        expected_output=(
            "Complete pipeline execution results with:\n"
            "• Trained and validated risk prediction model\n"
            "• Comprehensive evaluation and clinical validation reports\n"
            "• AI explanations and clinical interpretations\n"
            "• Functional clinical dashboard\n"
            "• Deployment documentation and recommendations"
        ),
        agent=agents.get('risk_assessor'),  # Primary orchestrating agent
        output_file=f"outputs/reports/complete_pipeline_report_{timestamp}.json",
        dependencies=[]
    )

if __name__ == "__main__":
    # Test task creation
    print("🧪 Testing task creation functions...")
    
    # Mock agents for testing
    mock_agents = {
        'data_processor': 'mock_data_processor',
        'risk_assessor': 'mock_risk_assessor', 
        'evaluator': 'mock_evaluator',
        'explainer': 'mock_explainer',
        'clinical_validator': 'mock_clinical_validator',
        'visualizer': 'mock_visualizer'
    }
    
    # Create tasks
    tasks = create_prediction_tasks(mock_agents)
    
    print(f"✅ Created {len(tasks)} tasks successfully:")
    for task_name, task in tasks.items():
        print(f"  - {task_name}: {task.description[:50]}...")
    
    print("✅ Task creation functions working correctly")
