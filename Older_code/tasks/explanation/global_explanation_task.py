"""
Global Explanation Task for population-level model interpretability.
Generates comprehensive insights about model behavior across all patients.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from crewai import Task

logger = logging.getLogger(__name__)

class GlobalExplanationTask:
    """Task for generating global model explanations and insights."""
    
    def __init__(self, config: Any, agents: Dict[str, Any]):
        """Initialize the global explanation task."""
        self.config = config
        self.agents = agents
        
        # Create the CrewAI task
        self.task = Task(
            description=self._get_task_description(),
            expected_output=self._get_expected_output(),
            agent=self.agents['explainability'].agent,
            tools=self.agents['explainability'].agent.tools
        )
        
        logger.info("GlobalExplanationTask initialized")
    
    def _get_task_description(self) -> str:
        """Get comprehensive task description."""
        return f"""
        Generate comprehensive global explanations for the chronic care risk prediction model 
        to provide population-level insights and clinical understanding of model behavior.
        
        **PRIMARY OBJECTIVE:**
        Create clinically meaningful, population-level explanations that help healthcare 
        providers understand how the AI model makes risk predictions across different 
        patient populations and clinical scenarios.
        
        **GLOBAL EXPLANATION FRAMEWORK:**
        
        1. **FEATURE IMPORTANCE ANALYSIS:**
           - Population-level feature importance rankings
           - Clinical domain-specific importance patterns
           - Temporal feature contribution analysis
           - Condition-specific feature relevance
           - Demographic variation in feature importance
        
        2. **MODEL BEHAVIOR CHARACTERIZATION:**
           - Risk prediction patterns across patient populations
           - Clinical threshold identification and validation
           - Decision boundary analysis for different conditions
           - Model confidence regions and uncertainty quantification
           - Prediction stability across similar patient profiles
        
        3. **CLINICAL PATHWAY ANALYSIS:**
           - Risk escalation pathways and trigger points
           - Protective factor identification and quantification
           - Intervention opportunity mapping
           - Care coordination decision support insights
           - Population health management guidance
        
        4. **SHAP-BASED GLOBAL INSIGHTS:**
           - Population SHAP value distributions
           - Feature interaction effects at scale
           - Conditional feature importance analysis
           - Risk factor synergy and antagonism patterns
           - Clinical decision rule extraction
        
        **EXPLANATION METHODOLOGIES:**
        
        **Feature Importance Techniques:**
        - TreeSHAP for tree-based model explanations
        - Permutation importance for robustness validation
        - Partial dependence analysis for feature effects
        - Feature interaction strength measurement
        - Accumulated local effects (ALE) plots
        
        **Clinical Contextualization:**
        - Medical terminology translation and mapping
        - Clinical significance thresholds identification
        - Evidence-based medicine correlation
        - Guideline compliance verification
        - Risk communication framework development
        
        **Population Segmentation Analysis:**
        - Age-stratified explanation patterns
        - Gender-specific risk factor profiles
        - Condition-specific model behavior
        - Comorbidity interaction effects
        - Socioeconomic factor impact assessment
        
        **CLINICAL INTERPRETATION REQUIREMENTS:**
        
        **Medical Domain Translation:**
        - Convert statistical importance to clinical significance
        - Align predictions with clinical decision frameworks
        - Provide evidence-based rationale for each insight
        - Map findings to established risk calculators
        - Correlate with clinical practice guidelines
        
        **Healthcare Provider Communication:**
        - Executive summary for clinical leadership
        - Detailed analysis for medical specialists
        - Quick reference guides for frontline providers
        - Training materials for clinical implementation
        - Quality improvement insights for care teams
        
        **Risk Stratification Insights:**
        - High-risk population characteristics
        - Protective factor identification
        - Modifiable risk factor prioritization
        - Intervention timing optimization
        - Resource allocation guidance
        
        **CLINICAL VALIDATION INTEGRATION:**
        
        **Guideline Alignment:**
        - ADA diabetes management correlation
        - AHA/ACC heart failure care alignment
        - Obesity medicine protocol consistency
        - Preventive care guideline integration
        - Quality measure correlation analysis
        
        **Evidence-Based Validation:**
        - Literature support for identified patterns
        - Clinical trial correlation analysis
        - Real-world evidence alignment
        - Expert consensus verification
        - Systematic review consistency check
        
        **ACTIONABLE INSIGHT GENERATION:**
        
        **Clinical Decision Support:**
        - Risk threshold recommendations
        - Intervention triggering criteria
        - Monitoring frequency guidance
        - Specialist referral indicators
        - Care coordination protocols
        
        **Population Health Management:**
        - High-risk cohort identification strategies
        - Preventive intervention targeting
        - Resource optimization recommendations
        - Quality improvement opportunities
        - Outcome prediction benchmarking
        
        Collaborate with clinical validator and risk assessor agents to ensure 
        all global explanations maintain medical accuracy and provide actionable 
        insights for healthcare delivery improvement.
        """
    
    def _get_expected_output(self) -> str:
        """Get expected output specification."""
        return f"""
        COMPREHENSIVE GLOBAL MODEL EXPLANATION REPORT:
        
        **EXECUTIVE SUMMARY:**
        ```
        ═══════════════════════════════════════════════════════════════════
        CHRONIC CARE RISK PREDICTION MODEL - GLOBAL EXPLANATION ANALYSIS
        ═══════════════════════════════════════════════════════════════════
        
        Model Interpretability: ✅ EXCELLENT (Clinically Transparent)
        Feature Importance: 🎯 20 primary predictors identified
        Clinical Alignment: ✅ 96.8% consistent with medical guidelines
        Population Insights: 📊 5 distinct risk profiles characterized
        Actionable Recommendations: 🎯 12 clinical decision rules extracted
        Provider Readiness: ✅ READY for clinical explanation deployment
        
        KEY INSIGHT: HbA1c trends and medication adherence account for 
        67% of prediction variance in diabetic population risk assessment.
        ═══════════════════════════════════════════════════════════════════
        ```
        
        **GLOBAL FEATURE IMPORTANCE ANALYSIS:**
        
        **Top 20 Risk Predictors (Population-Level):**
        | Rank | Feature | Importance | Clinical Domain | Medical Significance |
        |------|---------|------------|-----------------|---------------------|
        | 1 | `hba1c_trend_90d` | 12.4% | Diabetes Control | HbA1c progression pattern over 90 days |
        | 2 | `medication_adherence_composite` | 9.8% | Treatment Compliance | Multi-medication adherence score |
        | 3 | `hospitalization_recency_days` | 8.7% | Healthcare Utilization | Days since last inpatient admission |
        | 4 | `comorbidity_burden_weighted` | 7.4% | Disease Complexity | Charlson-weighted comorbidity index |
        | 5 | `systolic_bp_variability_30d` | 6.9% | Cardiovascular Control | Blood pressure stability indicator |
        | 6 | `age_risk_category` | 6.3% | Demographics | Age-stratified risk classification |
        | 7 | `egfr_decline_slope_180d` | 5.9% | Kidney Function | Renal function deterioration rate |
        | 8 | `emergency_visits_6m` | 5.5% | Acute Care Usage | Emergency department utilization |
        | 9 | `medication_count_active` | 5.1% | Polypharmacy | Total active prescription medications |
        | 10 | `diabetes_duration_years` | 4.8% | Disease History | Years since diabetes diagnosis |
        | 11 | `bmi_trajectory_12m` | 4.5% | Weight Management | Body mass index change pattern |
        | 12 | `social_determinant_risk` | 4.2% | Social Factors | Housing, transport, food security |
        | 13 | `lab_abnormality_frequency` | 3.9% | Laboratory Monitoring | Percentage of abnormal lab results |
        | 14 | `specialist_visit_frequency` | 3.6% | Care Coordination | Specialty care engagement level |
        | 15 | `symptom_burden_score` | 3.3% | Clinical Symptoms | Patient-reported symptom severity |
        | 16 | `insurance_stability_indicator` | 3.0% | Access to Care | Healthcare coverage consistency |
        | 17 | `medication_side_effect_count` | 2.7% | Treatment Tolerance | Documented adverse drug reactions |
        | 18 | `family_history_risk_score` | 2.4% | Genetic Factors | Hereditary disease risk assessment |
        | 19 | `lifestyle_risk_composite` | 2.1% | Behavioral Factors | Smoking, diet, exercise patterns |
        | 20 | `care_gap_indicator` | 1.8% | Preventive Care | Missed screening/monitoring services |
        
        **Clinical Domain Importance Distribution:**
        - 🩺 **Diabetes Management**: 34.2% (Primary driver of risk)
        - 💊 **Medication Management**: 22.6% (Critical adherence factor)
        - 🏥 **Healthcare Utilization**: 18.3% (Service engagement indicator)
        - ❤️ **Cardiovascular Health**: 12.4% (Comorbidity risk factor)
        - 🧬 **Patient Demographics**: 8.1% (Baseline risk modifiers)
        - 🏠 **Social Determinants**: 4.4% (Access and support factors)
        
        **POPULATION RISK SEGMENTATION:**
        
        **Identified Risk Profiles:**
        
        **Profile 1: High-Risk Diabetic (23% of population)**
        ```
        Characteristics:
        -  HbA1c >8.5% with worsening trend
        -  Poor medication adherence (<70%)
        -  Multiple recent hospitalizations
        -  Advanced age (>70 years)
        -  High comorbidity burden (3+ conditions)
        
        Key Predictors:
        1. HbA1c trend deterioration (Weight: 18.2%)
        2. Medication non-adherence (Weight: 15.7%)
        3. Frequent emergency visits (Weight: 12.3%)
        
        Clinical Recommendations:
        → Intensive diabetes management program
        → Medication adherence interventions
        → Enhanced monitoring (weekly contact)
        → Endocrinology referral within 30 days
        
        Risk Reduction Potential: 34% with targeted interventions
        ```
        
        **Profile 2: Heart Failure Progression (18% of population)**
        ```
        Characteristics:
        -  NYHA Class III-IV symptoms
        -  Declining kidney function (eGFR <45)
        -  Fluid retention indicators
        -  Recent cardiac hospitalizations
        -  Medication optimization needs
        
        Key Predictors:
        1. eGFR decline slope (Weight: 16.8%)
        2. BNP elevation pattern (Weight: 14.1%)
        3. Hospital readmission cycle (Weight: 11.9%)
        
        Clinical Recommendations:
        → Heart failure specialist consultation
        → Diuretic optimization protocols
        → Remote monitoring activation
        → Care transition programs
        
        Risk Reduction Potential: 28% with guideline-directed therapy
        ```
        
        **Profile 3: Complex Multi-Morbidity (22% of population)**
        ```
        Characteristics:
        -  4+ chronic conditions
        -  Polypharmacy (8+ medications)
        -  Multiple specialist providers
        -  Frequent care coordination needs
        -  Social complexity factors
        
        Key Predictors:
        1. Comorbidity burden score (Weight: 19.4%)
        2. Medication complexity (Weight: 13.2%)
        3. Care fragmentation index (Weight: 10.7%)
        
        Clinical Recommendations:
        → Comprehensive care management
        → Medication reconciliation services
        → Care coordinator assignment
        → Integrated care planning
        
        Risk Reduction Potential: 22% with coordinated care
        ```
        
        **Profile 4: Obesity-Related Metabolic (19% of population)**
        ```
        Characteristics:
        -  BMI >35 with metabolic syndrome
        -  Insulin resistance patterns
        -  Sleep apnea comorbidity
        -  Weight management challenges
        -  Lifestyle modification needs
        
        Key Predictors:
        1. BMI trajectory worsening (Weight: 15.6%)
        2. Metabolic panel abnormalities (Weight: 12.8%)
        3. Lifestyle adherence gaps (Weight: 9.4%)
        
        Clinical Recommendations:
        → Obesity medicine consultation
        → Structured lifestyle programs
        → Bariatric surgery evaluation
        → Metabolic monitoring protocols
        
        Risk Reduction Potential: 31% with comprehensive weight management
        ```
        
        **Profile 5: Social Determinant Risk (18% of population)**
        ```
        Characteristics:
        -  Transportation barriers
        -  Food insecurity issues
        -  Insurance instability
        -  Limited health literacy
        -  Social support deficits
        
        Key Predictors:
        1. Social risk composite (Weight: 21.3%)
        2. Care access barriers (Weight: 14.7%)
        3. Health literacy indicators (Weight: 8.9%)
        
        Clinical Recommendations:
        → Social work consultation
        → Community resource linkage
        → Patient navigation services
        → Health literacy interventions
        
        Risk Reduction Potential: 26% with social support interventions
        ```
        
        **FEATURE INTERACTION ANALYSIS:**
        
        **Critical Interaction Effects:**
        
        **1. Age × Comorbidity Burden Interaction:**
        - **Effect Magnitude**: 2.3x risk amplification
        - **Clinical Insight**: Elderly patients with multiple conditions show exponential risk increase
        - **Threshold**: Risk doubles when age >75 AND comorbidities ≥3
        - **Intervention**: Prioritize geriatric assessment for complex elderly patients
        
        **2. Medication Adherence × Disease Severity Interaction:**
        - **Effect Magnitude**: 1.8x risk amplification  
        - **Clinical Insight**: Poor adherence has greater impact in severe disease states
        - **Threshold**: Risk triples when adherence <60% AND HbA1c >9%
        - **Intervention**: Intensive adherence support for severe cases
        
        **3. Social Determinants × Clinical Complexity Interaction:**
        - **Effect Magnitude**: 1.6x risk amplification
        - **Clinical Insight**: Social barriers compound medical complexity effects
        - **Threshold**: Risk increases 60% when social risk high AND care fragmented
        - **Intervention**: Integrated social-medical care models
        
        **CLINICAL DECISION RULES EXTRACTED:**
        
        **Rule 1: High-Risk Diabetes Alert**
        ```
        IF (HbA1c_trend_90d > 0.5 AND medication_adherence < 0.7) 
        OR (recent_hospitalization = TRUE AND HbA1c > 8.5)
        THEN Risk_Level = "Critical" 
        AND Recommendation = "Immediate endocrinology referral + intensive management"
        
        Accuracy: 89.3% | Precision: 84.7% | Clinical Validation: ✅ Approved
        ```
        
        **Rule 2: Heart Failure Decompensation Warning**
        ```
        IF (eGFR_decline > 5 mL/min/year AND BNP_trend = "increasing") 
        OR (emergency_visits_30d ≥ 2 AND fluid_retention_signs = TRUE)
        THEN Risk_Level = "High"
        AND Recommendation = "Cardiology consultation + medication optimization"
        
        Accuracy: 86.1% | Precision: 81.9% | Clinical Validation: ✅ Approved
        ```
        
        **Rule 3: Multi-Morbidity Care Coordination Trigger**
        ```
        IF (active_conditions ≥ 4 AND active_medications ≥ 8)
        AND (specialist_providers ≥ 3 OR care_gaps_identified ≥ 2)
        THEN Risk_Level = "Moderate-High"
        AND Recommendation = "Care management enrollment + medication review"
        
        Accuracy: 83.5% | Precision: 78.2% | Clinical Validation: ✅ Approved
        ```
        
        **PARTIAL DEPENDENCE ANALYSIS:**
        
        **Key Feature Dependencies:**
        
        **HbA1c Impact on Risk Probability:**
        - HbA1c 6.0-7.0%: Baseline risk (1.0x multiplier)
        - HbA1c 7.1-8.0%: Moderate increase (1.3x multiplier)  
        - HbA1c 8.1-9.0%: High risk (2.1x multiplier)
        - HbA1c >9.0%: Critical risk (3.4x multiplier)
        - **Clinical Threshold**: Risk doubles at HbA1c 8.1%
        
        **Medication Adherence Impact:**
        - Adherence >90%: Protective effect (-0.15 risk units)
        - Adherence 70-90%: Neutral effect (0.0 risk units)
        - Adherence 50-70%: Moderate risk (+0.25 risk units)
        - Adherence <50%: High risk (+0.58 risk units)
        - **Clinical Threshold**: Risk increases significantly below 70% adherence
        
        **Age Risk Stratification:**
        - Age 18-40: Low baseline risk
        - Age 40-65: Gradual risk increase (linear)
        - Age 65-75: Moderate risk acceleration  
        - Age >75: Steep risk increase (exponential)
        - **Clinical Threshold**: Risk acceleration begins at age 65
        
        **CLINICAL PATHWAY RECOMMENDATIONS:**
        
        **Primary Prevention Pathways:**
        1. **Early Diabetes Intervention**: Target HbA1c 6.5-7.5% maintenance
        2. **Medication Adherence Optimization**: Achieve >80% adherence rates
        3. **Lifestyle Modification Programs**: Focus on modifiable risk factors
        4. **Preventive Care Completion**: Close identified care gaps
        
        **Secondary Prevention Pathways:**
        1. **Intensive Disease Management**: Specialist care coordination
        2. **Complication Prevention**: Enhanced monitoring protocols
        3. **Care Transition Support**: Hospital-to-home programs
        4. **Social Support Integration**: Address determinant barriers
        
        **Tertiary Prevention Pathways:**
        1. **Advanced Care Planning**: Complex case management
        2. **Palliative Care Integration**: Quality of life focus
        3. **Family/Caregiver Support**: Care network strengthening
        4. **Emergency Preparedness**: Crisis intervention protocols
        
        **POPULATION HEALTH INSIGHTS:**
        
        **Risk Distribution Analysis:**
        - **Low Risk (<30%)**: 62.3% of population - Standard care protocols
        - **Moderate Risk (30-70%)**: 23.1% of population - Enhanced monitoring  
        - **High Risk (>70%)**: 14.6% of population - Intensive intervention
        
        **Intervention Impact Modeling:**
        - **Targeted HbA1c Programs**: 18% population risk reduction potential
        - **Adherence Improvement Initiatives**: 12% risk reduction potential
        - **Care Coordination Enhancement**: 8% risk reduction potential
        - **Social Determinant Interventions**: 6% risk reduction potential
        
        **Resource Allocation Guidance:**
        - **High-Intensity Programs**: Focus on 14.6% high-risk population
        - **Care Management**: Target 23.1% moderate-risk population
        - **Prevention Programs**: Maintain 62.3% low-risk population
        - **ROI Optimization**: Prioritize adherence and diabetes control initiatives
        
        **CLINICAL IMPLEMENTATION GUIDELINES:**
        
        **Provider Training Requirements:**
        - Understanding of feature importance rankings
        - Interpretation of risk score components  
        - Clinical decision rule application
        - Population segmentation recognition
        - Intervention prioritization strategies
        
        **Quality Improvement Integration:**
        - Performance metric alignment with model insights
        - Care protocol optimization based on risk profiles
        - Resource allocation guided by population analysis
        - Outcome measurement framework development
        
        **Continuous Learning Framework:**
        - Monthly model performance review
        - Quarterly feature importance updates  
        - Semi-annual population segmentation analysis
        - Annual clinical guideline alignment verification
        
        **FILES GENERATED:**
        ```
        📁 global_explanation_outputs/
        ├── 📊 global_model_explanation_report.pdf           # Complete 54-page analysis
        ├── 🎯 feature_importance_clinical_guide.pdf        # Provider reference document  
        ├── 👥 population_risk_segmentation_analysis.xlsx   # Detailed population profiles
        ├── 🔄 feature_interaction_visualization.html       # Interactive exploration tool
        ├── 📈 partial_dependence_plots.png                 # Feature effect visualizations
        ├── ⚖️ clinical_decision_rules_extracted.json      # Implementable decision logic
        ├── 🏥 population_health_insights_dashboard.html   # Population management tool
        ├── 📋 clinical_pathway_recommendations.pdf        # Care protocol guidance
        ├── 🎓 provider_training_materials.pptx           # Educational content
        ├── 📊 quality_improvement_metrics.csv            # QI measurement framework
        ├── 🔍 model_interpretability_summary.md          # Technical documentation
        └── 🎯 implementation_readiness_checklist.pdf     # Deployment guide
        ```
        
        **CLINICAL VALIDATION SUMMARY:**
        ✅ All explanations reviewed by medical specialists
        ✅ Feature importance clinically validated  
        ✅ Decision rules tested against clinical scenarios
        ✅ Population insights verified with epidemiological data
        ✅ Implementation guidelines approved for clinical use
        ✅ Provider training materials clinically reviewed
        
        **CONCLUSION:**
        The global explanation analysis provides comprehensive, clinically meaningful insights 
        into model behavior that directly support healthcare delivery improvement. The identified 
        risk profiles, feature interactions, and clinical decision rules offer actionable guidance 
        for population health management and individual patient care optimization.
        """


class LocalExplanationTask:
    """Task for generating individual patient-level explanations."""
    
    def __init__(self, config: Any, agents: Dict[str, Any]):
        """Initialize the local explanation task."""
        self.config = config
        self.agents = agents
        
        # Create the CrewAI task
        self.task = Task(
            description=self._get_task_description(),
            expected_output=self._get_expected_output(),
            agent=self.agents['explainability'].agent,
            tools=self.agents['explainability'].agent.tools
        )
        
        logger.info("LocalExplanationTask initialized")
    
    def _get_task_description(self) -> str:
        """Get comprehensive task description."""
        return f"""
        Generate detailed, patient-specific explanations for individual risk predictions 
        to support clinical decision-making and patient communication.
        
        **PRIMARY OBJECTIVE:**
        Create personalized, clinically relevant explanations that help healthcare providers 
        understand why a specific patient received a particular risk score and what actions 
        can be taken to improve outcomes.
        
        **LOCAL EXPLANATION FRAMEWORK:**
        
        1. **INDIVIDUAL PATIENT RISK ANALYSIS:**
           - Patient-specific SHAP value computation
           - Contributing factor identification and quantification  
           - Risk factor ranking for individual case
           - Confidence interval for prediction uncertainty
           - Comparative analysis against population norms
        
        2. **CLINICAL CONTEXTUALIZATION:**
           - Medical history integration and timeline analysis
           - Current clinical status assessment
           - Risk factor progression tracking
           - Intervention opportunity identification
           - Care plan optimization suggestions
        
        3. **ACTIONABLE RECOMMENDATION GENERATION:**
           - Modifiable risk factor prioritization
           - Specific intervention recommendations
           - Monitoring frequency adjustments
           - Specialist referral indicators
           - Patient education focus areas
        
        4. **COUNTERFACTUAL SCENARIO ANALYSIS:**
           - "What-if" intervention impact modeling
           - Risk reduction potential calculations
           - Alternative care pathway exploration  
           - Optimal outcome scenario planning
           - Resource requirement estimation
        
        **EXPLANATION METHODOLOGIES:**
        
        **SHAP-Based Individual Analysis:**
        - TreeExplainer for precise feature attributions
        - Waterfall plots for additive risk factor visualization
        - Force plots for positive/negative contribution display
         - Individual conditional expectation (ICE) curves
        - Local feature interaction identification
        
        **Clinical Risk Communication:**
        - Risk level categorization with clinical significance
        - Timeline-based factor contribution analysis
        - Comparative risk assessment (patient vs population)
        - Intervention impact quantification
        - Outcome probability scenarios
        
        **Patient-Centered Communication:**
        - Plain language risk explanation
        - Visual risk communication tools
        - Personalized health education content
        - Behavior change motivation insights
        - Family/caregiver communication guidance
        
        **CLINICAL DECISION SUPPORT INTEGRATION:**
        
        **Provider-Facing Explanations:**
        - Clinical summary with key risk drivers
        - Evidence-based intervention recommendations
        - Monitoring protocol adjustments
        - Care coordination requirements
        - Quality metric impact assessment
        
        **Patient-Facing Explanations:**
        - Understandable risk level communication
        - Personal health factor identification
        - Actionable lifestyle recommendations
        - Medication adherence importance
        - Healthcare engagement guidance
        
        **Care Team Communication:**
        - Specialist consultation indicators
        - Care coordination priorities
        - Family involvement recommendations
        - Resource allocation guidance
        - Quality improvement opportunities
        
        **TEMPORAL ANALYSIS INTEGRATION:**
        
        **Risk Trajectory Modeling:**
        - Historical risk factor evolution
        - Current trajectory projection
        - Intervention impact timeline
        - Monitoring milestone identification
        - Outcome probability changes over time
        
        **Intervention Timing Optimization:**
        - Critical intervention windows
        - Sequential intervention planning
        - Resource allocation timing
        - Monitoring frequency optimization
        - Outcome measurement scheduling
        
        **CLINICAL VALIDATION REQUIREMENTS:**
        
        **Medical Accuracy Verification:**
        - Clinical plausibility of factor combinations
        - Guideline compliance verification
        - Evidence-based recommendation validation
        - Contraindication screening
        - Drug interaction assessment
        
        **Patient Safety Considerations:**
        - High-risk factor identification
        - Emergency intervention triggers
        - Safety monitoring requirements
        - Adverse event prevention protocols
        - Clinical escalation pathways
        
        **PERSONALIZATION FACTORS:**
        
        **Individual Clinical Context:**
        - Medical history complexity
        - Current medication regimen
        - Comorbidity interactions
        - Social determinant impacts
        - Healthcare access patterns
        
        **Patient Preference Integration:**
        - Treatment goal alignment
        - Lifestyle modification capacity
        - Healthcare engagement preferences
        - Family support availability
        - Cultural consideration factors
        
        **QUALITY ASSURANCE PROTOCOLS:**
        
        **Explanation Accuracy:**
        - SHAP value verification and validation
        - Clinical logic consistency checking
        - Recommendation evidence base verification
        - Outcome prediction accuracy assessment
        - Provider feedback integration
        
        **Communication Effectiveness:**
        - Clinical clarity and actionability
        - Patient comprehension optimization
        - Provider workflow integration
        - Decision support utility
        - Continuous improvement feedback
        
        Generate explanations that seamlessly integrate into clinical workflows 
        while providing clear, actionable insights for both providers and patients.
        Ensure all recommendations align with evidence-based medicine and clinical guidelines.
        """
    
    def _get_expected_output(self) -> str:
        """Get expected output specification."""
        return f"""
        INDIVIDUAL PATIENT RISK EXPLANATION REPORT:
        
        **PATIENT RISK SUMMARY:**
        ```
        ════════════════════════════════════════════════════════════════════
        PATIENT RISK PREDICTION EXPLANATION
        ════════════════════════════════════════════════════════════════════
        
        Patient ID: SYNTH_004257
        Prediction Date: {datetime.now().strftime('%B %d, %Y')}
        
        🎯 RISK ASSESSMENT:
        Overall Risk Score: 73.2% (HIGH RISK)
        Risk Category: High - Requires immediate clinical attention
        Confidence Level: 89.4% (Strong prediction confidence)
        Time Horizon: 90-day deterioration risk
        
        🚨 CLINICAL PRIORITY: Urgent intervention recommended within 48 hours
        ════════════════════════════════════════════════════════════════════
        ```
        
        **TOP RISK FACTORS CONTRIBUTING TO THIS PATIENT:**
        
        **Primary Risk Contributors (Positive Impact):**
        | Factor | Contribution | Clinical Value | Normal Range | Patient Status |
        |---------|-------------|----------------|--------------|----------------|
        | **HbA1c Worsening Trend** | +18.7% | 9.8% → 10.2% | <7.0% | ⚠️ **Critical - Rapid deterioration** |
        | **Poor Medication Adherence** | +15.2% | 42% adherent | >80% | 🔴 **Poor - Major concern** |
        | **Recent Hospitalization** | +12.4% | 8 days ago | None recent | 🚨 **Recent - High impact** |
        | **Kidney Function Decline** | +9.8% | eGFR: 38 mL/min | >60 mL/min | 🟡 **Moderate decline** |
        | **Multiple Comorbidities** | +7.3% | 5 conditions | 0-2 typical | 🟠 **Complex case** |
        | **Advanced Age** | +6.1% | 78 years | N/A | ℹ️ **Age-related risk** |
        
        **Protective Factors (Negative Impact):**
        | Factor | Contribution | Clinical Value | Protective Threshold | Patient Status |
        |---------|-------------|----------------|---------------------|----------------|
        | **Regular Specialist Care** | -3.2% | Monthly visits | Regular engagement | ✅ **Good engagement** |
        | **Family Support System** | -2.1% | Strong support | Available support | ✅ **Adequate support** |
        | **Stable Housing** | -1.8% | Stable housing | Housing security | ✅ **Stable situation** |
        
        **CLINICAL INTERPRETATION:**
        
        **Risk Driver Analysis:**
        ```
        🔍 PRIMARY CONCERN: Diabetes Control Crisis
        
        This patient's high risk is driven by a perfect storm of:
        1. Rapidly worsening diabetes control (HbA1c 9.8% → 10.2%)
        2. Critical medication non-adherence (42% vs target 80%+)  
        3. Recent hospitalization indicating system stress
        4. Declining kidney function complicating diabetes management
        5. Age-related complexity with multiple comorbidities
        
        ⚡ CLINICAL INSIGHT: The combination of poor glycemic control 
        and medication non-adherence creates a 2.3x amplified risk 
        beyond individual factor effects.
        ```
        
        **Individual Risk Trajectory:**
        ```
        📈 RISK PROGRESSION OVER TIME:
        
        6 months ago: 28% risk (Moderate)
        ├── HbA1c: 8.1% (elevated but stable)
        ├── Medication adherence: 75% (suboptimal)  
        └── No recent hospitalizations
        
        3 months ago: 45% risk (Moderate-High)
        ├── HbA1c: 8.9% (worsening trend)
        ├── Medication adherence: 58% (declining)
        └── Emergency visit for hyperglycemia
        
        Current: 73.2% risk (High)
        ├── HbA1c: 10.2% (critical deterioration)
        ├── Medication adherence: 42% (poor)
        └── Recent hospitalization (DKA episode)
        
        🎯 TRAJECTORY: Rapid escalation over 6 months
        Key Inflection Point: Medication adherence decline 3 months ago
        ```
        
        **COUNTERFACTUAL ANALYSIS - "WHAT IF" SCENARIOS:**
        
        **Scenario 1: Optimal Medication Adherence**
        ```
        IF medication adherence improved to 85%:
        Current Risk: 73.2% → Projected Risk: 52.1% (-21.1%)
        Risk Category: HIGH → MODERATE
        Clinical Impact: Significant risk reduction
        Implementation: Adherence counseling + monitoring
        Timeline: 30-60 days for full effect
        ```
        
        **Scenario 2: Intensive Diabetes Management**
        ```
        IF HbA1c improved to 8.0% through intensive management:
        Current Risk: 73.2% → Projected Risk: 41.8% (-31.4%)
        Risk Category: HIGH → MODERATE
        Clinical Impact: Major risk reduction
        Implementation: Endocrinology referral + insulin optimization
        Timeline: 60-90 days for achievement
        ```
        
        **Scenario 3: Combined Optimal Intervention**
        ```
        IF medication adherence (85%) + HbA1c control (8.0%):
        Current Risk: 73.2% → Projected Risk: 28.7% (-44.5%)
        Risk Category: HIGH → LOW-MODERATE  
        Clinical Impact: Transformative risk reduction
        Implementation: Comprehensive diabetes program
        Timeline: 90 days for combined effect
        ```
        
        **IMMEDIATE CLINICAL RECOMMENDATIONS:**
        
        **🚨 URGENT ACTIONS (Next 24-48 hours):**
        1. **Endocrinology Consultation** - Same-day referral for DKA management
        2. **Medication Reconciliation** - Complete review with clinical pharmacist
        3. **Adherence Assessment** - Identify and address specific barriers
        4. **Laboratory Monitoring** - Daily glucose, ketones until stable
        5. **Care Coordination** - Activate case management services
        
        **📋 SHORT-TERM INTERVENTIONS (1-4 weeks):**
        1. **Insulin Optimization** - Adjust regimen based on specialist recommendations
        2. **Adherence Support** - Implement pillbox, reminders, family involvement
        3. **Diabetes Education** - Reinforce self-management skills
        4. **Social Support** - Address transportation, financial barriers
        5. **Frequent Monitoring** - Weekly clinical contact until stabilized
        
        **🎯 LONG-TERM MANAGEMENT (1-3 months):**
        1. **HbA1c Target** - Achieve <8.0% within 90 days
        2. **Adherence Maintenance** - Sustain >80% medication compliance
        3. **Complication Prevention** - Nephrology referral for CKD management
        4. **Quality of Life** - Balance intensive management with patient goals
        5. **Prevention Focus** - Prevent future hospitalizations
        
        **MONITORING PROTOCOL:**
        
        **Weekly Monitoring (Next 4 weeks):**
        - Glucose log review and insulin adjustment
        - Medication adherence assessment
        - Symptom check (DKA warning signs)
        - Weight and vital signs
        - Psychosocial support needs
        
        **Monthly Monitoring (Months 2-6):**
        - HbA1c trending toward target
        - Comprehensive medication review
        - Diabetic complication screening
        - Care plan effectiveness assessment
        - Risk score re-evaluation
        
        **PATIENT COMMUNICATION GUIDE:**
        
        **For Healthcare Providers:**
        ```
        "Mr. Johnson, your diabetes monitoring shows that you're at higher 
        risk for complications over the next 90 days. The main concerns are 
        your blood sugar levels, which have gotten harder to control, and 
        challenges with taking medications consistently. 
        
        The good news is that we can significantly reduce this risk by 
        working together on better diabetes management. We'd like to connect 
        you with our diabetes specialist and pharmacist to optimize your 
        treatment plan and address any barriers to taking medications.
        
        With focused attention over the next few months, we can get your 
        diabetes back on track and reduce your risk substantially."
        ```
        
        **For Patient/Family:**
        ```
        🏠 PATIENT-FRIENDLY EXPLANATION:
        
        "Your Health Risk Assessment Results"
        
        What This Means:
        -  Your diabetes needs more attention right now
        -  Your blood sugar levels have been getting harder to control  
        -  Missing medications is making things worse
        -  We can help get things back on track
        
        What We'll Do Together:
        -  Meet with a diabetes specialist this week
        -  Review all your medications with a pharmacist
        -  Set up easier ways to remember medications
        -  Check in with you more often until you're stable
        
        Your Role:
        -  Take medications as prescribed every day
        -  Check blood sugar as recommended
        -  Come to all appointments
        -  Call if you feel unwell or have questions
        
        The Goal:
        -  Get blood sugar levels back in good range
        -  Prevent hospital visits
        -  Keep you healthy at home with family
        ```
        
        **SPECIALIST REFERRAL RECOMMENDATIONS:**
        
        **Urgent Endocrinology Referral:**
        - **Indication**: Recent DKA, HbA1c >10%, rapid deterioration
        - **Timeline**: Within 24-48 hours
        - **Focus**: Insulin regimen optimization, acute management
        - **Expected Outcome**: Stabilization within 2-4 weeks
        
        **Clinical Pharmacist Consultation:**
        - **Indication**: Poor adherence, complex regimen
        - **Timeline**: Within 1 week  
        - **Focus**: Adherence barriers, regimen simplification
        - **Expected Outcome**: Adherence improvement to >80%
        
        **Nephrology Evaluation:**
        - **Indication**: eGFR 38 mL/min, diabetic nephropathy
        - **Timeline**: Within 2-4 weeks
        - **Focus**: CKD management, medication adjustments
        - **Expected Outcome**: Preservation of kidney function
        
        **CARE COORDINATION REQUIREMENTS:**
        
        **Primary Care Team:**
        - Weekly contact for 4 weeks
        - Medication adherence monitoring
        - Glucose log review
        - Symptom assessment
        - Care plan coordination
        
        **Specialist Care Integration:**
        - Endocrinology: Insulin management
        - Pharmacy: Adherence optimization
        - Nephrology: CKD progression prevention
        - Case Management: Social support coordination
        
        **QUALITY IMPROVEMENT OPPORTUNITIES:**
        
        **System-Level Insights:**
        - Medication adherence program enhancement needed
        - Post-hospitalization transition care gaps identified
        - Diabetes education program effectiveness review
        - Care coordination process optimization opportunity
        
        **RISK REASSESSMENT SCHEDULE:**
        
        **2 Weeks**: Expect 15-20% risk reduction with acute interventions
        **1 Month**: Target 25-35% risk reduction with adherence improvement  
        **3 Months**: Goal of 40-50% risk reduction with comprehensive management
        **6 Months**: Sustained risk reduction maintenance assessment
        
        **FILES GENERATED FOR THIS PATIENT:**
        ```
        📁 patient_SYNTH_004257_explanation/
        ├── 📄 individual_risk_explanation_report.pdf        # Complete clinical analysis
        ├── 🎯 provider_action_checklist.pdf               # Immediate intervention guide
        ├── 👥 patient_family_communication_guide.pdf      # Understandable explanation
        ├── 📊 risk_factor_visualization.png               # SHAP waterfall plot
        ├── 📈 counterfactual_scenarios_analysis.pdf       # "What-if" intervention modeling
        ├── 📋 specialist_referral_orders.pdf              # Pre-filled referral forms
        ├── 🗓️ monitoring_protocol_schedule.pdf            # Structured follow-up plan
        ├── 💊 medication_adherence_action_plan.pdf        # Adherence improvement strategy
        ├── 📞 care_coordination_communication.pdf         # Team communication summary
        └── 🔄 risk_reassessment_timeline.pdf             # Future evaluation schedule
        ```
        
        **CLINICAL VALIDATION CONFIRMATION:**
        ✅ Risk factors clinically validated by specialist review
        ✅ Recommendations align with diabetes management guidelines  
        ✅ Intervention timeline appropriate for risk level
        ✅ Patient communication reviewed for clarity and accuracy
        ✅ Care coordination plan feasible within healthcare system
        ✅ Monitoring protocol evidence-based and practical
        
        **CONCLUSION:**
        This patient requires immediate, intensive diabetes management with a focus on 
        medication adherence optimization and glycemic control improvement. The explanation 
        provides clear rationale for high-risk classification and actionable pathways for 
        risk reduction through coordinated clinical intervention.
        """
