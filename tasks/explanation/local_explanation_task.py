"""
Local Explanation Task for individual patient risk interpretation.
Generates personalized explanations for specific patient predictions.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from crewai import Task

logger = logging.getLogger(__name__)

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
        
        **SHAP EXPLANATION BREAKDOWN:**
        
        **Risk Factor Contributions:**
        | Factor | SHAP Value | Direction | Clinical Impact | Modifiable |
        |---------|------------|-----------|-----------------|------------|
        | HbA1c Trend (10.2%) | +0.187 | ↑ Risk | Critical diabetes control | ✅ Yes |
        | Med Adherence (42%) | +0.152 | ↑ Risk | Poor compliance pattern | ✅ Yes |
        | Recent Hospital (8d) | +0.124 | ↑ Risk | System stress indicator | ⏰ Time |
        | eGFR Decline (38) | +0.098 | ↑ Risk | Kidney function loss | 🔄 Partially |
        | Comorbidities (5) | +0.073 | ↑ Risk | Disease complexity | ❌ No |
        | Age (78 years) | +0.061 | ↑ Risk | Advanced age factor | ❌ No |
        | Specialist Care | -0.032 | ↓ Risk | Good engagement | ✅ Maintain |
        | Family Support | -0.021 | ↓ Risk | Social protective factor | ✅ Leverage |
        
        **CLINICAL INTERPRETATION & IMMEDIATE ACTIONS:**
        
        **🚨 URGENT INTERVENTIONS (24-48 hours):**
        1. **Endocrinology Referral** - DKA management and insulin optimization
        2. **Medication Review** - Address adherence barriers immediately  
        3. **Care Coordination** - Activate intensive case management
        4. **Safety Monitoring** - Daily clinical contact until stable
        
        **📋 TARGETED RISK REDUCTION PLAN:**
        - **Primary Target**: Medication adherence (42% → 85% = -21% risk reduction)
        - **Secondary Target**: HbA1c control (10.2% → 8.0% = -31% risk reduction) 
        - **Combined Impact**: Potential 44% overall risk reduction achievable
        
        **MONITORING PROTOCOL:**
        - Week 1-4: Weekly clinical contact + daily glucose monitoring
        - Month 2-3: Bi-weekly visits + monthly lab work
        - Month 4-6: Monthly follow-up + quarterly comprehensive assessment
        
        **PATIENT COMMUNICATION SUMMARY:**
        "Your diabetes needs immediate attention. We can significantly reduce your risk 
        by improving medication consistency and working with specialists to better 
        control your blood sugar. You have strong family support which helps your 
        recovery chances."
        
        This explanation provides individualized, actionable guidance for immediate 
        clinical intervention while maintaining clear communication pathways for 
        both providers and patients.
        """
