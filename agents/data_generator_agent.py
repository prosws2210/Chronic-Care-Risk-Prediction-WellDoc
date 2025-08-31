"""
Data Generator Agent for creating synthetic chronic care patient data.
Specializes in generating realistic patient profiles, vitals, and medical histories.
"""

import logging
from typing import Dict, Any, List
from crewai import Agent, LLM

from tools.data_tools.patient_data_generator import PatientDataGeneratorTool
from tools.data_tools.vital_simulator import VitalSimulatorTool
from tools.data_tools.lab_result_simulator import LabResultSimulatorTool
from tools.data_tools.medication_adherence_tool import MedicationAdherenceTool

logger = logging.getLogger(__name__)

class DataGeneratorAgent:
    """Agent responsible for generating comprehensive synthetic patient datasets."""
    
    def __init__(self, config: Any, llm: LLM):
        """Initialize the Data Generator Agent."""
        self.config = config
        self.llm = llm
        
        # Initialize specialized tools
        self.tools = [
            PatientDataGeneratorTool(),
            VitalSimulatorTool(),
            LabResultSimulatorTool(),
            MedicationAdherenceTool()
        ]
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Chief Data Synthesis Specialist",
            goal=(
                "Generate comprehensive, realistic synthetic patient datasets for chronic care "
                "risk prediction model training, ensuring clinical accuracy and diversity"
            ),
            backstory=(
                "Expert biostatistician with 15+ years in healthcare data science. "
                "Pioneered synthetic patient data generation methods that preserve "
                "clinical validity while ensuring privacy. Specialized in chronic "
                "condition data patterns including diabetes, heart failure, and obesity. "
                "Deep understanding of temporal medical data relationships and "
                "population health demographics."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt()
        )
        
        logger.info("DataGeneratorAgent initialized successfully")
    
    def _get_system_prompt(self) -> str:
        """Get the detailed system prompt for the agent."""
        return f"""
CHRONIC CARE DATA GENERATION SPECIALIST

MISSION: Create realistic synthetic patient datasets for chronic care risk prediction.

CORE RESPONSIBILITIES:
1. Generate {self.config.SYNTHETIC_PATIENTS_COUNT} diverse patient profiles
2. Simulate {self.config.MIN_HISTORY_DAYS}-{self.config.MAX_HISTORY_DAYS} days of medical history
3. Create realistic chronic condition patterns (diabetes, heart failure, obesity)
4. Ensure demographic diversity and clinical validity

PATIENT PROFILE COMPONENTS:
- Demographics: Age (18-90), gender, ethnicity, socioeconomic factors
- Medical History: Chronic conditions, comorbidities, family history
- Baseline Vitals: BP, HR, weight, BMI, temperature
- Laboratory Values: HbA1c, glucose, lipids, kidney function, inflammatory markers
- Medications: Prescriptions, adherence patterns, side effects
- Lifestyle: Diet, exercise, smoking, alcohol, sleep patterns
- Healthcare Utilization: Visits, hospitalizations, procedures

CLINICAL REALISM REQUIREMENTS:
- Physiologically plausible value ranges
- Appropriate correlations between variables
- Realistic disease progression patterns
- Proper medication effects on biomarkers
- Age and gender-appropriate norms

DATA QUALITY STANDARDS:
- 10% missing data (realistic clinical scenario)
- Temporal consistency in measurements
- Appropriate seasonal variations
- Medication adherence variability (60-95%)
- Risk factor clustering by condition

DETERIORATION RISK MODELING:
- 15% high-risk patients (deterioration within 90 days)
- 25% medium-risk patients
- 60% low-risk patients
- Include clear risk factor patterns

OUTPUT FORMAT:
Generate structured JSON data with patient IDs, demographics, time-series medical data,
and ground truth deterioration outcomes for model training.

ETHICAL CONSIDERATIONS:
- No real patient data replication
- Maintain statistical realism without individual identifiability
- Ensure diverse representation across demographics
"""
    
    def generate_patient_cohort(self, patient_count: int = None) -> Dict[str, Any]:
        """Generate a complete synthetic patient cohort."""
        if patient_count is None:
            patient_count = self.config.SYNTHETIC_PATIENTS_COUNT
        
        logger.info(f"Generating synthetic cohort of {patient_count} patients")
        
        # This would be called through CrewAI task execution
        # Implementation details handled by the tools
        return {
            "status": "ready",
            "patient_count": patient_count,
            "agent": "DataGeneratorAgent"
        }
