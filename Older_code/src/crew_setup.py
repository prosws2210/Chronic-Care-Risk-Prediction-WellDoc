"""
CrewAI setup and orchestration for the Chronic Care Risk Prediction Engine.
Manages all agents, tasks, and workflows for the healthcare AI system.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from crewai import Agent, Task, Crew, LLM
from crewai.process import Process

# Import agents
from agents.data_generator_agent import DataGeneratorAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.model_evaluator_agent import ModelEvaluatorAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.clinical_validator_agent import ClinicalValidatorAgent
from agents.risk_assessor_agent import RiskAssessorAgent

# Import specialists
from agents.specialists.diabetes_specialist import DiabetesSpecialistAgent
from agents.specialists.cardiology_specialist import CardiologySpecialistAgent
from agents.specialists.obesity_specialist import ObesitySpecialistAgent

# Import evaluators
from agents.evaluators.performance_evaluator import PerformanceEvaluatorAgent
from agents.evaluators.bias_detector import BiasDetectorAgent  
from agents.evaluators.clinical_safety_evaluator import ClinicalSafetyEvaluatorAgent

# Import tasks
from tasks.generation.synthetic_data_task import SyntheticDataTask
from tasks.generation.feature_engineering_task import FeatureEngineeringTask
from tasks.training.model_training_task import ModelTrainingTask
from tasks.training.hyperparameter_tuning_task import HyperparameterTuningTask
from tasks.evaluation.model_evaluation_task import ModelEvaluationTask
from tasks.evaluation.clinical_validation_task import ClinicalValidationTask
from tasks.explanation.global_explanation_task import GlobalExplanationTask
from tasks.explanation.local_explanation_task import LocalExplanationTask

from src.config import Config

logger = logging.getLogger(__name__)

class ChronicCareRiskCrew:
    """Main CrewAI orchestrator for chronic care risk prediction."""
    
    def __init__(self, config: Config):
        """Initialize the chronic care risk prediction crew."""
        self.config = config
        self.llm = self._setup_llm()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize tasks
        self.tasks = self._initialize_tasks()
        
        # Initialize crews for different workflows
        self.crews = self._initialize_crews()
        
        logger.info("ChronicCareRiskCrew initialized successfully")
    
    def _setup_llm(self) -> LLM:
        """Set up the primary language model."""
        try:
            # Set OpenAI API key if available
            if self.config.OPENAI_API_KEY:
                os.environ["OPENAI_API_KEY"] = self.config.OPENAI_API_KEY
            
            # Create LLM instance
            llm = LLM(
                model=self.config.PRIMARY_MODEL,
                temperature=self.config.TEMPERATURE,
                max_tokens=self.config.MAX_TOKENS
            )
            
            logger.info(f"LLM initialized: {self.config.PRIMARY_MODEL}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents for the system."""
        agents = {}
        
        try:
            # Core agents
            agents['data_generator'] = DataGeneratorAgent(self.config, self.llm)
            agents['model_trainer'] = ModelTrainerAgent(self.config, self.llm)
            agents['model_evaluator'] = ModelEvaluatorAgent(self.config, self.llm)
            agents['explainability'] = ExplainabilityAgent(self.config, self.llm)
            agents['clinical_validator'] = ClinicalValidatorAgent(self.config, self.llm)
            agents['risk_assessor'] = RiskAssessorAgent(self.config, self.llm)
            
            # Specialist agents
            agents['diabetes_specialist'] = DiabetesSpecialistAgent(self.config, self.llm)
            agents['cardiology_specialist'] = CardiologySpecialistAgent(self.config, self.llm)
            agents['obesity_specialist'] = ObesitySpecialistAgent(self.config, self.llm)
            
            # Evaluator agents
            agents['performance_evaluator'] = PerformanceEvaluatorAgent(self.config, self.llm)
            agents['bias_detector'] = BiasDetectorAgent(self.config, self.llm)
            agents['safety_evaluator'] = ClinicalSafetyEvaluatorAgent(self.config, self.llm)
            
            logger.info(f"Initialized {len(agents)} agents")
            return agents
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    
    def _initialize_tasks(self) -> Dict[str, Task]:
        """Initialize all tasks for the system."""
        tasks = {}
        
        try:
            # Data generation tasks
            tasks['synthetic_data'] = SyntheticDataTask(self.config, self.agents)
            tasks['feature_engineering'] = FeatureEngineeringTask(self.config, self.agents)
            
            # Training tasks
            tasks['model_training'] = ModelTrainingTask(self.config, self.agents)
            tasks['hyperparameter_tuning'] = HyperparameterTuningTask(self.config, self.agents)
            
            # Evaluation tasks
            tasks['model_evaluation'] = ModelEvaluationTask(self.config, self.agents)
            tasks['clinical_validation'] = ClinicalValidationTask(self.config, self.agents)
            
            # Explanation tasks
            tasks['global_explanation'] = GlobalExplanationTask(self.config, self.agents)
            tasks['local_explanation'] = LocalExplanationTask(self.config, self.agents)
            
            logger.info(f"Initialized {len(tasks)} tasks")
            return tasks
            
        except Exception as e:
            logger.error(f"Failed to initialize tasks: {str(e)}")
            raise
    
    def _initialize_crews(self) -> Dict[str, Crew]:
        """Initialize crews for different workflows."""
        crews = {}
        
        try:
            # Data generation crew
            crews['data_generation'] = Crew(
                agents=[
                    self.agents['data_generator'].agent,
                    self.agents['diabetes_specialist'].agent,
                    self.agents['cardiology_specialist'].agent,
                    self.agents['obesity_specialist'].agent
                ],
                tasks=[
                    self.tasks['synthetic_data'].task,
                    self.tasks['feature_engineering'].task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Model training crew
            crews['model_training'] = Crew(
                agents=[
                    self.agents['model_trainer'].agent,
                    self.agents['diabetes_specialist'].agent,
                    self.agents['cardiology_specialist'].agent,
                    self.agents['obesity_specialist'].agent
                ],
                tasks=[
                    self.tasks['model_training'].task,
                    self.tasks['hyperparameter_tuning'].task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Model evaluation crew
            crews['model_evaluation'] = Crew(
                agents=[
                    self.agents['model_evaluator'].agent,
                    self.agents['performance_evaluator'].agent,
                    self.agents['bias_detector'].agent,
                    self.agents['safety_evaluator'].agent
                ],
                tasks=[
                    self.tasks['model_evaluation'].task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Explanation crew
            crews['explanation'] = Crew(
                agents=[
                    self.agents['explainability'].agent,
                    self.agents['clinical_validator'].agent
                ],
                tasks=[
                    self.tasks['global_explanation'].task,
                    self.tasks['local_explanation'].task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Clinical validation crew
            crews['clinical_validation'] = Crew(
                agents=[
                    self.agents['clinical_validator'].agent,
                    self.agents['diabetes_specialist'].agent,
                    self.agents['cardiology_specialist'].agent,
                    self.agents['obesity_specialist'].agent,
                    self.agents['safety_evaluator'].agent
                ],
                tasks=[
                    self.tasks['clinical_validation'].task
                ],
                process=Process.sequential,
                verbose=True
            )
            
            # Risk assessment crew (for real-time predictions)
            crews['risk_assessment'] = Crew(
                agents=[
                    self.agents['risk_assessor'].agent,
                    self.agents['explainability'].agent
                ],
                tasks=[],  # Tasks will be created dynamically
                process=Process.sequential,
                verbose=True
            )
            
            logger.info(f"Initialized {len(crews)} crews")
            return crews
            
        except Exception as e:
            logger.error(f"Failed to initialize crews: {str(e)}")
            raise
    
    def run_data_generation(self) -> Optional[Dict[str, Any]]:
        """Execute the data generation workflow."""
        try:
            logger.info("Starting data generation workflow...")
            
            crew = self.crews['data_generation']
            result = crew.kickoff()
            
            logger.info("Data generation workflow completed")
            return self._process_crew_result(result, 'data_generation')
            
        except Exception as e:
            logger.error(f"Data generation workflow failed: {str(e)}")
            return None
    
    def run_model_training(self) -> Optional[Dict[str, Any]]:
        """Execute the model training workflow."""
        try:
            logger.info("Starting model training workflow...")
            
            crew = self.crews['model_training']
            result = crew.kickoff()
            
            logger.info("Model training workflow completed")
            return self._process_crew_result(result, 'model_training')
            
        except Exception as e:
            logger.error(f"Model training workflow failed: {str(e)}")
            return None
    
    def run_model_evaluation(self) -> Optional[Dict[str, Any]]:
        """Execute the model evaluation workflow."""
        try:
            logger.info("Starting model evaluation workflow...")
            
            crew = self.crews['model_evaluation']
            result = crew.kickoff()
            
            logger.info("Model evaluation workflow completed")
            return self._process_crew_result(result, 'model_evaluation')
            
        except Exception as e:
            logger.error(f"Model evaluation workflow failed: {str(e)}")
            return None
    
    def run_explanation_generation(self) -> Optional[Dict[str, Any]]:
        """Execute the explanation generation workflow."""
        try:
            logger.info("Starting explanation generation workflow...")
            
            crew = self.crews['explanation']
            result = crew.kickoff()
            
            logger.info("Explanation generation workflow completed")
            return self._process_crew_result(result, 'explanation')
            
        except Exception as e:
            logger.error(f"Explanation generation workflow failed: {str(e)}")
            return None
    
    def run_clinical_validation(self) -> Optional[Dict[str, Any]]:
        """Execute the clinical validation workflow."""
        try:
            logger.info("Starting clinical validation workflow...")
            
            crew = self.crews['clinical_validation']
            result = crew.kickoff()
            
            logger.info("Clinical validation workflow completed")
            return self._process_crew_result(result, 'clinical_validation')
            
        except Exception as e:
            logger.error(f"Clinical validation workflow failed: {str(e)}")
            return None
    
    def run_risk_assessment(self, patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute risk assessment for a specific patient."""
        try:
            logger.info("Starting risk assessment...")
            
            # Create dynamic task for risk assessment
            risk_task = Task(
                description=f"Assess 90-day deterioration risk for patient data: {patient_data}",
                expected_output="Risk score (0-100%) with explanations and recommendations",
                agent=self.agents['risk_assessor'].agent
            )
            
            # Create temporary crew with the dynamic task
            temp_crew = Crew(
                agents=[
                    self.agents['risk_assessor'].agent,
                    self.agents['explainability'].agent
                ],
                tasks=[risk_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = temp_crew.kickoff()
            
            logger.info("Risk assessment completed")
            return self._process_crew_result(result, 'risk_assessment')
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return None
    
    def _process_crew_result(self, result: Any, workflow_name: str) -> Dict[str, Any]:
        """Process and format crew execution results."""
        try:
            timestamp = datetime.now().isoformat()
            
            processed_result = {
                "workflow": workflow_name,
                "timestamp": timestamp,
                "status": "success",
                "result": result,
                "metadata": {
                    "config": self.config.get_model_config(),
                    "agents_used": list(self.agents.keys()),
                    "tasks_executed": list(self.tasks.keys())
                }
            }
            
            # Save result to outputs
            output_file = self.config.OUTPUTS_DIR / f"{workflow_name}_result_{timestamp.replace(':', '-')}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_file, 'w') as f:
                json.dump(processed_result, f, indent=2, default=str)
            
            logger.info(f"Result saved to: {output_file}")
            return processed_result
            
        except Exception as e:
            logger.error(f"Failed to process crew result: {str(e)}")
            return {
                "workflow": workflow_name,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def validate_ai_models(self) -> bool:
        """Validate that AI models are accessible."""
        try:
            # Test primary LLM
            test_prompt = "Hello, this is a test."
            response = self.llm.call(test_prompt)
            
            if response:
                logger.info("AI model validation successful")
                return True
            else:
                logger.error("AI model validation failed - no response")
                return False
                
        except Exception as e:
            logger.error(f"AI model validation failed: {str(e)}")
            return False
    
    def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents."""
        status = {}
        
        for name, agent_wrapper in self.agents.items():
            try:
                status[name] = {
                    "initialized": True,
                    "role": agent_wrapper.agent.role,
                    "goal": agent_wrapper.agent.goal,
                    "tools_count": len(agent_wrapper.agent.tools) if agent_wrapper.agent.tools else 0
                }
            except Exception as e:
                status[name] = {
                    "initialized": False,
                    "error": str(e)
                }
        
        return status
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "llm_status": self.validate_ai_models(),
                "agents_status": self.get_agent_status(),
                "config_valid": self.config.validate()["valid"],
                "crews_initialized": len(self.crews),
                "tasks_initialized": len(self.tasks)
            }
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
