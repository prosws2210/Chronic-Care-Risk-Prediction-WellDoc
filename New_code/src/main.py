import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from typing import Optional, Dict, List
from pydantic import BaseModel
import joblib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from config.settings import *
from agents.healthcare_agents import create_healthcare_agents
from tasks.prediction_tasks import create_prediction_tasks
from tools.data_tools import DataPreprocessingTool, ModelTrainingTool, ModelEvaluationTool
from tools.visualization_tools import VisualizationTool, DashboardTool

# ---- Logging Configuration ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/training/risk_prediction_{timestamp}.txt"

# Ensure log directory exists
os.makedirs("logs/training", exist_ok=True)

logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"🏥 Risk Prediction Pipeline initialized - Log: {log_filename}")

# ---- File Configuration ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PREPROCESSING_REPORT = f"outputs/reports/preprocessing_report_{timestamp}.json"
MODEL_RESULTS_FILE = f"outputs/reports/model_results_{timestamp}.json"
EVALUATION_REPORT = f"outputs/reports/evaluation_report_{timestamp}.json"
FEATURE_IMPORTANCE_FILE = f"outputs/reports/feature_importance_{timestamp}.json"
PATIENT_PREDICTIONS_FILE = f"outputs/reports/patient_predictions_{timestamp}.json"
CLINICAL_VALIDATION_FILE = f"outputs/reports/clinical_validation_{timestamp}.json"
EXPLANATION_RESULTS_FILE = f"outputs/reports/explanation_results_{timestamp}.json"

# ---- Model Loading Functions (from your original code pattern) ----
def is_model_loaded(model_name: str) -> bool:
    """Check if model is loaded in LM Studio"""
    try:
        import requests
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()["data"]
            return any(model_name in m["id"] for m in models)
        return False
    except Exception:
        return False

def load_model(model_name: str):
    """Load model in LM Studio"""
    logger.info(f"Loading model: {model_name}")
    try:
        import requests
        response = requests.post(
            f"{BASE_URL}/models/load",
            json={"model_path": model_name},
            timeout=MODEL_LOAD_TIMEOUT
        )
        if response.status_code == 200:
            logger.info(f"Model loaded successfully: {model_name}")
            return True
        logger.error(f"Failed to load model: {response.text}")
        return False
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        return False

def ensure_model_loaded(model_name: str):
    """Ensure model is loaded before proceeding"""
    if is_model_loaded(model_name):
        logger.info(f"Model already loaded: {model_name}")
        return True
    return load_model(model_name)

# ---- LLM Setup (following your pattern) ----
os.environ["OPENAI_API_KEY"] = "dummy-key"

# Ensure models are loaded
ensure_model_loaded(RISK_MODEL_NAME)
ensure_model_loaded(EXPLANATION_MODEL_NAME)

# Initialize LLMs
llm_primary = LLM(
    model=f"lm_studio/{RISK_MODEL_NAME}",
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT
)

llm_secondary = LLM(
    model=f"lm_studio/{EXPLANATION_MODEL_NAME}",
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=TIMEOUT
)

class RiskPredictionPipeline:
    """Main Risk Prediction Pipeline orchestrating all agents and tasks"""
    
    def __init__(self):
        logger.info("🔧 Initializing Risk Prediction Pipeline...")
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Create agents
        self.agents = create_healthcare_agents(llm_primary, llm_secondary, self.tools)
        
        # Create tasks
        self.tasks = create_prediction_tasks(self.agents)
        
        # Create crew
        self.crew = self._create_crew()
        
        logger.info("✅ Pipeline initialization complete")
    
    def _initialize_tools(self):
        """Initialize all tools for the pipeline"""
        return {
            'data_preprocessing': DataPreprocessingTool(),
            'model_training': ModelTrainingTool(),
            'model_evaluation': ModelEvaluationTool(),
            'visualization': VisualizationTool(),
            'dashboard': DashboardTool()
        }
    
    def _create_crew(self):
        """Create CrewAI crew with all agents and tasks"""
        return Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            verbose=True,
            memory=True
        )
    
    def run_full_pipeline(self, data_path: str = "data/processed/chronic_care_data.csv"):
        """Execute the complete risk prediction pipeline"""
        try:
            logger.info("🚀 Starting Complete Risk Prediction Pipeline...")
            
            # Step 1: Data Preprocessing
            logger.info("📊 Step 1: Clinical Data Preprocessing...")
            preprocessing_result = self._run_preprocessing_step(data_path)
            
            # Step 2: Model Training
            logger.info("🤖 Step 2: Risk Prediction Model Training...")
            training_result = self._run_training_step(preprocessing_result)
            
            # Step 3: Model Evaluation
            logger.info("📈 Step 3: Model Performance Evaluation...")
            evaluation_result = self._run_evaluation_step(training_result)
            
            # Step 4: Generate Explanations
            logger.info("💡 Step 4: Generating Clinical Explanations...")
            explanation_result = self._run_explanation_step(evaluation_result)
            
            # Step 5: Clinical Validation
            logger.info("🩺 Step 5: Clinical Validation...")
            validation_result = self._run_validation_step(explanation_result)
            
            # Step 6: Generate Visualizations
            logger.info("📊 Step 6: Creating Clinical Visualizations...")
            viz_result = self._run_visualization_step(validation_result)
            
            # Compile final results
            final_results = self._compile_final_results({
                'preprocessing': preprocessing_result,
                'training': training_result,
                'evaluation': evaluation_result,
                'explanation': explanation_result,
                'validation': validation_result,
                'visualization': viz_result
            })
            
            logger.info("🎉 Pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            return None
    
    def _run_preprocessing_step(self, data_path):
        """Execute data preprocessing step"""
        preprocessing_task = Task(
            description=(
                f"Preprocess chronic care patient data for risk prediction:\n"
                f"1. Load or generate synthetic patient data from {data_path}\n"
                f"2. Handle missing values using clinical best practices\n"
                f"3. Create temporal features from longitudinal data\n"
                f"4. Engineer clinically meaningful risk features\n"
                f"5. Validate data quality and clinical ranges\n"
                f"6. Prepare dataset for machine learning training\n\n"
                f"Generate comprehensive data quality report with patient cohort statistics."
            ),
            expected_output="Comprehensive preprocessing report with processed dataset location",
            agent=self.agents['data_processor'],
            output_file=PREPROCESSING_REPORT
        )
        
        result = self.agents['data_processor'].execute_task(preprocessing_task)
        self._save_step_result('preprocessing', result, PREPROCESSING_REPORT)
        return result
    
    def _run_training_step(self, preprocessing_result):
        """Execute model training step"""
        training_task = Task(
            description=(
                "Train machine learning models for 90-day deterioration risk prediction:\n"
                "1. Load preprocessed patient data\n"
                "2. Split data with proper stratification and temporal validation\n"
                "3. Train ensemble of models (Random Forest, XGBoost, Neural Networks)\n"
                "4. Optimize for clinical metrics (sensitivity ≥80%, specificity ≥75%)\n"
                "5. Ensure model calibration for probability interpretation\n"
                "6. Select best model balancing performance and interpretability\n\n"
                "Target: AUROC ≥ 0.75, AUPRC ≥ 0.65, well-calibrated predictions"
            ),
            expected_output="Model training results with performance metrics and saved models",
            agent=self.agents['risk_assessor'],
            output_file=MODEL_RESULTS_FILE
        )
        
        result = self.agents['risk_assessor'].execute_task(training_task)
        self._save_step_result('training', result, MODEL_RESULTS_FILE)
        return result
    
    def _run_evaluation_step(self, training_result):
        """Execute model evaluation step"""
        evaluation_task = Task(
            description=(
                "Comprehensively evaluate risk prediction model performance:\n"
                "1. Calculate clinical metrics (AUROC, AUPRC, sensitivity, specificity)\n"
                "2. Generate confusion matrix and calibration plots\n"
                "3. Perform subgroup analysis by age, gender, chronic conditions\n"
                "4. Assess temporal stability and performance consistency\n"
                "5. Evaluate clinical utility and decision impact\n"
                "6. Generate performance visualization suite\n\n"
                "Focus on clinically relevant metrics and actionable insights."
            ),
            expected_output="Comprehensive evaluation report with clinical interpretation",
            agent=self.agents['evaluator'],
            output_file=EVALUATION_REPORT
        )
        
        result = self.agents['evaluator'].execute_task(evaluation_task)
        self._save_step_result('evaluation', result, EVALUATION_REPORT)
        return result
    
    def _run_explanation_step(self, evaluation_result):
        """Execute explanation generation step"""
        explanation_task = Task(
            description=(
                "Generate clinician-friendly explanations for risk predictions:\n"
                "1. Calculate SHAP values for global and local explanations\n"
                "2. Identify key risk factors and feature interactions\n"
                "3. Translate technical features into clinical language\n"
                "4. Generate patient-specific risk factor contributions\n"
                "5. Create actionable insights for care team interventions\n"
                "6. Develop explanation visualizations and summaries\n\n"
                "Ensure explanations support clinical decision-making and patient care."
            ),
            expected_output="Explanation package with global and local interpretations",
            agent=self.agents['explainer'],
            output_file=EXPLANATION_RESULTS_FILE
        )
        
        result = self.agents['explainer'].execute_task(explanation_task)
        self._save_step_result('explanation', result, EXPLANATION_RESULTS_FILE)
        return result
    
    def _run_validation_step(self, explanation_result):
        """Execute clinical validation step"""
        validation_task = Task(
            description=(
                "Validate predictions against clinical guidelines and best practices:\n"
                "1. Check alignment with ADA, ACC/AHA, and other clinical guidelines\n"
                "2. Identify biologically implausible predictions\n"
                "3. Assess algorithmic fairness and bias across populations\n"
                "4. Validate clinical actionability of recommendations\n"
                "5. Review for safety concerns and contraindications\n"
                "6. Ensure evidence-based care pathway alignment\n\n"
                "Prioritize patient safety and clinical workflow integration."
            ),
            expected_output="Clinical validation report with safety assessment",
            agent=self.agents['clinical_validator'],
            output_file=CLINICAL_VALIDATION_FILE
        )
        
        result = self.agents['clinical_validator'].execute_task(validation_task)
        self._save_step_result('validation', result, CLINICAL_VALIDATION_FILE)
        return result
    
    def _run_visualization_step(self, validation_result):
        """Execute visualization generation step"""
        viz_task = Task(
            description=(
                "Create comprehensive visualizations for clinical dashboard:\n"
                "1. Generate risk distribution plots and patient cohort analysis\n"
                "2. Create time series visualizations for vital signs trends\n"
                "3. Design feature importance charts and SHAP plots\n"
                "4. Build calibration plots and performance metrics displays\n"
                "5. Develop patient-level risk factor contribution charts\n"
                "6. Create dashboard-ready visualization components\n\n"
                "Optimize for clinical workflow integration and decision support."
            ),
            expected_output="Complete visualization suite for clinical dashboard",
            agent=self.agents['visualizer'],
            output_file="outputs/reports/visualization_results.json"
        )
        
        result = self.agents['visualizer'].execute_task(viz_task)
        return result
    
    def _save_step_result(self, step_name, result, filepath):
        """Save individual step results"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                except:
                    result_dict = {"raw_result": result}
            else:
                result_dict = result
            
            with open(filepath, "w") as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            logger.info(f"✅ {step_name.capitalize()} results saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save {step_name} results: {str(e)}")
    
    def _compile_final_results(self, all_results):
        """Compile and save final pipeline results"""
        try:
            final_report = {
                "pipeline_info": {
                    "execution_timestamp": datetime.now().isoformat(),
                    "pipeline_version": "1.0.0",
                    "total_steps": len(all_results),
                    "success": True
                },
                "step_results": all_results,
                "output_files": {
                    "preprocessing_report": PREPROCESSING_REPORT,
                    "model_results": MODEL_RESULTS_FILE,
                    "evaluation_report": EVALUATION_REPORT,
                    "explanation_results": EXPLANATION_RESULTS_FILE,
                    "clinical_validation": CLINICAL_VALIDATION_FILE
                },
                "next_steps": {
                    "dashboard_launch": "streamlit run dashboard/app.py",
                    "model_deployment": "models/saved/risk_prediction_model.pkl",
                    "clinical_integration": "Review validation results before deployment"
                }
            }
            
            # Save comprehensive final report
            final_report_path = f"outputs/reports/final_pipeline_report_{timestamp}.json"
            with open(final_report_path, "w") as f:
                json.dump(final_report, f, indent=2, default=str)
            
            logger.info(f"📋 Final pipeline report saved to {final_report_path}")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Failed to compile final results: {str(e)}")
            return {"error": str(e)}

def run_quick_demo():
    """Run a quick demonstration of the pipeline"""
    logger.info("🎯 Running Quick Demo Pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = RiskPredictionPipeline()
        
        # Run with minimal data for demo
        results = pipeline.run_full_pipeline()
        
        if results and results.get("pipeline_info", {}).get("success"):
            print("\n" + "="*60)
            print("🎉 DEMO PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"📊 Preprocessing: {'✅' if 'preprocessing' in results['step_results'] else '❌'}")
            print(f"🤖 Model Training: {'✅' if 'training' in results['step_results'] else '❌'}")
            print(f"📈 Evaluation: {'✅' if 'evaluation' in results['step_results'] else '❌'}")
            print(f"💡 Explanations: {'✅' if 'explanation' in results['step_results'] else '❌'}")
            print(f"🩺 Validation: {'✅' if 'validation' in results['step_results'] else '❌'}")
            print(f"📊 Visualizations: {'✅' if 'visualization' in results['step_results'] else '❌'}")
            print("\n🚀 Next Steps:")
            print("   1. Launch dashboard: streamlit run dashboard/app.py")
            print("   2. Review reports in: outputs/reports/")
            print("   3. Check model artifacts: models/saved/")
            print("="*60)
        else:
            print("❌ Demo pipeline failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"Demo pipeline failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")

def main():
    """Main execution function"""
    try:
        print("\n🏥 AI-DRIVEN RISK PREDICTION ENGINE")
        print("="*50)
        print("Chronic Care Patient Deterioration Prediction")
        print("="*50)
        
        # Check if this is a demo run
        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
            run_quick_demo()
            return
        
        # Initialize full pipeline
        pipeline = RiskPredictionPipeline()
        
        # Run complete pipeline
        data_path = "data/processed/chronic_care_data.csv"
        logger.info(f"🎯 Starting full pipeline with data path: {data_path}")
        
        results = pipeline.run_full_pipeline(data_path)
        
        if results and results.get("pipeline_info", {}).get("success"):
            print(f"\n{'='*60}")
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"📁 Results Location: outputs/reports/")
            print(f"🤖 Model Location: models/saved/")
            print(f"📊 Dashboard Command: streamlit run dashboard/app.py")
            print(f"📋 Final Report: {results['output_files']['preprocessing_report']}")
            print(f"{'='*60}")
        else:
            print("❌ Pipeline execution failed. Check logs for details.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⏹️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"❌ Pipeline failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
