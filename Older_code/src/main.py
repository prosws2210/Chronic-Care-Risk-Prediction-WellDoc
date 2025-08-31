#!/usr/bin/env python3
"""
Main entry point for the Chronic Care Risk Prediction Engine.
Orchestrates the entire multi-agent workflow using CrewAI.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import Config
from src.crew_setup import ChronicCareRiskCrew
from src.utils.logging_setup import setup_logging
from src.utils.file_manager import FileManager

# Initialize logging
logger = setup_logging()

class ChronicCareRiskEngine:
    """Main orchestrator for the chronic care risk prediction system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the risk prediction engine."""
        self.config = Config(config_path)
        self.file_manager = FileManager(self.config)
        self.crew = None
        
        logger.info("Chronic Care Risk Prediction Engine initialized")
    
    def setup(self) -> bool:
        """Set up the system for first-time use."""
        try:
            logger.info("Setting up Chronic Care Risk Prediction Engine...")
            
            # Create necessary directories
            self.file_manager.create_directories()
            
            # Initialize crew
            self.crew = ChronicCareRiskCrew(self.config)
            
            # Validate configuration
            if not self._validate_setup():
                logger.error("Setup validation failed")
                return False
            
            logger.info("Setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Execute the complete risk prediction pipeline."""
        try:
            logger.info("Starting full chronic care risk prediction pipeline...")
            
            if not self.crew:
                self.crew = ChronicCareRiskCrew(self.config)
            
            # Step 1: Generate synthetic patient data
            logger.info("Phase 1: Generating synthetic patient data...")
            data_result = self.crew.run_data_generation()
            if not data_result:
                logger.error("Data generation failed")
                return False
            
            # Step 2: Train risk prediction models
            logger.info("Phase 2: Training risk prediction models...")
            training_result = self.crew.run_model_training()
            if not training_result:
                logger.error("Model training failed")
                return False
            
            # Step 3: Evaluate model performance
            logger.info("Phase 3: Evaluating model performance...")
            evaluation_result = self.crew.run_model_evaluation()
            if not evaluation_result:
                logger.error("Model evaluation failed")
                return False
            
            # Step 4: Generate explanations
            logger.info("Phase 4: Generating model explanations...")
            explanation_result = self.crew.run_explanation_generation()
            if not explanation_result:
                logger.error("Explanation generation failed")
                return False
            
            # Step 5: Clinical validation
            logger.info("Phase 5: Performing clinical validation...")
            validation_result = self.crew.run_clinical_validation()
            if not validation_result:
                logger.error("Clinical validation failed")
                return False
            
            logger.info("Full pipeline completed successfully!")
            self._generate_final_report()
            return True
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            return False
    
    def run_prediction_only(self, patient_data_path: str) -> Optional[dict]:
        """Run prediction on new patient data."""
        try:
            logger.info(f"Running prediction on data: {patient_data_path}")
            
            if not self.crew:
                self.crew = ChronicCareRiskCrew(self.config)
            
            # Load patient data
            patient_data = self.file_manager.load_patient_data(patient_data_path)
            if not patient_data:
                logger.error("Failed to load patient data")
                return None
            
            # Run risk assessment
            prediction_result = self.crew.run_risk_assessment(patient_data)
            
            logger.info("Prediction completed successfully")
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None
    
    def launch_dashboard(self) -> bool:
        """Launch the interactive dashboard."""
        try:
            logger.info("Launching chronic care risk dashboard...")
            
            from dashboard.app import create_app
            
            app = create_app(self.config)
            app.run(
                host=self.config.DASHBOARD_HOST,
                port=self.config.DASHBOARD_PORT,
                debug=self.config.DEBUG_MODE
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Dashboard launch failed: {str(e)}")
            return False
    
    def _validate_setup(self) -> bool:
        """Validate system setup."""
        required_dirs = [
            self.config.DATA_DIR,
            self.config.MODELS_DIR,
            self.config.OUTPUTS_DIR,
            self.config.LOGS_DIR
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                logger.error(f"Required directory missing: {dir_path}")
                return False
        
        # Validate AI model access
        if not self.crew.validate_ai_models():
            logger.error("AI model validation failed")
            return False
        
        return True
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.config.OUTPUTS_DIR / f"final_report_{timestamp}.md"
            
            report_content = f"""# Chronic Care Risk Prediction Engine - Final Report

## Execution Summary
- **Timestamp**: {datetime.now().isoformat()}
- **Configuration**: {self.config.config_file}
- **Total Patients Generated**: {self.config.SYNTHETIC_PATIENTS_COUNT}
- **Prediction Window**: {self.config.PREDICTION_WINDOW_DAYS} days

## Pipeline Results
- ✅ Data Generation: Completed
- ✅ Model Training: Completed  
- ✅ Model Evaluation: Completed
- ✅ Explanation Generation: Completed
- ✅ Clinical Validation: Completed

## Output Files
- **Synthetic Data**: {self.config.DATA_DIR / 'synthetic'}
- **Trained Models**: {self.config.MODELS_DIR / 'trained'}
- **Evaluation Reports**: {self.config.OUTPUTS_DIR / 'reports'}
- **Predictions**: {self.config.OUTPUTS_DIR / 'predictions'}

## Next Steps
1. Review model performance metrics
2. Launch dashboard for visualization
3. Begin clinical pilot program
4. Prepare for regulatory review

---
Generated by Chronic Care Risk Prediction Engine v1.0.0
"""
            
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Final report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate final report: {str(e)}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Chronic Care Risk Prediction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --setup                    # First-time setup
  python src/main.py --run-pipeline             # Full pipeline execution
  python src/main.py --predict data.json        # Prediction only
  python src/main.py --dashboard                # Launch dashboard
        """
    )
    
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up the system for first-time use"
    )
    
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="Execute the complete risk prediction pipeline"
    )
    
    parser.add_argument(
        "--predict",
        type=str,
        metavar="DATA_FILE",
        help="Run prediction on specified patient data file"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the interactive dashboard"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        metavar="CONFIG_FILE",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Adjust logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize engine
    engine = ChronicCareRiskEngine(args.config)
    
    # Execute requested action
    if args.setup:
        success = engine.setup()
        sys.exit(0 if success else 1)
    
    elif args.run_pipeline:
        success = engine.run_full_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.predict:
        result = engine.run_prediction_only(args.predict)
        if result:
            print(f"Risk Score: {result.get('risk_score', 'N/A')}%")
            print(f"Risk Level: {result.get('risk_level', 'N/A')}")
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.dashboard:
        success = engine.launch_dashboard()
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
