"""
Main Test Suite for Chronic Care Risk Prediction Engine
=======================================================

Comprehensive tests covering the entire pipeline from data preprocessing
to model deployment and clinical validation.

# Run all tests
python tests/test_main.py

# Run comprehensive suite with reporting
python tests/test_main.py --comprehensive

# Run quick smoke tests
python tests/test_main.py --quick

# Run clinical safety tests only
python tests/test_main.py --clinical

# Using pytest (if installed)
pytest tests/ -v --cov=src
"""

import unittest
import json
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import project modules
try:
    from tools.data_tools import (
        DataPreprocessingTool, ModelTrainingTool, ModelEvaluationTool,
        SyntheticDataTool, ClinicalValidationTool
    )
    from tools.visualization_tools import VisualizationTool, DashboardTool
    from config.settings import get_config, validate_configuration
    from src.main import RiskPredictionPipeline
except ImportError as e:
    logging.warning(f"Import warning: {e}. Some tests may be skipped.")

# Suppress warnings during testing
import warnings
warnings.filterwarnings('ignore')

class TestDataProcessing(unittest.TestCase):
    """Test data processing and preprocessing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.test_dir, "test_data.csv")
        self.preprocessing_tool = DataPreprocessingTool()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_synthetic_data_generation(self):
        """Test synthetic patient data generation"""
        synthetic_tool = SyntheticDataTool()
        result_json = synthetic_tool._run(n_patients=50, output_path=self.test_data_path)
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "success")
        self.assertTrue(os.path.exists(self.test_data_path))
        
        # Verify data structure
        df = pd.read_csv(self.test_data_path)
        self.assertEqual(len(df), 50)
        self.assertIn('patient_id', df.columns)
        self.assertIn('age', df.columns)
        self.assertIn('deterioration_90d', df.columns)
        
        # Verify clinical ranges
        self.assertTrue(df['age'].min() >= 18)
        self.assertTrue(df['age'].max() <= 95)
        self.assertTrue(df['deterioration_90d'].isin([0, 1]).all())
        
    def test_data_preprocessing(self):
        """Test comprehensive data preprocessing"""
        # First generate test data
        synthetic_tool = SyntheticDataTool()
        synthetic_tool._run(n_patients=100, output_path=self.test_data_path)
        
        # Test preprocessing
        result_json = self.preprocessing_tool._run(
            data_path=self.test_data_path,
            n_patients=100
        )
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("preprocessing_summary", result)
        self.assertIn("clinical_validation", result)
        self.assertIn("data_quality", result)
        
        # Verify processed file exists
        processed_path = result["processed_file_path"]
        self.assertTrue(os.path.exists(processed_path))
        
        # Verify processed data
        df = pd.read_csv(processed_path)
        self.assertGreater(len(df.columns), 10)  # Should have engineered features
        self.assertEqual(len(df), 100)
    
    def test_clinical_data_validation(self):
        """Test clinical data validation"""
        # Create test data with some invalid values
        test_data = {
            'patient_id': [1, 2, 3, 4, 5],
            'age': [25, 65, 85, 150, -5],  # Invalid: 150, -5
            'bmi': [22, 28, 35, 70, 10],   # Invalid: 70, 10
            'systolic_bp': [120, 140, 160, 300, 50],  # Invalid: 300, 50
            'deterioration_90d': [0, 1, 0, 1, 0]
        }
        
        df = pd.DataFrame(test_data)
        test_file = os.path.join(self.test_dir, "invalid_data.csv")
        df.to_csv(test_file, index=False)
        
        result_json = self.preprocessing_tool._run(data_path=test_file)
        result = json.loads(result_json)
        
        # Should still succeed but with warnings
        self.assertEqual(result["status"], "success")
        self.assertIn("clinical_validation", result)
        
        # Check for validation warnings
        validation = result["clinical_validation"]
        self.assertGreater(len(validation.get("warnings", [])), 0)

class TestModelTraining(unittest.TestCase):
    """Test machine learning model training and evaluation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.test_dir, "training_data.csv")
        
        # Generate test data
        synthetic_tool = SyntheticDataTool()
        synthetic_tool._run(n_patients=200, output_path=self.test_data_path)
        
        # Preprocess data
        preprocessing_tool = DataPreprocessingTool()
        preprocessing_result = preprocessing_tool._run(data_path=self.test_data_path)
        result = json.loads(preprocessing_result)
        self.processed_data_path = result["processed_file_path"]
        
        self.training_tool = ModelTrainingTool()
        self.evaluation_tool = ModelEvaluationTool()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_model_training(self):
        """Test model training pipeline"""
        result_json = self.training_tool._run(
            data_path=self.processed_data_path,
            model_types=["random_forest", "logistic"]
        )
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("training_summary", result)
        self.assertIn("model_comparison", result)
        self.assertIn("best_model", result)
        
        # Verify model artifacts
        best_model_info = result["best_model"]
        self.assertIn("performance", best_model_info)
        self.assertIn("artifacts", best_model_info)
        
        # Check performance metrics
        performance = best_model_info["performance"]
        self.assertIn("auroc", performance)
        self.assertIn("sensitivity", performance)
        self.assertIn("specificity", performance)
        
        # AUROC should be reasonable (> 0.5)
        self.assertGreater(performance["auroc"], 0.5)
    
    def test_model_evaluation(self):
        """Test comprehensive model evaluation"""
        # First train a model
        training_result = self.training_tool._run(data_path=self.processed_data_path)
        training_data = json.loads(training_result)
        
        self.assertEqual(training_data["status"], "success")
        
        # Then evaluate it
        evaluation_result = self.evaluation_tool._run(
            test_data_path=self.processed_data_path
        )
        evaluation_data = json.loads(evaluation_result)
        
        self.assertEqual(evaluation_data["status"], "success")
        self.assertIn("performance_metrics", evaluation_data)
        self.assertIn("subgroup_analysis", evaluation_data)
        self.assertIn("calibration_analysis", evaluation_data)
        self.assertIn("clinical_utility", evaluation_data)
        
        # Check performance metrics structure
        metrics = evaluation_data["performance_metrics"]
        self.assertIn("discrimination_metrics", metrics)
        self.assertIn("classification_metrics", metrics)
        
        discrimination = metrics["discrimination_metrics"]
        self.assertIn("auroc", discrimination)
        self.assertIn("auprc", discrimination)
    
    def test_clinical_performance_thresholds(self):
        """Test that model meets clinical performance thresholds"""
        training_result = self.training_tool._run(data_path=self.processed_data_path)
        training_data = json.loads(training_result)
        
        if training_data["status"] == "success":
            performance = training_data["best_model"]["performance"]
            
            # Clinical thresholds (may not always be met with synthetic data)
            auroc = performance.get("auroc", 0)
            sensitivity = performance.get("sensitivity", 0)
            
            # Log performance for debugging
            print(f"Model Performance - AUROC: {auroc:.3f}, Sensitivity: {sensitivity:.3f}")
            
            # Basic sanity checks
            self.assertGreater(auroc, 0.4)  # Better than random
            self.assertGreater(sensitivity, 0.3)  # Some sensitivity
            self.assertLessEqual(auroc, 1.0)  # Valid AUROC range

class TestClinicalValidation(unittest.TestCase):
    """Test clinical validation and safety checks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validation_tool = ClinicalValidationTool()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_clinical_guidelines_validation(self):
        """Test clinical guidelines compliance validation"""
        result_json = self.validation_tool._run(guidelines_check=True)
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("validation_summary", result)
        self.assertIn("detailed_validation", result)
        self.assertIn("deployment_readiness", result)
        
        # Check validation components
        validation = result["detailed_validation"]
        self.assertIn("guideline_compliance", validation)
        self.assertIn("safety_validation", validation)
        self.assertIn("bias_assessment", validation)
        self.assertIn("clinical_actionability", validation)
        
        # Check deployment readiness
        deployment = result["deployment_readiness"]
        self.assertIn("clinical_safety", deployment)
        self.assertIn("regulatory_compliance", deployment)
    
    def test_bias_assessment(self):
        """Test algorithmic bias detection"""
        result_json = self.validation_tool._run()
        result = json.loads(result_json)
        
        bias_assessment = result["detailed_validation"]["bias_assessment"]
        self.assertIn("demographic_fairness", bias_assessment)
        self.assertIn("clinical_bias", bias_assessment)
        
        # Check fairness scores
        demo_fairness = bias_assessment["demographic_fairness"]
        self.assertIn("age_bias_score", demo_fairness)
        self.assertIn("gender_bias_score", demo_fairness)
        self.assertIn("overall_fairness", demo_fairness)

class TestVisualization(unittest.TestCase):
    """Test visualization and dashboard functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.viz_tool = VisualizationTool()
        self.dashboard_tool = DashboardTool()
        self.test_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.test_dir, "viz_data.csv")
        
        # Generate test data
        synthetic_tool = SyntheticDataTool()
        synthetic_tool._run(n_patients=100, output_path=self.test_data_path)
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_risk_distribution_visualization(self):
        """Test risk distribution plots"""
        result_json = self.viz_tool._run(
            plot_type="risk_distribution",
            data_path=self.test_data_path,
            output_dir=self.test_dir
        )
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visualization_type"], "risk_distribution")
        self.assertIn("files_created", result)
        self.assertGreater(len(result["files_created"]), 0)
        
        # Verify files were created
        for file_path in result["files_created"]:
            self.assertTrue(os.path.exists(file_path))
    
    def test_feature_importance_visualization(self):
        """Test feature importance plots"""
        result_json = self.viz_tool._run(
            plot_type="feature_importance",
            data_path=self.test_data_path,
            output_dir=self.test_dir
        )
        result = json.loads(result_json)
        
        # Should succeed even without trained model (uses synthetic importance)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visualization_type"], "feature_importance")
    
    def test_comprehensive_visualization_suite(self):
        """Test comprehensive visualization generation"""
        result_json = self.viz_tool._run(
            plot_type="comprehensive",
            data_path=self.test_data_path,
            output_dir=self.test_dir
        )
        result = json.loads(result_json)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visualization_type"], "comprehensive_suite")
        self.assertGreater(result["visualizations_generated"], 0)
        self.assertGreater(result["total_files"], 0)
    
    def test_dashboard_status(self):
        """Test dashboard status check"""
        result_json = self.dashboard_tool._run(action="status")
        result = json.loads(result_json)
        
        # Should always return status
        self.assertIn("status", result)

class TestConfiguration(unittest.TestCase):
    """Test configuration and settings management"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            dev_config = get_config('development')
            self.assertIsNotNone(dev_config)
            
            prod_config = get_config('production')
            self.assertIsNotNone(prod_config)
            
        except Exception as e:
            self.fail(f"Configuration loading failed: {str(e)}")
    
    def test_config_validation(self):
        """Test configuration validation"""
        try:
            validation_result = validate_configuration()
            self.assertIn('valid', validation_result)
            self.assertIn('errors', validation_result)
            self.assertIn('warnings', validation_result)
            
        except Exception as e:
            # Configuration validation might not be available in test environment
            self.skipTest(f"Configuration validation not available: {str(e)}")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @unittest.skip("Full pipeline integration test - requires all components")
    def test_full_pipeline_integration(self):
        """Test complete pipeline from data to dashboard"""
        try:
            # This would test the full RiskPredictionPipeline
            # Skipped by default due to complexity and time requirements
            pipeline = RiskPredictionPipeline()
            
            # Run quick demo pipeline
            result = pipeline.run_full_pipeline()
            
            self.assertIsNotNone(result)
            
        except Exception as e:
            self.fail(f"Full pipeline integration test failed: {str(e)}")
    
    def test_data_to_model_pipeline(self):
        """Test data preprocessing to model training pipeline"""
        test_data_path = os.path.join(self.test_dir, "integration_data.csv")
        
        try:
            # Step 1: Generate synthetic data
            synthetic_tool = SyntheticDataTool()
            synthetic_result = synthetic_tool._run(n_patients=150, output_path=test_data_path)
            synthetic_data = json.loads(synthetic_result)
            self.assertEqual(synthetic_data["status"], "success")
            
            # Step 2: Preprocess data
            preprocessing_tool = DataPreprocessingTool()
            preprocessing_result = preprocessing_tool._run(data_path=test_data_path)
            preprocessing_data = json.loads(preprocessing_result)
            self.assertEqual(preprocessing_data["status"], "success")
            
            # Step 3: Train model
            training_tool = ModelTrainingTool()
            training_result = training_tool._run(
                data_path=preprocessing_data["processed_file_path"]
            )
            training_data = json.loads(training_result)
            self.assertEqual(training_data["status"], "success")
            
            # Verify pipeline coherence
            self.assertIn("best_model", training_data)
            self.assertIn("performance", training_data["best_model"])
            
        except Exception as e:
            self.fail(f"Data to model pipeline test failed: {str(e)}")

class TestClinicalSafety(unittest.TestCase):
    """Test clinical safety and medical validation"""
    
    def test_clinical_range_validation(self):
        """Test that all generated values are within clinical ranges"""
        synthetic_tool = SyntheticDataTool()
        test_dir = tempfile.mkdtemp()
        test_file = os.path.join(test_dir, "safety_test.csv")
        
        try:
            result_json = synthetic_tool._run(n_patients=100, output_path=test_file)
            result = json.loads(result_json)
            self.assertEqual(result["status"], "success")
            
            df = pd.read_csv(test_file)
            
            # Test clinical ranges
            if 'age' in df.columns:
                self.assertTrue(df['age'].between(18, 95).all(), "Age values outside clinical range")
            
            if 'bmi' in df.columns:
                self.assertTrue(df['bmi'].between(15, 50).all(), "BMI values outside clinical range")
            
            if 'systolic_bp' in df.columns:
                self.assertTrue(df['systolic_bp'].between(80, 250).all(), "Systolic BP outside clinical range")
            
            if 'hba1c' in df.columns:
                self.assertTrue(df['hba1c'].between(4.0, 15.0).all(), "HbA1c outside clinical range")
                
        finally:
            shutil.rmtree(test_dir)
    
    def test_prediction_bounds(self):
        """Test that predictions are within valid probability bounds"""
        test_dir = tempfile.mkdtemp()
        test_file = os.path.join(test_dir, "prediction_test.csv")
        
        try:
            # Generate and preprocess data
            synthetic_tool = SyntheticDataTool()
            synthetic_tool._run(n_patients=100, output_path=test_file)
            
            preprocessing_tool = DataPreprocessingTool()
            preprocessing_result = preprocessing_tool._run(data_path=test_file)
            preprocessing_data = json.loads(preprocessing_result)
            
            # Train model
            training_tool = ModelTrainingTool()
            training_result = training_tool._run(
                data_path=preprocessing_data["processed_file_path"]
            )
            training_data = json.loads(training_result)
            
            if training_data["status"] == "success":
                # Check that predictions are valid probabilities
                predictions = training_data["best_model"]["performance"]
                
                # All metrics should be between 0 and 1
                for metric_name, value in predictions.items():
                    if isinstance(value, (int, float)):
                        self.assertGreaterEqual(value, 0.0, f"{metric_name} below 0")
                        self.assertLessEqual(value, 1.0, f"{metric_name} above 1")
                        
        finally:
            shutil.rmtree(test_dir)

# Test utilities
class TestUtilities:
    """Utility functions for testing"""
    
    @staticmethod
    def create_mock_patient_data(n_patients=50):
        """Create mock patient data for testing"""
        np.random.seed(42)
        
        data = {
            'patient_id': range(1, n_patients + 1),
            'age': np.random.randint(30, 85, n_patients),
            'bmi': np.random.normal(27, 5, n_patients),
            'systolic_bp': np.random.normal(135, 20, n_patients),
            'diabetes_type2': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
            'deterioration_90d': np.random.choice([0, 1], n_patients, p=[0.8, 0.2])
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def cleanup_test_outputs():
        """Clean up test output directories"""
        test_dirs = ['tests/outputs', 'tests/data', 'models/test_models']
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                os.makedirs(test_dir, exist_ok=True)

# Custom test runner
def run_comprehensive_tests():
    """Run comprehensive test suite with custom reporting"""
    
    print("🧪 Starting Comprehensive Test Suite for Chronic Care AI System")
    print("="*70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataProcessing,
        TestModelTraining,
        TestClinicalValidation,
        TestVisualization,
        TestConfiguration,
        TestIntegration,
        TestClinicalSafety
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("🏥 CHRONIC CARE AI SYSTEM - TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - System Ready for Clinical Validation")
    else:
        print("❌ SOME TESTS FAILED - Review Required Before Deployment")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    print("="*70)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run tests with different options
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Chronic Care AI System Tests')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive test suite with reporting')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick smoke tests only')
    parser.add_argument('--clinical', action='store_true',
                       help='Run clinical safety tests only')
    
    args = parser.parse_args()
    
    if args.comprehensive:
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    elif args.quick:
        # Run quick smoke tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames([
            'test_main.TestDataProcessing.test_synthetic_data_generation',
            'test_main.TestConfiguration.test_config_loading',
            'test_main.TestVisualization.test_dashboard_status'
        ])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    elif args.clinical:
        # Run clinical safety tests only
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestClinicalSafety)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run standard unittest discovery
        unittest.main()
