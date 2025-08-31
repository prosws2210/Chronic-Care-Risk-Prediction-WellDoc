#!/usr/bin/env python3
"""
Main execution script for the Chronic Care AI Risk Prediction Pipeline
=====================================================================

This script provides multiple execution modes for the chronic care risk prediction system.
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def run_full_pipeline(data_path=None, demo=False):
    """Run the complete AI pipeline"""
    try:
        from src.main import RiskPredictionPipeline
        
        print("🏥 Starting Chronic Care AI Risk Prediction Pipeline...")
        
        pipeline = RiskPredictionPipeline()
        
        if demo:
            print("🎯 Running in demo mode with synthetic data...")
            results = pipeline.run_full_pipeline()
        else:
            data_file = data_path or "data/processed/chronic_care_data.csv"
            print(f"📊 Processing data from: {data_file}")
            results = pipeline.run_full_pipeline(data_file)
        
        if results:
            print("✅ Pipeline completed successfully!")
            print("📋 Check outputs/reports/ for detailed results")
            print("🖥️  Launch dashboard with: streamlit run dashboard/app.py")
            return 0
        else:
            print("❌ Pipeline failed. Check logs for details.")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python scripts/setup.py")
        return 1
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return 1

def run_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        import subprocess
        print("🚀 Launching Chronic Care Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])
        return 0
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")
        return 1

def run_tests(test_type="all"):
    """Run the test suite"""
    try:
        import subprocess
        
        print(f"🧪 Running {test_type} tests...")
        
        if test_type == "quick":
            cmd = [sys.executable, "tests/test_main.py", "--quick"]
        elif test_type == "clinical":
            cmd = [sys.executable, "tests/test_main.py", "--clinical"]
        elif test_type == "comprehensive":
            cmd = [sys.executable, "tests/test_main.py", "--comprehensive"]
        else:
            cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
        
        result = subprocess.run(cmd)
        return result.returncode
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Chronic Care AI Risk Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run.py                          # Run full pipeline
  python scripts/run.py --demo                   # Run with demo data
  python scripts/run.py --dashboard              # Launch dashboard only
  python scripts/run.py --test quick             # Run quick tests
  python scripts/run.py --data patients.csv     # Run with specific data
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run pipeline with synthetic demo data')
    parser.add_argument('--dashboard', action='store_true',
                       help='Launch Streamlit dashboard only')
    parser.add_argument('--test', choices=['all', 'quick', 'clinical', 'comprehensive'],
                       help='Run test suite')
    parser.add_argument('--data', type=str,
                       help='Path to patient data file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Ensure required directories exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)
    
    try:
        if args.dashboard:
            return run_dashboard()
        elif args.test:
            return run_tests(args.test)
        else:
            return run_full_pipeline(args.data, args.demo)
            
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
