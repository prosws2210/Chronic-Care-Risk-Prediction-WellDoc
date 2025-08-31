#!/usr/bin/env python3
"""
Setup and installation script for Chronic Care AI Risk Prediction Engine
========================================================================

This script handles environment setup, dependency installation, and system validation.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🏥 Chronic Care AI Risk Prediction Engine - Setup")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print("✅ Python version compatible")
    return True

def install_requirements():
    """Install package requirements"""
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False
    
    print("📦 Installing package requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary project directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/synthetic",
        "models/saved",
        "models/checkpoints",
        "outputs/reports",
        "outputs/figures",
        "logs/training",
        "logs/inference"
    ]
    
    print("📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ Directories created successfully")
    return True

def validate_installation():
    """Validate that key packages are installed"""
    print("🔍 Validating installation...")
    
    required_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "streamlit",
        "crewai"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are available")
    return True

def setup_git_hooks():
    """Setup git hooks for development"""
    if not os.path.exists('.git'):
        print("ℹ️  Not a git repository, skipping git hooks setup")
        return True
    
    print("🔗 Setting up git hooks...")
    
    # Create pre-commit hook
    hook_content = """#!/bin/bash
# Pre-commit hook for chronic care AI project
echo "Running pre-commit checks..."

# Run quick tests
python tests/test_main.py --quick
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Commit aborted."
    exit 1
fi

echo "✅ Pre-commit checks passed"
"""
    
    hooks_dir = ".git/hooks"
    if os.path.exists(hooks_dir):
        hook_file = os.path.join(hooks_dir, "pre-commit")
        with open(hook_file, 'w') as f:
            f.write(hook_content)
        os.chmod(hook_file, 0o755)
        print("  ✓ Pre-commit hook installed")
    
    return True

def run_initial_tests():
    """Run initial system tests"""
    print("🧪 Running initial system tests...")
    
    try:
        # Test data generation
        from tools.data_tools import SyntheticDataTool
        
        tool = SyntheticDataTool()
        result = tool._run(n_patients=10, output_path="data/test_patients.csv")
        
        if "success" in result:
            print("  ✓ Data generation test passed")
            # Clean up test file
            if os.path.exists("data/test_patients.csv"):
                os.remove("data/test_patients.csv")
        else:
            print("  ❌ Data generation test failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Initial tests failed: {e}")
        return False
    
    print("✅ Initial tests passed")
    return True

def print_next_steps():
    """Print next steps for users"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Run the pipeline:")
    print("   python scripts/run.py --demo")
    print("\n2. Launch the dashboard:")
    print("   python scripts/run.py --dashboard")
    print("\n3. Run tests:")
    print("   python scripts/run.py --test quick")
    print("\n4. View documentation:")
    print("   open docs/README.md")
    print("\n💡 For help: python scripts/run.py --help")

def main():
    """Main setup function"""
    print_header()
    
    # Check system compatibility
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Create directories
    if not create_directories():
        return 1
    
    # Validate installation
    if not validate_installation():
        return 1
    
    # Setup development tools
    setup_git_hooks()
    
    # Run initial tests
    if not run_initial_tests():
        print("⚠️  Initial tests failed, but setup is complete")
        print("💡 Try running: python scripts/run.py --test quick")
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
    