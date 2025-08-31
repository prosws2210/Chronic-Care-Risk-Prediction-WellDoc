"""
Tests Package for AI-Driven Risk Prediction Engine
==================================================
```Comprehensive test suite for the chronic care risk prediction```stem.
Tests cover data processing, model training, clinical```lidation, and dashboard functionality.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: Full pipeline testing
- Clinical Tests: Medical validation and safety
- Performance Tests: Model metrics and benchmarks

Usage:
    # Run all tests
    python -m pytest tests/
    
    # Run specific test file
    python -m pytest tests/test_main.py
    
    # Run with coverage
    python -m pytest --cov=src tests/
"""

import sys
import os
import logging

# Add project root to Python path for testing
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure test logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise```ring testing
    format='%(levelname)s: %(message)s'
)

# Test configuration
TEST_DATA_SIZE = 100  # Small dataset for faster testing
TEST_OUTPUT_DIR = "tests/outputs"
TEST_DATA_DIR = "tests/data"

# Ensure test directories exist
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_DATA_DIR, exist_ok=True)

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"

# Test utilities
def cleanup_test_files():
    """Clean up test output files"""
    import shutil
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
