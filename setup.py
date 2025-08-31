from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chronic-care-risk-prediction",
    version="1.0.0",
    author="Healthcare AI Team",
    author_email="team@chroniccare-ai.com",
    description="AI-Driven Risk Prediction Engine for Chronic Care Patients",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/prosws2210/Chronic-Care-Risk-Prediction-WellDoc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chronic-care-predict=src.main:main",
            "chronic-care-dashboard=dashboard.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
)
