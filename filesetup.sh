# Create main project directory
mkdir -p chronic_care_risk_prediction
cd chronic_care_risk_prediction

# Create main directories (including scripts)
mkdir -p {src,data,models,outputs,config,tools,agents,tasks,dashboard,tests,logs,docs,scripts}

# Create subdirectories
mkdir -p data/{raw,processed,synthetic}
mkdir -p models/{trained,evaluation,checkpoints}
mkdir -p outputs/{predictions,reports,visualizations,datasets}
mkdir -p src/{core,utils,preprocessing}
mkdir -p agents/{specialists,evaluators}
mkdir -p tasks/{generation,training,evaluation,explanation}
mkdir -p dashboard/{static,templates,components}
mkdir -p tools/{data_tools,ml_tools,health_tools}
mkdir -p tests/{unit,integration,model_tests}
mkdir -p docs/{api,user_guide,technical}

# Create main Python files
touch src/main.py
touch src/config.py
touch src/crew_setup.py

# Create agent files
touch agents/__init__.py
touch agents/data_generator_agent.py
touch agents/model_trainer_agent.py
touch agents/model_evaluator_agent.py
touch agents/explainability_agent.py
touch agents/clinical_validator_agent.py
touch agents/risk_assessor_agent.py

# Create specialist agent files
touch agents/specialists/__init__.py
touch agents/specialists/diabetes_specialist.py
touch agents/specialists/cardiology_specialist.py
touch agents/specialists/obesity_specialist.py

# Create evaluator agent files
touch agents/evaluators/__init__.py
touch agents/evaluators/performance_evaluator.py
touch agents/evaluators/bias_detector.py
touch agents/evaluators/clinical_safety_evaluator.py

# Create task files
touch tasks/__init__.py
touch tasks/generation/__init__.py
touch tasks/generation/synthetic_data_task.py
touch tasks/generation/feature_engineering_task.py
touch tasks/training/__init__.py
touch tasks/training/model_training_task.py
touch tasks/training/hyperparameter_tuning_task.py
touch tasks/evaluation/__init__.py
touch tasks/evaluation/model_evaluation_task.py
touch tasks/evaluation/clinical_validation_task.py
touch tasks/explanation/__init__.py
touch tasks/explanation/global_explanation_task.py
touch tasks/explanation/local_explanation_task.py

# Create tool files
touch tools/__init__.py
touch tools/data_tools/__init__.py
touch tools/data_tools/patient_data_generator.py
touch tools/data_tools/vital_simulator.py
touch tools/data_tools/lab_result_simulator.py
touch tools/data_tools/medication_adherence_tool.py
touch tools/ml_tools/__init__.py
touch tools/ml_tools/risk_prediction_model.py
touch tools/ml_tools/feature_selector.py
touch tools/ml_tools/model_evaluator.py
touch tools/health_tools/__init__.py
touch tools/health_tools/clinical_calculator.py
touch tools/health_tools/risk_scorer.py
touch tools/health_tools/deterioration_detector.py

# Create core source files
touch src/core/__init__.py
touch src/core/data_models.py
touch src/core/patient_schema.py
touch src/core/risk_calculator.py
touch src/utils/__init__.py
touch src/utils/logging_setup.py
touch src/utils/file_manager.py
touch src/utils/model_utils.py
touch src/preprocessing/__init__.py
touch src/preprocessing/data_cleaner.py
touch src/preprocessing/feature_engineer.py
touch src/preprocessing/data_validator.py

# Create dashboard files
touch dashboard/__init__.py
touch dashboard/app.py
touch dashboard/cohort_view.py
touch dashboard/patient_detail_view.py
touch dashboard/risk_dashboard.py
touch dashboard/components/__init__.py
touch dashboard/components/charts.py
touch dashboard/components/tables.py
touch dashboard/components/alerts.py
touch dashboard/static/style.css
touch dashboard/templates/index.html
touch dashboard/templates/cohort.html
touch dashboard/templates/patient_detail.html

# Create configuration files
touch config/__init__.py
touch config/model_config.yaml
touch config/data_config.yaml
touch config/agent_config.yaml
touch config/dashboard_config.yaml

# Create test files
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/unit/test_agents.py
touch tests/unit/test_tools.py
touch tests/unit/test_models.py
touch tests/integration/__init__.py
touch tests/integration/test_workflow.py
touch tests/integration/test_dashboard.py
touch tests/model_tests/__init__.py
touch tests/model_tests/test_prediction_accuracy.py
touch tests/model_tests/test_model_fairness.py

# Create documentation files
touch docs/README.md
touch docs/api/agents_api.md
touch docs/api/tools_api.md
touch docs/user_guide/setup_guide.md
touch docs/user_guide/usage_guide.md
touch docs/technical/architecture.md
touch docs/technical/data_schema.md

# Create root level files
touch requirements.txt
touch setup.py
touch README.md
touch .gitignore
touch .env.example
touch docker-compose.yml
touch Dockerfile

# Create sample data files
touch data/raw/sample_patient_data.json
touch data/raw/clinical_guidelines.json
touch data/processed/.gitkeep
touch data/synthetic/.gitkeep

# Create model directories with placeholder files
touch models/trained/.gitkeep
touch models/evaluation/.gitkeep
touch models/checkpoints/.gitkeep

# Create output directories with placeholder files
touch outputs/predictions/.gitkeep
touch outputs/reports/.gitkeep
touch outputs/visualizations/.gitkeep
touch outputs/datasets/.gitkeep

# Create logs directory
touch logs/.gitkeep

# Create script files (AFTER creating scripts directory)
touch scripts/run_data_generation.sh
touch scripts/run_model_training.sh
touch scripts/run_evaluation.sh
touch scripts/deploy_dashboard.sh

# Make executable scripts
chmod +x scripts/*.sh

echo "File structure created successfully!"
echo "Total directories created: $(find . -type d | wc -l)"
echo "Total files created: $(find . -type f | wc -l)"
