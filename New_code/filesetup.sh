#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "$0")"

# Create main directories
mkdir -p {src,data,models,outputs,config,tools,agents,tasks,dashboard,tests,logs,docs,scripts}

# Create subdirectories
mkdir -p data/{raw,processed,synthetic}
mkdir -p models/{checkpoints,saved,onnx}
mkdir -p outputs/{figures,reports}
mkdir -p config/{dev,prod}
mkdir -p tools/{preprocessing,visualization,utils}
mkdir -p agents/{risk,patient,doctor}
mkdir -p tasks/{classification,risk_prediction}
mkdir -p dashboard/{frontend,backend}
mkdir -p tests/{unit,integration}
mkdir -p logs/{training,inference}
mkdir -p docs/{references,tutorials}

# Create main source files
touch src/__init__.py
touch src/main.py

# Create configuration files
touch config/settings.py
touch config/__init__.py

# Create agent files (consolidated)
touch agents/healthcare_agents.py
touch agents/__init__.py

# Create task files (consolidated)
touch tasks/prediction_tasks.py
touch tasks/__init__.py

# Create tool files (max 2)
touch tools/data_tools.py
touch tools/visualization_tools.py
touch tools/__init__.py

# Create dashboard files
touch dashboard/app.py
touch dashboard/utils.py

# Create test files
touch tests/test_main.py
touch tests/__init__.py

# Create documentation
touch docs/README.md
touch docs/API_GUIDE.md

# Create script files
touch scripts/run.py
touch scripts/setup.py

# Create placeholder files for data directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/synthetic/.gitkeep

# Create placeholder files for model directories
touch models/checkpoints/.gitkeep
touch models/saved/.gitkeep
touch models/onnx/.gitkeep

# Create placeholder files for output directories
touch outputs/figures/.gitkeep
touch outputs/reports/.gitkeep

# Create placeholder files for log directories
touch logs/training/.gitkeep
touch logs/inference/.gitkeep

# Create main project files
touch README.md
touch requirements.txt
touch .gitignore

echo "✅ Minimal project structure created successfully in $(pwd)"
echo ""
echo "📁 Project Structure:"
echo "├── src/"
echo "│   ├── __init__.py"
echo "│   └── main.py"
echo "├── config/"
echo "│   ├── __init__.py"
echo "│   └── settings.py"
echo "├── agents/"
echo "│   ├── __init__.py"
echo "│   └── healthcare_agents.py"
echo "├── tasks/"
echo "│   ├── __init__.py"
echo "│   └── prediction_tasks.py"
echo "├── tools/"
echo "│   ├── __init__.py"
echo "│   ├── data_tools.py"
echo "│   └── visualization_tools.py"
echo "├── dashboard/"
echo "│   ├── app.py"
echo "│   └── utils.py"
echo "├── tests/"
echo "│   ├── __init__.py"
echo "│   └── test_main.py"
echo "├── docs/"
echo "│   ├── README.md"
echo "│   └── API_GUIDE.md"
echo "├── scripts/"
echo "│   ├── run.py"
echo "│   └── setup.py"
echo "├── data/{raw,processed,synthetic}/"
echo "├── models/{checkpoints,saved,onnx}/"
echo "├── outputs/{figures,reports}/"
echo "├── logs/{training,inference}/"
echo "├── README.md"
echo "├── requirements.txt"
echo "└── .gitignore"
