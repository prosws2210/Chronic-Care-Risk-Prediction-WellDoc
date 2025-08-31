#!/bin/bash

# Navigate to the directory containing this script
cd "$(dirname "$0")"

# Create main directories (including scripts)
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

# Create placeholder README files for clarity
touch src/__init__.py
touch README.md

echo "✅ Project structure created successfully in $(pwd)"
echo "Directories and subdirectories have been set up."