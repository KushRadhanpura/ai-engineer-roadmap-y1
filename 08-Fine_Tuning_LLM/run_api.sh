#!/bin/bash
# This script starts the FastAPI server for the fine-tuned model.

# Get the absolute path of the script's directory to ensure paths are correct
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Set the PYTHONPATH to include the project directory, so that 'app' can be imported
export PYTHONPATH=$SCRIPT_DIR

# Path to the python executable in the shared virtual environment
VENV_PYTHON="$SCRIPT_DIR/../06-The_ML_Core/ai_roadmap_env/bin/python"

echo "Starting server..."
# Run the uvicorn server
$VENV_PYTHON -m uvicorn app.main:app --host 127.0.0.1 --port 8000
