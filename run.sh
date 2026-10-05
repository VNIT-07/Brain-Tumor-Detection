#!/usr/bin/env bash
# ==============================================================================
# NeuroScan AI - Launcher Script (macOS / Linux)
# Automatically sets up the virtual environment, verifies dependencies,
# and starts the Streamlit application.
# ==============================================================================

set -e

# Change directory to the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🧠 Starting NeuroScan AI (YOLOv8 Brain Tumor Detection)"
echo "============================================================"

# Check if Python 3 is available
if ! command -v python3 &>/dev/null; then
    echo "❌ Error: python3 is not installed or not in your PATH."
    echo "Please install Python 3.9 - 3.11 and try again."
    exit 1
fi

# Setup Virtual Environment if missing
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    python3 -m venv .venv
    echo "📦 Installing required dependencies from requirements.txt..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Ensure Ultralytics and Streamlit local configs
mkdir -p .ultralytics
export YOLO_CONFIG_DIR="$SCRIPT_DIR/.ultralytics"
export STREAMLIT_CREDENTIALS_FILE="$SCRIPT_DIR/.streamlit/credentials.toml"
export STREAMLIT_CONFIG_FILE="$SCRIPT_DIR/.streamlit/config.toml"

# Verify model checkpoint exists
if [ ! -f "best.pt" ]; then
    echo "⚠️ Warning: 'best.pt' not found in project root."
fi

echo "🚀 Launching Streamlit interface..."
echo "Web app will open in your default browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server."
echo "============================================================"

.venv/bin/streamlit run app.py
