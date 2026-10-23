#!/bin/bash
# BRISC Web App - Startup Script

set -e

WEBAPP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$WEBAPP_DIR/backend"
FRONTEND_DIR="$WEBAPP_DIR/frontend"
MODEL_DIR="$WEBAPP_DIR/backend/models"

echo "🧠 BRISC Brain Tumor Analysis Web App"
echo "====================================="

# Check models exist
echo "📦 Checking model checkpoints..."
if [ ! -f "$MODEL_DIR/mambaout_brisc.pth" ]; then
    echo "⚠️  Warning: mambaout_brisc.pth not found in $MODEL_DIR"
    echo "   Please copy your trained classification checkpoint there"
fi

if [ ! -f "$MODEL_DIR/bgsf_brisc.pth" ]; then
    echo "⚠️  Warning: bgsf_brisc.pth not found in $MODEL_DIR"
    echo "   Please copy your trained segmentation checkpoint there"
fi

# ─── BACKEND ───
echo ""
echo "🚀 Starting FastAPI backend on http://localhost:8000"
cd "$BACKEND_DIR"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3.11 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt
# Install torch from PyTorch index (handles Python 3.12+ compatibility)
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set model directory
export BRISC_MODEL_DIR="$MODEL_DIR"

# Run backend in background
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "✓ Backend PID: $BACKEND_PID"

# ─── FRONTEND ───
echo ""
echo "🎨 Starting React frontend on http://localhost:3000"
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "✓ Frontend PID: $FRONTEND_PID"

# ─── CLEANUP ───
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

echo ""
echo "✅ BRISC Web App is running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"

wait