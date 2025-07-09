#!/bin/bash

# Setup script for Static Camera Object Tracking
# This script sets up the Python environment and installs all dependencies

set -e  # Exit on any error

echo "🎯 Setting up Static Camera Object Tracking"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Then restart your terminal and run this script again."
    exit 1
fi

echo "✅ uv found: $(uv --version)"

# Create virtual environment with Python 3.12
echo "🐍 Creating virtual environment with Python 3.12..."
uv venv --python 3.12

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install PyTorch with CUDA support first
echo "🔥 Installing PyTorch with CUDA support..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
echo "📦 Installing project dependencies..."
uv sync

# Install TensorRT if the wheel exists
TENSORRT_WHEEL="/c/TensorRT-10.12.0.36/python/tensorrt_lean-10.12.0.36-cp312-none-win_amd64.whl"
if [ -f "$TENSORRT_WHEEL" ]; then
    echo "🚀 Installing TensorRT..."
    uv pip install "$TENSORRT_WHEEL"
    echo "✅ TensorRT installed successfully"
else
    echo "⚠️  TensorRT wheel not found at $TENSORRT_WHEEL"
    echo "   TensorRT acceleration will not be available"
    echo "   You can install it later if the wheel becomes available"
fi

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
except ImportError as e:
    print(f'PyTorch import error: {e}')

try:
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
except ImportError as e:
    print(f'OpenCV import error: {e}')

try:
    from ultralytics import YOLO
    print('✅ Ultralytics YOLO available')
except ImportError as e:
    print(f'Ultralytics import error: {e}')

try:
    import tensorrt as trt
    print(f'TensorRT version: {trt.__version__}')
except ImportError:
    print('⚠️  TensorRT not available')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Place your video files in the 'videos/' directory"
echo "2. Run the demo: python demo.py"
echo "3. Process a video: python src/main.py --input videos/your_video.mp4 --display"
echo ""
echo "💡 Tips:"
echo "- Use --list-videos to see available videos"
echo "- Use --help to see all available options"
echo "- The first run will download the YOLO model (~6MB)"
echo ""
echo "🔧 Environment activation:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source .venv/Scripts/activate"
else
    echo "   source .venv/bin/activate"
fi
