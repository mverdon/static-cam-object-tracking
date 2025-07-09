@echo off
REM Setup script for Static Camera Object Tracking (Windows)
REM This script sets up the Python environment and installs all dependencies

echo 🎯 Setting up Static Camera Object Tracking
echo ==========================================

REM Check if uv is installed
uv --version >nul 2>&1
if errorlevel 1 (
    echo ❌ uv is not installed. Please install it first:
    echo    curl -LsSf https://astral.sh/uv/install.sh ^| sh
    echo    Then restart your terminal and run this script again.
    pause
    exit /b 1
)

echo ✅ uv found
uv --version

REM Create virtual environment with Python 3.12
echo 🐍 Creating virtual environment with Python 3.12...
uv venv --python 3.12

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install PyTorch with CUDA support first
echo 🔥 Installing PyTorch with CUDA support...
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Install project dependencies
echo 📦 Installing project dependencies...
uv sync

REM Install TensorRT if the wheel exists
set TENSORRT_WHEEL=C:\TensorRT-10.12.0.36\python\tensorrt_lean-10.12.0.36-cp312-none-win_amd64.whl
if exist "%TENSORRT_WHEEL%" (
    echo 🚀 Installing TensorRT...
    uv pip install "%TENSORRT_WHEEL%"
    echo ✅ TensorRT installed successfully
) else (
    echo ⚠️  TensorRT wheel not found at %TENSORRT_WHEEL%
    echo    TensorRT acceleration will not be available
    echo    You can install it later if the wheel becomes available
)

REM Verify installation
echo 🔍 Verifying installation...
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "try: import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); except: print('PyTorch error')"
python -c "try: import cv2; print(f'OpenCV: {cv2.__version__}'); except: print('OpenCV error')"
python -c "try: from ultralytics import YOLO; print('✅ YOLO available'); except: print('YOLO error')"
python -c "try: import tensorrt as trt; print(f'TensorRT: {trt.__version__}'); except: print('⚠️  TensorRT not available')"

echo.
echo 🎉 Setup completed successfully!
echo.
echo 📋 Next steps:
echo 1. Place your video files in the 'videos/' directory
echo 2. Run the demo: python demo.py
echo 3. Process a video: python src/main.py --input videos/your_video.mp4 --display
echo.
echo 💡 Tips:
echo - Use --list-videos to see available videos
echo - Use --help to see all available options
echo - The first run will download the YOLO model (~6MB)
echo.
echo 🔧 Environment activation:
echo    .venv\Scripts\activate.bat

pause
