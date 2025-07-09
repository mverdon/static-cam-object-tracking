# Quick Start Guide

## 🚀 Getting Started

### 1. Setup Environment
Run the setup script to install all dependencies:

**Windows:**
```bash
setup.bat
```

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Activate Environment
```bash
# Windows
.venv\Scripts\activate.bat

# Linux/macOS
source .venv/bin/activate
```

### 3. Add Video Files
Place your video files in the `videos/` directory:
```
videos/
├── my_video.mp4
├── test_clip.avi
└── security_footage.mov
```

### 4. Run Demo
```bash
python demo.py
```

### 5. Process a Video
```bash
# Basic usage
python src/main.py --input videos/my_video.mp4

# With display window
python src/main.py --input videos/my_video.mp4 --display

# Custom output location
python src/main.py --input videos/my_video.mp4 --output outputs/result.mp4

# List available videos
python src/main.py --list-videos
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **Model**: Change YOLO model size (n/s/m/l/x)
- **Thresholds**: Adjust confidence and IoU thresholds
- **Hardware**: Enable/disable CUDA and TensorRT
- **Output**: Modify video quality and display options

## 🛠️ VS Code Tasks

Use Ctrl+Shift+P → "Tasks: Run Task" to access:

- **Setup Environment**: Install all dependencies
- **Run Demo**: Quick demo of the system
- **List Videos**: Show available video files
- **Install TensorRT**: Manual TensorRT installation

## 🔧 Troubleshooting

### CUDA Issues
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### TensorRT Issues
Install manually if the automatic installation fails:
```bash
uv pip install TensorRT-10.12.0.36/python/tensorrt_lean-10.12.0.36-cp312-none-win_amd64.whl
```

### Video Issues
- Ensure video files are in supported formats (mp4, avi, mov, mkv, etc.)
- Check video file permissions
- Verify video is not corrupted

## 📈 Performance Tips

1. **Use smaller YOLO models** (yolo11n.pt) for faster processing
2. **Enable TensorRT** for GPU acceleration
3. **Adjust frame skip** in config to process fewer frames
4. **Resize videos** to lower resolution for speed
5. **Use SSD storage** for better I/O performance

## 🎯 Example Commands

```bash
# Fast processing with nano model
python src/main.py --input videos/test.mp4 --model yolo11n.pt

# High accuracy with extra large model
python src/main.py --input videos/test.mp4 --model yolo11x.pt

# Lower confidence threshold for more detections
python src/main.py --input videos/test.mp4 --confidence 0.3

# Disable GPU acceleration
python src/main.py --input videos/test.mp4 --no-cuda --no-tensorrt
```
