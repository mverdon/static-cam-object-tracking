# Static Camera Object Tracking

A real-time object tracking system using YOLOv11 with CUDA and TensorRT acceleration for static camera setups.

## Features

- YOLOv11-based object detection and tracking
- CUDA acceleration for GPU processing
- TensorRT optimization for enhanced performance
- Support for various video formats
- Configurable tracking parameters
- Output video generation with tracking visualization

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support
- TensorRT 10.12.0.36 (Windows)
- uv package manager

## Installation

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:
```bash
uv venv --python 3.12
source .venv/Scripts/activate  # Windows
```

3. Install dependencies:
```bash
uv sync
```

4. Install TensorRT manually:
```bash
uv pip install TensorRT-10.12.0.36/python/tensorrt_lean-10.12.0.36-cp312-none-win_amd64.whl
```

## Usage

1. Place your input video in the `videos/` folder
2. Run the tracking script:
```bash
python src/main.py --input videos/your_video.mp4 --output outputs/tracked_video.mp4
```

## Project Structure

```
static-cam-object-tracking/
├── src/
│   ├── main.py              # Main tracking script
│   ├── tracker.py           # Object tracking logic
│   ├── detector.py          # YOLOv11 detection wrapper
│   └── utils.py             # Utility functions
├── videos/                  # Input videos
├── outputs/                 # Output videos with tracking
├── models/                  # Model weights (auto-downloaded)
├── pyproject.toml           # Project configuration
└── README.md               # This file
```

## Configuration

The tracker can be configured by modifying parameters in `src/config.py` or passing command-line arguments to control:

- Detection confidence threshold
- Tracking algorithm parameters
- Output video quality
- Frame processing rate

## License

MIT License
