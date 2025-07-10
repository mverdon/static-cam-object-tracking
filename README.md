# Static Camera Object Tracking

A real-time object tracking system using YOLOv11 with built-in ByteTrack and BotsSort tracking algorithms, featuring CUDA and TensorRT acceleration for static camera setups.

## Features

- YOLOv11-based object detection with built-in tracking
- Support for ByteTrack and BotsSort tracking algorithms
- CUDA acceleration for GPU processing
- TensorRT optimization for enhanced performance
- Support for various video formats
- Configurable tracking parameters
- Output video generation with tracking visualization
- Unified entry point for detection and tracking

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

### Command Line Options

- `--tracker`: Choose tracking algorithm (`bytetrack` or `botsort`)
- `--confidence`: Set detection confidence threshold
- `--fps`: Override output video FPS (default: uses input video FPS)
- `--display`: Show video during processing
- `--no-cuda`: Disable CUDA acceleration
- `--no-tensorrt`: Disable TensorRT optimization

Example with different tracker and custom FPS:
```bash
python src/main.py --input videos/your_video.mp4 --tracker botsort --fps 30 --display
```

## Project Structure

```
static-cam-object-tracking/
├── src/
│   ├── main.py                      # Main tracking script (unified entry point)
│   ├── detector_with_tracking.py    # YOLOv11 with built-in tracking
│   ├── yolo_tracker.py              # Simplified tracker interface
│   ├── config.py                    # Configuration settings
│   └── utils.py                     # Utility functions
├── videos/                          # Input videos
├── outputs/                         # Output videos with tracking
├── models/                          # Model weights (auto-downloaded)
├── pyproject.toml                   # Project configuration
└── README.md                        # This file
```

## Configuration

The tracker can be configured by modifying parameters in `src/config.py` or passing command-line arguments to control:

- Detection confidence threshold
- Tracking algorithm (ByteTrack or BotsSort)
- Tracking algorithm parameters
- Output video quality
- Frame processing rate
- CUDA and TensorRT settings

## Tracking Algorithms

The system supports two state-of-the-art tracking algorithms:

- **ByteTrack**: Robust tracking with two-step association using high and low confidence detections
- **BotsSort**: Advanced tracking with appearance features and motion compensation

Both algorithms are built into YOLO and optimized for performance.

## License

MIT License
