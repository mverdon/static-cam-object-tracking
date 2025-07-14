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
- `--track-videos`: Generate separate video files for each detected track
- `--track-crop-ratio`: Ratio to expand bounding box for track video cropping (default: 1.5)

Example with different tracker and custom FPS:
```bash
python src/main.py --input videos/your_video.mp4 --tracker botsort --fps 30 --display
```

Example with track video generation:
```bash
python src/main.py --input videos/your_video.mp4 --track-videos --track-crop-ratio 2.0
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
├── track_outputs/                   # Individual track videos (when enabled)
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

## Track Video Generation

The system can generate separate video files for each detected and tracked object, providing cropped views that follow individual objects throughout the video.

### Features

- **Individual Track Videos**: Each detected object gets its own video file with fixed dimensions
- **Smart Cropping Algorithm**: Uses object center and height with configurable padding
- **Consistent Video Dimensions**: All track videos have identical size (1280x720 by default)
- **Edge Handling**: Automatic black padding for objects near frame boundaries
- **Preserved Quality**: High-quality cropping with proper aspect ratio maintenance

### Usage

Enable track video generation with the `--track-videos` flag:

```bash
python src/main.py --input videos/your_video.mp4 --track-videos
```

Customize the crop area around objects:

```bash
python src/main.py --input videos/your_video.mp4 --track-videos --track-crop-ratio 2.0
```

### Configuration

Track video settings can be configured in `src/config.py`:

- `TRACK_VIDEO_CROP_RATIO`: Default crop expansion ratio (1.5)
- `TRACK_VIDEO_MIN_SIZE`: Minimum crop dimensions (128x72)
- `TRACK_VIDEO_MAX_SIZE`: Fixed output dimensions for all videos (1280x720)
- `TRACK_VIDEOS_DIR`: Output directory for track videos

### Output

Track videos are saved in the `track_outputs/` directory with filenames like:
- `track_0001_horse.mp4`
- `track_0002_person.mp4`
- `track_0003_horse.mp4`

Each filename includes the track ID and detected object class for easy identification.

## License

MIT License
