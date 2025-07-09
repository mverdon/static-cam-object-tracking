#!/usr/bin/env python3
"""
System verification script for Static Camera Object Tracking
"""

import sys
import platform
from pathlib import Path


def check_system():
    """Check system requirements and installation."""
    print("🔍 System Verification")
    print("=" * 50)

    # Python version
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

    # Core dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("ultralytics", "YOLOv11"),
        ("numpy", "NumPy"),
        ("tensorrt", "TensorRT")
    ]

    for module, name in dependencies:
        try:
            if module == "cv2":
                import cv2
                print(f"✅ {name}: {cv2.__version__}")
            elif module == "torch":
                import torch
                print(f"✅ {name}: {torch.__version__}")
                print(f"   CUDA Available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    print(f"   CUDA Version: {torch.version.cuda}")
                    print(f"   GPU: {torch.cuda.get_device_name(0)}")
            elif module == "ultralytics":
                from ultralytics import YOLO
                print(f"✅ {name}: Available")
            elif module == "numpy":
                import numpy as np
                print(f"✅ {name}: {np.__version__}")
            elif module == "tensorrt":
                import tensorrt as trt
                print(f"✅ {name}: {trt.__version__}")
        except ImportError:
            print(f"❌ {name}: Not available")
        except Exception as e:
            print(f"⚠️  {name}: Error - {e}")

    print()

    # Directory structure
    print("📁 Project Structure:")
    project_root = Path(__file__).parent
    dirs = ["src", "videos", "outputs", "models"]

    for dir_name in dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            count = len(list(dir_path.iterdir()))
            print(f"✅ {dir_name}/: {count} items")
        else:
            print(f"❌ {dir_name}/: Missing")

    print()

    # Available videos
    videos_dir = project_root / "videos"
    if videos_dir.exists():
        video_files = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

        for file_path in videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                video_files.append(file_path.name)

        if video_files:
            print(f"🎥 Available Videos ({len(video_files)}):")
            for video in video_files[:5]:  # Show first 5
                print(f"   - {video}")
            if len(video_files) > 5:
                print(f"   ... and {len(video_files) - 5} more")
        else:
            print("📹 No video files found in videos/ directory")
            print("   Add some video files to test the tracking system")

    print()

    # Ready status
    print("🚀 System Status:")
    try:
        import torch, cv2, tensorrt
        from ultralytics import YOLO

        if torch.cuda.is_available():
            print("✅ Ready for GPU-accelerated object tracking!")
            print("✅ CUDA and TensorRT optimization available")
        else:
            print("✅ Ready for CPU-based object tracking")
            print("⚠️  No GPU detected - processing will be slower")

        print()
        print("💡 Quick Start:")
        print("   python demo.py                    # Check installation")
        print("   python src/main.py --list-videos  # List available videos")
        print("   python src/main.py --input videos/your_video.mp4 --display")

    except ImportError as e:
        print(f"❌ System not ready: {e}")
        print("   Run: uv sync")
        print("   Or: pip install -r requirements.txt")


if __name__ == "__main__":
    check_system()
