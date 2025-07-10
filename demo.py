"""
Simple demo script to test the    print("📋 Other useful commands:")
    print("   python src/main.py --list-videos    # List all available videos")
    print("   python src/main.py --help           # Show all options")
    print("   python src/main.py --tracker botsort # Use BotsSort instead of ByteTrack")

    print(f"\n📊 Current configuration:")
    print(f"   Model: {config.YOLO_MODEL}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"   Tracker: {config.TRACKER_TYPE}")
    print(f"   CUDA enabled: {config.USE_CUDA}")
    print(f"   TensorRT enabled: {config.USE_TENSORRT}")racking system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import config
from src.utils import get_available_video_files

def main():
    """Run a simple demo."""
    print("🎯 Static Camera Object Tracking Demo")
    print("=" * 40)

    # Check for available videos
    videos = get_available_video_files(str(config.VIDEOS_DIR))

    if not videos:
        print(f"❌ No videos found in {config.VIDEOS_DIR}")
        print("Please add some video files to the videos/ folder and try again.")
        return

    print(f"📁 Found {len(videos)} video(s) in {config.VIDEOS_DIR}:")
    for i, video in enumerate(videos, 1):
        video_name = Path(video).name
        print(f"   {i}. {video_name}")

    print("\n🚀 To run tracking on a video, use:")
    print(f"   python src/main.py --input \"{videos[0]}\" --display")

    print("\n📋 Other useful commands:")
    print("   python src/main.py --list-videos    # List all available videos")
    print("   python src/main.py --help           # Show all options")
    print("   python src/main.py --tracker botsort # Use BotsSort instead of ByteTrack")

    print(f"\n📊 Current configuration:")
    print(f"   Model: {config.YOLO_MODEL}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"   Tracker: {config.TRACKER_TYPE}")
    print(f"   CUDA enabled: {config.USE_CUDA}")
    print(f"   TensorRT enabled: {config.USE_TENSORRT}")


if __name__ == "__main__":
    main()
