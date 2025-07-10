"""
Main script for static camera object tracking.
"""

import argparse
import logging
import time
from pathlib import Path
import cv2
import numpy as np

from src.detector_with_tracking import YOLODetectorWithTracking
from src import config
from src.utils import (
    get_video_info,
    create_video_writer,
    resize_frame,
    validate_video_file,
    get_available_video_files,
    setup_logging,
    print_system_info
)

logger = logging.getLogger(__name__)


def draw_yolo_tracks(frame: np.ndarray, tracks: list, show_info: bool = True) -> np.ndarray:
    """
    Draw YOLO tracks on frame.

    Args:
        frame: Input frame
        tracks: List of track dictionaries
        show_info: Whether to show track information

    Returns:
        Frame with tracks drawn
    """
    output_frame = frame.copy()

    for track in tracks:
        track_id = track['track_id']
        class_id = track['class_id']
        confidence = track['confidence']
        bbox = track['bbox']
        class_name = track['class_name']

        # Get coordinates
        x1, y1, x2, y2 = map(int, bbox)

        # Get color
        color = config.CLASS_COLORS.get(class_id, config.DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        if show_info:
            label_parts = []

            if config.SHOW_TRACK_ID:
                label_parts.append(f"ID:{track_id}")

            if config.SHOW_CLASS_NAME:
                label_parts.append(class_name)

            if config.SHOW_CONFIDENCE:
                label_parts.append(f"{confidence:.2f}")

            label = " ".join(label_parts)

            if label:
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    output_frame,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    output_frame,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

    return output_frame


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Static Camera Object Tracking')

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input video file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output video file'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=config.YOLO_MODEL,
        help=f'YOLO model to use (default: {config.YOLO_MODEL})'
    )

    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold (default: {config.CONFIDENCE_THRESHOLD})'
    )

    parser.add_argument(
        '--iou',
        type=float,
        default=config.IOU_THRESHOLD,
        help=f'IoU threshold (default: {config.IOU_THRESHOLD})'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA acceleration'
    )

    parser.add_argument(
        '--cuda-device',
        type=int,
        default=config.CUDA_DEVICE,
        help=f'CUDA device index to use (default: {config.CUDA_DEVICE})'
    )

    parser.add_argument(
        '--no-tensorrt',
        action='store_true',
        help='Disable TensorRT optimization'
    )

    parser.add_argument(
        '--display',
        action='store_true',
        help='Display video during processing'
    )

    parser.add_argument(
        '--no-mask',
        action='store_true',
        help='Disable mask detection and application'
    )

    parser.add_argument(
        '--show-mask',
        action='store_true',
        help='Show mask overlay on output video'
    )

    parser.add_argument(
        '--specified-only',
        action='store_true',
        help='Detect and track specified objects only'
    )

    parser.add_argument(
        '--object-confidence',
        type=float,
        default=config.OBJECT_MIN_CONFIDENCE,
        help=f'Minimum confidence for object detections (default: {config.OBJECT_MIN_CONFIDENCE})'
    )

    parser.add_argument(
        '--fps',
        type=float,
        help='Override output video FPS (default: uses input video FPS)'
    )

    parser.add_argument(
        '--tracker', '-t',
        type=str,
        choices=['bytetrack', 'botsort'],
        default=config.TRACKER_TYPE,
        help=f'Tracker type to use (default: {config.TRACKER_TYPE})'
    )

    parser.add_argument(
        '--list-videos',
        action='store_true',
        help='List available videos in the videos directory'
    )

    parser.add_argument(
        '--list-gpus',
        action='store_true',
        help='List available CUDA devices and exit'
    )

    return parser.parse_args()


def list_available_videos():
    """List available video files."""
    videos = get_available_video_files(str(config.VIDEOS_DIR))

    if not videos:
        print(f"No video files found in {config.VIDEOS_DIR}")
        return

    print(f"Available videos in {config.VIDEOS_DIR}:")
    for i, video in enumerate(videos, 1):
        video_path = Path(video)
        try:
            info = get_video_info(video)
            print(f"{i:2d}. {video_path.name}")
            print(f"     Size: {info['width']}x{info['height']}")
            print(f"     FPS: {info['fps']:.2f}")
            print(f"     Duration: {info['duration']:.2f}s")
            print(f"     Frames: {info['frame_count']}")
        except Exception as e:
            print(f"{i:2d}. {video_path.name} (Error: {e})")
        print()


def process_video(input_path: str, output_path: str, args):
    """Process a video file with object tracking."""

    # Validate input
    if not validate_video_file(input_path):
        raise ValueError(f"Invalid or corrupted video file: {input_path}")

    # Get video information
    video_info = get_video_info(input_path)
    logger.info(f"Processing video: {input_path}")
    logger.info(f"Video info: {video_info}")

    # Load mask if available and masking is enabled
    mask = None
    if config.USE_MASK and not args.no_mask:
        from src.utils import load_video_mask
        mask = load_video_mask(input_path)
        if mask is not None:
            logger.info(f"Mask loaded successfully - shape: {mask.shape}")
        else:
            logger.info("No mask found")
    elif args.no_mask:
        logger.info("Masking disabled via command line argument")
    else:
        logger.info("Masking disabled in configuration")

    # Initialize detector with built-in tracking
    logger.info("Initializing YOLOv11 detector with built-in tracking...")
    detector = YOLODetectorWithTracking(
        model_path=args.model,
        confidence=args.confidence,
        iou_threshold=args.iou,
        use_cuda=not args.no_cuda,
        use_tensorrt=not args.no_tensorrt,
        cuda_device=args.cuda_device,
        tracker_type=args.tracker
    )

    # Warm up the detector
    detector.warm_up()

    # Get class names
    class_names = detector.get_class_names()
    logger.info(f"Loaded {len(class_names)} object classes")
    logger.info(f"Using {args.tracker} tracker")

    # Open video capture
    cap = cv2.VideoCapture(input_path)

    # Determine output dimensions
    output_width = config.RESIZE_WIDTH or video_info['width']
    output_height = config.RESIZE_HEIGHT or video_info['height']
    output_fps = args.fps or config.TARGET_FPS or video_info['fps']

    logger.info(f"Output video settings: {output_width}x{output_height} @ {output_fps} FPS")

    # Create video writer
    writer = create_video_writer(output_path, output_fps, output_width, output_height)

    # Processing variables
    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    logger.info("Starting video processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if configured
            if frame_count % config.FRAME_SKIP != 0:
                continue

            processed_frames += 1

            # Resize frame if needed
            if config.RESIZE_WIDTH or config.RESIZE_HEIGHT:
                frame = resize_frame(frame, config.RESIZE_WIDTH, config.RESIZE_HEIGHT)

            # Detect and track objects (with mask if available)
            tracks = detector.detect_and_track(frame, mask)

            # Filter tracks for specified objects only if requested
            if args.specified_only:
                tracks = [t for t in tracks if t['class_id'] == config.OBJECT_CLASS_ID]

            # Draw tracking results
            output_frame = draw_yolo_tracks(frame, tracks)

            # Add mask overlay if enabled
            if mask is not None and (config.SHOW_MASK_OVERLAY or args.show_mask):
                # Resize mask to match output frame if needed
                if mask.shape[:2] != output_frame.shape[:2]:
                    display_mask = cv2.resize(mask, (output_frame.shape[1], output_frame.shape[0]))
                else:
                    display_mask = mask

                # Create colored mask overlay (semi-transparent red for masked-out areas)
                mask_overlay = np.zeros_like(output_frame)
                mask_overlay[:, :, 2] = (255 - display_mask)  # Red channel where mask is 0 (masked areas)
                output_frame = cv2.addWeighted(output_frame, 0.8, mask_overlay, 0.2, 0)

            # Add frame info
            info_text = f"Frame: {processed_frames} | Tracks: {len(tracks)}"
            cv2.putText(
                output_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            # Write frame
            writer.write(output_frame)

            # Display if requested
            if args.display:
                cv2.imshow('Object Tracking', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress update
            if processed_frames % 100 == 0:
                elapsed = time.time() - start_time
                fps = processed_frames / elapsed
                progress = frame_count / video_info['frame_count'] * 100
                logger.info(f"Progress: {progress:.1f}% | FPS: {fps:.1f} | Tracks: {len(tracks)}")

    finally:
        # Cleanup
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    # Final statistics
    elapsed_time = time.time() - start_time
    avg_fps = processed_frames / elapsed_time

    logger.info("Processing completed!")
    logger.info(f"Processed {processed_frames} frames in {elapsed_time:.2f}s")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Output saved to: {output_path}")


def main():
    """Main function."""
    # Setup logging
    setup_logging()

    # Parse arguments
    args = parse_arguments()

    # List GPUs if requested
    if args.list_gpus:
        from src.utils import list_cuda_devices
        list_cuda_devices()
        return

    # Print system info
    print_system_info()

    # List videos if requested
    if args.list_videos:
        list_available_videos()
        return

    # Validate arguments
    if not args.input:
        logger.error("Input video path is required. Use --list-videos to see available videos.")
        return

    if not args.output:
        # Generate output filename
        input_path = Path(args.input)
        output_filename = f"{input_path.stem}_tracked{input_path.suffix}"
        args.output = str(config.OUTPUTS_DIR / output_filename)
        logger.info(f"Output path not specified, using: {args.output}")

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Process video
        process_video(args.input, args.output, args)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
