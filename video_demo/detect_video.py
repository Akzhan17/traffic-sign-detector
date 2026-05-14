"""
Real-time Video Detection for Traffic Signs
Supports: Webcam, Video Files, and Image Files
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import argparse
import sys
import time


class TrafficSignDetector:
    def __init__(self, model_path='model/best.pt', confidence=0.5, device='auto'):
        """
        Initialize Traffic Sign Detector

        Args:
            model_path: Path to trained YOLO model
            confidence: Minimum confidence threshold for detections
            device: Device to run inference on ('auto', 'cuda', 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"🚀 Loading model from {model_path}...")
        print(f"📱 Using device: {self.device}")

        # Load model
        try:
            self.model = YOLO(model_path)
            # Move to device if needed
            if self.device == 'cuda':
                self.model.to('cuda')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("⚠️  Make sure you have trained the model first!")
            sys.exit(1)

        self.confidence = confidence
        self.class_names = self._get_class_names()

    def _get_class_names(self):
        """Get class names from the model"""
        # Default traffic sign classes (common ones)
        return {
            0: "Speed Limit 20",
            1: "Speed Limit 30",
            2: "Speed Limit 50",
            3: "Speed Limit 60",
            4: "Speed Limit 70",
            5: "Speed Limit 80",
            6: "End Speed Limit 80",
            7: "Speed Limit 100",
            8: "Speed Limit 120",
            9: "No Passing",
            10: "No Passing Trucks",
            11: "Right of Way",
            12: "Priority Road",
            13: "Yield",
            14: "Stop",
            15: "No Vehicles",
            16: "Trucks Prohibited",
            17: "No Entry",
            18: "General Caution",
            19: "Dangerous Curve Left",
            20: "Dangerous Curve Right",
            21: "Double Curve",
            22: "Bumpy Road",
            23: "Slippery Road",
            24: "Road Narrows",
            25: "Road Work",
            26: "Traffic Signals",
            27: "Pedestrians",
            28: "Children Crossing",
            29: "Bicycles Crossing",
            30: "Ice/Snow",
            31: "Wild Animals",
            32: "End Limits",
            33: "Turn Right",
            34: "Turn Left",
            35: "Ahead Only",
            36: "Straight or Right",
            37: "Straight or Left",
            38: "Keep Right",
            39: "Keep Left",
            40: "Roundabout",
            41: "End No Passing",
            42: "End No Passing Trucks"
        }

    def process_frame(self, frame):
        """
        Process a single frame for detection

        Args:
            frame: Input image (numpy array)

        Returns:
            annotated_frame: Frame with bounding boxes
            detections: List of detection dictionaries
        """
        # Run inference
        results = self.model(frame, conf=self.confidence, device=self.device)

        # Get detections
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = self.class_names.get(class_id, f"Class_{class_id}")

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })

        # Get annotated frame
        annotated_frame = results[0].plot()

        return annotated_frame, detections

    def process_webcam(self, camera_id=0):
        """
        Process real-time webcam feed

        Args:
            camera_id: Webcam device ID (default 0)
        """
        print("\n🎥 Opening webcam...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("❌ Failed to open webcam!")
            return

        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"📹 Webcam resolution: {width}x{height}")
        print(f"⚡ Target FPS: {fps}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'd' - Toggle detection info")
        print("-" * 40)

        frame_count = 0
        start_time = time.time()
        show_info = True

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Process frame
                annotated_frame, detections = self.process_frame(frame)

                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps_display = frame_count / elapsed_time
                    if show_info:
                        cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)

                # Show detection count
                if show_info:
                    cv2.putText(annotated_frame, f"Detections: {len(detections)}",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)

                # Display frame
                cv2.imshow('Traffic Sign Detection - Press Q to quit', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                elif key == ord('d'):
                    show_info = not show_info
                    print(f"ℹ️ Detection info: {'ON' if show_info else 'OFF'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Webcam closed")

    def process_video(self, video_path):
        """
        Process video file

        Args:
            video_path: Path to video file
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n🎬 Processing video: {video_path}")
        print(f"📹 Resolution: {width}x{height}")
        print(f"⚡ FPS: {fps}")
        print(f"🎞️ Total frames: {total_frames}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  ' ' - Pause/Resume")
        print("-" * 40)

        frame_count = 0
        start_time = time.time()
        paused = False

        # Create output video writer
        output_path = f"output_{Path(video_path).stem}.mp4"
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (width, height))

        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\n✅ Video processing complete!")
                        break

                    # Process frame
                    annotated_frame, detections = self.process_frame(frame)

                    # Write frame
                    out.write(annotated_frame)

                    # Show progress
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps_display = frame_count / elapsed_time
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% | FPS: {fps_display:.1f} | Detections: {len(detections)}",
                              end='\r')

                    # Display frame
                    cv2.imshow('Traffic Sign Detection - Press Q to quit', annotated_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n⏸️ {'Paused' if paused else 'Resumed'}")

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\n✅ Output video saved to: {output_path}")

    def process_image(self, image_path):
        """
        Process a single image

        Args:
            image_path: Path to image file
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return

        print(f"\n🖼️ Processing image: {image_path}")

        # Process image
        annotated_image, detections = self.process_frame(image)

        # Print detections
        print(f"\n📊 Found {len(detections)} traffic signs:")
        for i, detection in enumerate(detections, 1):
            print(f"  {i}. {detection['class_name']} (Confidence: {detection['confidence']:.2f})")

        # Save result
        output_path = f"output_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Result saved to: {output_path}")

        # Display image
        cv2.imshow('Traffic Sign Detection - Press any key to close', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_folder(self, folder_path):
        """
        Process all images in a folder

        Args:
            folder_path: Path to folder containing images
        """
        folder = Path(folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        images = []
        for ext in image_extensions:
            images.extend(folder.glob(f"*{ext}"))

        print(f"\n📁 Found {len(images)} images in {folder_path}")

        for i, image_path in enumerate(images, 1):
            print(f"\nProcessing {i}/{len(images)}: {image_path.name}")
            self.process_image(str(image_path))


def main():
    parser = argparse.ArgumentParser(description='Traffic Sign Detection System')
    parser.add_argument('--source', type=str, default='webcam',
                        help='Source: webcam, video file path, image path, or folder path')
    parser.add_argument('--model', type=str, default='model/best.pt',
                        help='Path to model file')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu')

    args = parser.parse_args()

    # Initialize detector
    detector = TrafficSignDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device
    )

    # Process based on source type
    if args.source == 'webcam':
        detector.process_webcam()
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detector.process_video(args.source)
    elif args.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        detector.process_image(args.source)
    elif Path(args.source).is_dir():
        detector.process_folder(args.source)
    else:
        print(f"❌ Unknown source type: {args.source}")
        print("Supported sources: webcam, video file, image file, folder")


if __name__ == "__main__":
    main()