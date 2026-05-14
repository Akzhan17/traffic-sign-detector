"""
Fixed Training Script for Traffic Sign Detection
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import torch
import os


def setup_and_train():
    """Setup dataset path and train model"""

    print("=" * 60)
    print("🚦 TRAFFIC SIGN DETECTION TRAINING")
    print("=" * 60)

    # Get the absolute path to your dataset
    project_root = Path(__file__).parent.parent
    data_path = project_root / "dataset" / "data"

    print(f"\n📂 Looking for dataset at: {data_path}")

    # Check if dataset exists
    if not data_path.exists():
        print(f"\n❌ Dataset not found at {data_path}")
        print("\nPlease move your dataset to this location:")
        print(f"  {data_path}")
        print("\nExpected structure:")
        print("  training/data/")
        print("  ├── images/")
        print("  │   ├── train/")
        print("  │   ├── val/")
        print("  │   └── test/")
        print("  └── labels/")
        print("      ├── train/")
        print("      ├── val/")
        print("      └── test/")
        return False

    # Find the correct structure
    possible_train_paths = [
        data_path / "images" / "train",
        data_path / "train" / "images",
        data_path / "train",
    ]

    train_path = None
    for path in possible_train_paths:
        if path.exists():
            train_path = path
            break

    if train_path is None:
        print(f"\n❌ Could not find training images in {data_path}")
        print("\nContents of data directory:")
        for item in data_path.iterdir():
            print(f"  - {item.name}")
        return False

    print(f"\n✅ Found training images at: {train_path}")

    # Count images
    train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png")) + list(train_path.glob("*.jpeg"))
    print(f"📊 Training images found: {len(train_images)}")

    if len(train_images) == 0:
        print("❌ No images found! Check your dataset structure.")
        return False

    # Create dataset.yaml
    dataset_config = {
        'path': str(data_path.absolute()),
        'train': 'images/train' if (data_path / 'images' / 'train').exists() else 'train',
        'val': 'images/val' if (data_path / 'images' / 'val').exists() else 'val',
        'nc': 43,
        'names': {
            0: 'speed_limit_20',
            1: 'speed_limit_30',
            2: 'speed_limit_50',
            3: 'speed_limit_60',
            4: 'speed_limit_70',
            5: 'speed_limit_80',
            6: 'end_of_speed_limit_80',
            7: 'speed_limit_100',
            8: 'speed_limit_120',
            9: 'no_passing',
            10: 'no_passing_trucks',
            11: 'right_of_way_at_next_intersection',
            12: 'priority_road',
            13: 'yield',
            14: 'stop',
            15: 'no_vehicles',
            16: 'vehicles_over_3_5_tons_prohibited',
            17: 'no_entry',
            18: 'general_caution',
            19: 'dangerous_curve_left',
            20: 'dangerous_curve_right',
            21: 'double_curve',
            22: 'bumpy_road',
            23: 'slippery_road',
            24: 'road_narrows_on_right',
            25: 'road_work',
            26: 'traffic_signals',
            27: 'pedestrians',
            28: 'children_crossing',
            29: 'bicycles_crossing',
            30: 'ice_snow',
            31: 'wild_animals_crossing',
            32: 'end_of_all_speed_and_passing_limits',
            33: 'turn_right_ahead',
            34: 'turn_left_ahead',
            35: 'ahead_only',
            36: 'go_straight_or_right',
            37: 'go_straight_or_left',
            38: 'keep_right',
            39: 'keep_left',
            40: 'roundabout',
            41: 'end_of_no_passing',
            42: 'end_of_no_passing_trucks'
        }
    }

    # Save YAML file
    yaml_path = project_root / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"\n✅ Created dataset configuration: {yaml_path}")

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'  # Use Apple Silicon GPU
        print(f"🍎 Using Apple MPS (Metal Performance Shaders)")
    else:
        print(f"📱 Using device: {device}")

    # Initialize model
    print("\n📦 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    # Training parameters
    print("\n🎯 Starting training...")
    print(f"   Epochs: 50")
    print(f"   Batch size: 8")
    print(f"   Image size: 640")
    print(f"   Device: {device}")

    # Train the model
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=50,
            imgsz=640,
            batch=8,
            device=device,
            workers=4,
            patience=10,
            save=True,
            save_period=10,
            project='traffic_sign_detection',
            name='yolov8n_traffic_signs',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5
        )

        # Copy best model to model directory
        best_model_path = project_root / 'model' / 'best.pt'
        best_model_path.parent.mkdir(exist_ok=True)

        trained_model = Path('traffic_sign_detection/yolov8n_traffic_signs/weights/best.pt')
        if trained_model.exists():
            import shutil
            shutil.copy(trained_model, best_model_path)
            print(f"\n✅ Model saved to: {best_model_path}")

        print("\n🎉 Training completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = setup_and_train()

    if success:
        print("\n" + "=" * 60)
        print("📋 NEXT STEPS:")
        print("=" * 60)
        print("\n1. Test your model:")
        print("   python video_demo/detect_video.py --source webcam")
        print("\n2. Start the API server:")
        print("   cd backend && uvicorn app:app --reload")
        print("\n3. Launch the web interface:")
        print("   streamlit run frontend/streamlit_app.py")
        print("\n4. Test with an image:")
        print("   python video_demo/detect_video.py --source path/to/image.jpg")
    else:
        print("\n❌ Training failed. Check the error messages above.")