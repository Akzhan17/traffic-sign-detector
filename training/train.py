"""
Traffic Sign Detection Model Training - FIXED VERSION
Handles custom dataset locations automatically
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
import shutil

def find_dataset_path():
    """Find the dataset location automatically"""
    possible_paths = [
        Path(__file__).parent / 'data',  # training/data
        Path.cwd() / 'training' / 'data',  # ./training/data
        Path.cwd() / 'data',  # ./data
        Path.home() / 'Downloads' / 'Traffic_Sign_Detection_Europe',  # Downloads
        Path.home() / 'Downloads' / 'archive',  # Alternative
    ]

    for path in possible_paths:
        if path.exists() and (path / 'images').exists():
            print(f"✅ Found dataset at: {path}")
            return path

    return None

def get_dataset_structure(data_path):
    """Check dataset structure and return correct paths"""
    # Check different possible structures
    structures = [
        # Structure 1: data/images/train, data/labels/train
        {'train_img': data_path / 'images' / 'train',
         'val_img': data_path / 'images' / 'val',
         'train_lbl': data_path / 'labels' / 'train',
         'val_lbl': data_path / 'labels' / 'val'},

        # Structure 2: data/train/images, data/train/labels
        {'train_img': data_path / 'train' / 'images',
         'val_img': data_path / 'val' / 'images',
         'train_lbl': data_path / 'train' / 'labels',
         'val_lbl': data_path / 'val' / 'labels'},

        # Structure 3: data/train, data/valid (no subfolders)
        {'train_img': data_path / 'train',
         'val_img': data_path / 'valid',
         'train_lbl': data_path / 'train',
         'val_lbl': data_path / 'valid'},
    ]

    for struct in structures:
        if struct['train_img'].exists() and struct['val_img'].exists():
            print(f"📁 Found structure: {struct['train_img'].parent}")
            return struct

    return None

def create_dataset_yaml(data_path, struct):
    """Create dataset.yaml file with correct paths"""

    # Convert to absolute paths
    train_img_path = struct['train_img'].absolute()
    val_img_path = struct['val_img'].absolute()

    # For Roboflow Traffic Sign Detection Europe dataset
    dataset_config = {
        'path': str(data_path.absolute()),  # Root path
        'train': str(train_img_path.relative_to(data_path)),  # Relative path
        'val': str(val_img_path.relative_to(data_path)),  # Relative path
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

    # Save the yaml file
    yaml_path = Path(__file__).parent / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"✅ Created dataset.yaml at {yaml_path}")
    print(f"   Path: {data_path}")
    print(f"   Train: {dataset_config['train']}")
    print(f"   Val: {dataset_config['val']}")

    return yaml_path

def train_model():
    """Train YOLOv8 model on traffic sign dataset"""

    print("🚀 Starting Traffic Sign Detection Model Training")
    print("=" * 50)

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}")

    # Find dataset
    data_path = find_dataset_path()

    if not data_path:
        print("\n❌ Dataset not found!")
        print("\nPlease download the dataset from Roboflow:")
        print("1️⃣ Go to: https://universe.roboflow.com/roboflow-universe-projects/traffic-sign-detection-europe")
        print("2️⃣ Click 'Download Dataset'")
        print("3️⃣ Select 'YOLOv8' format")
        print("4️⃣ Extract the downloaded ZIP file")
        print("5️⃣ Move the extracted folder to: training/data/")
        print("\nOR run this command to download automatically:")
        print("   pip install roboflow")
        print("   python download_dataset.py")
        return False

    # Get dataset structure
    struct = get_dataset_structure(data_path)

    if not struct:
        print(f"\n❌ Invalid dataset structure at {data_path}")
        print("Expected structure:")
        print("  training/data/")
        print("  ├── images/")
        print("  │   ├── train/")
        print("  │   └── val/")
        print("  └── labels/")
        print("      ├── train/")
        print("      └── val/")
        return False

    # Create dataset.yaml
    yaml_path = create_dataset_yaml(data_path, struct)

    # Check if we have images
    train_images = list(struct['train_img'].glob('*.jpg')) + list(struct['train_img'].glob('*.png'))
    val_images = list(struct['val_img'].glob('*.jpg')) + list(struct['val_img'].glob('*.png'))

    print(f"\n📊 Dataset statistics:")
    print(f"   Training images: {len(train_images)}")
    print(f"   Validation images: {len(val_images)}")

    if len(train_images) == 0:
        print("\n❌ No training images found!")
        return False

    # Initialize YOLO model
    print("\n📦 Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    # Training parameters (optimized for CPU/Mac)
    training_params = {
        'data': str(yaml_path),
        'epochs': 50,
        'imgsz': 640,
        'batch': 8,  # Reduced batch size for CPU
        'device': device,
        'workers': 2,  # Reduced workers for Mac
        'patience': 10,
        'save': True,
        'save_period': 10,
        'project': 'traffic_sign_detection',
        'name': 'yolov8n_traffic_signs',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }

    print("\n🎯 Starting training with parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")

    print("\n🔄 Training in progress...")
    print("⏰ This may take a while on CPU. Consider using GPU for faster training.\n")

    try:
        # Train the model
        results = model.train(**training_params)

        # Save the best model to root model directory
        best_model_path = Path(__file__).parent.parent / 'model' / 'best.pt'
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy best model
        trained_model = Path('traffic_sign_detection/yolov8n_traffic_signs/weights/best.pt')
        if trained_model.exists():
            shutil.copy(trained_model, best_model_path)
            print(f"\n✅ Model saved to {best_model_path}")

        # Validate the model
        print("\n📊 Validating model...")
        val_results = model.val()
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")

        print("\n✅ Training completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def download_dataset_automatically():
    """Automatically download dataset using Roboflow API"""
    try:
        from roboflow import Roboflow
        print("\n📥 Downloading dataset automatically...")

        # Initialize Roboflow
        rf = Roboflow(api_key="YOUR_API_KEY")  # You need to get API key from roboflow.com
        project = rf.workspace("roboflow-universe-projects").project("traffic-sign-detection-europe")

        # Download dataset
        version = project.version(1)
        dataset = version.download(model_format="yolov8", location="training/data")

        print(f"✅ Dataset downloaded to: training/data")
        return True

    except ImportError:
        print("\n⚠️ Roboflow library not installed. Install with: pip install roboflow")
        return False
    except Exception as e:
        print(f"\n❌ Auto-download failed: {e}")
        return False

def main():
    """Main training function"""
    print("=" * 60)
    print("🚦 TRAFFIC SIGN DETECTION MODEL TRAINING")
    print("=" * 60)

    # Check for dataset
    data_path = find_dataset_path()

    if not data_path:
        print("\n📋 Dataset not found!")
        print("\nOptions:")
        print("1️⃣ Manual download (recommended):")
        print("   - Go to: https://universe.roboflow.com/roboflow-universe-projects/traffic-sign-detection-europe")
        print("   - Download YOLOv8 format")
        print("   - Extract to: training/data/")
        print("\n2️⃣ Auto-download (requires API key):")
        print("   - Get API key from https://roboflow.com")
        print("   - Run: pip install roboflow")

        response = input("\n👉 Have you downloaded the dataset? (yes/no/auto): ").lower()

        if response == 'auto':
            if not download_dataset_automatically():
                return
        elif response != 'yes':
            print("❌ Please download the dataset first, then run this script again.")
            return

    # Train the model
    success = train_model()

    if success:
        print("\n🎉 Model is ready for inference!")
        print("📍 Model location: model/best.pt")
        print("\nNext steps:")
        print("1️⃣ Test with image: python video_demo/detect_video.py --source path/to/image.jpg")
        print("2️⃣ Test with webcam: python video_demo/detect_video.py --source webcam")
        print("3️⃣ Start API server: cd backend && uvicorn app:app --reload")
        print("4️⃣ Launch web interface: streamlit run frontend/streamlit_app.py")
    else:
        print("\n❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()