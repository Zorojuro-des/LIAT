#!/usr/bin/env python3
"""
Quick System Test Script

This script runs basic tests to verify the player tracking system is working correctly.
"""

import os
import sys
import json
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import shutil


def test_dependencies():
    """Test if all required dependencies are available."""
    print("🔍 Testing dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'pandas', 'scipy', 'sklearn', 
        'ultralytics', 'tqdm', 'matplotlib'
    ]
    
    success = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            success = False
    
    return success


def test_model_loading():
    """Test if YOLOv11 model can be loaded."""
    print("\n🤖 Testing model loading...")
    
    model_path = "best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def test_video_access():
    """Test if video files can be accessed."""
    print("\n🎥 Testing video access...")
    
    video_files = ["broadcast.mp4", "tacticam.mp4"]
    success = True
    
    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"❌ Video file not found: {video_file}")
            success = False
            continue
        
        try:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_file}")
                success = False
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"✅ {video_file}: {frame_count} frames, {fps:.1f} FPS")
            cap.release()
        except Exception as e:
            print(f"❌ Video access error for {video_file}: {e}")
            success = False
    
    return success


def test_basic_detection():
    """Test basic player detection on a single frame."""
    print("\n👥 Testing basic detection...")
    
    model_path = "best.pt"
    video_path = "broadcast.mp4"
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("❌ Required files not found for detection test")
        return False
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Read first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ Cannot read frame from video")
            return False
        
        # Run detection
        results = model.predict(frame, verbose=False)[0]
        
        # Count player detections
        player_count = 0
        for cls in results.boxes.cls:
            if model.names[int(cls)] == "player":
                player_count += 1
        
        print(f"✅ Detected {player_count} players in first frame")
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction functionality."""
    print("\n🔧 Testing feature extraction...")
    
    try:
        # Create a dummy detection
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_bbox = [100, 100, 200, 200]
        
        # Import our tracker
        from player_tracker import PlayerTracker
        
        tracker = PlayerTracker("best.pt")
        features = tracker.extract_features(dummy_frame, dummy_bbox)
        
        if features is None:
            print("❌ Feature extraction returned None")
            return False
        
        required_keys = ['position', 'size', 'color_hist_h', 'color_hist_s', 'color_hist_v']
        for key in required_keys:
            if key not in features:
                print(f"❌ Missing feature: {key}")
                return False
        
        print("✅ Feature extraction working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False


def test_output_creation():
    """Test if output files can be created."""
    print("\n📝 Testing output creation...")
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Test CSV creation
        import pandas as pd
        dummy_data = [{
            'frame': 0,
            'view': 'test',
            'player_id': 1,
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'position': [0.5, 0.5]
        }]
        
        df = pd.DataFrame(dummy_data)
        csv_path = os.path.join(temp_dir, "test.csv")
        df.to_csv(csv_path, index=False)
        
        # Test JSON creation
        json_path = os.path.join(temp_dir, "test.json")
        with open(json_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        
        # Check if files exist
        if os.path.exists(csv_path) and os.path.exists(json_path):
            print("✅ Output file creation working")
            success = True
        else:
            print("❌ Output file creation failed")
            success = False
        
        # Cleanup
        shutil.rmtree(temp_dir)
        return success
        
    except Exception as e:
        print(f"❌ Output creation test failed: {e}")
        return False


def run_all_tests():
    """Run all system tests."""
    print("🚀 LIAT SYSTEM TESTS")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Loading", test_model_loading),
        ("Video Access", test_video_access),
        ("Basic Detection", test_basic_detection),
        ("Feature Extraction", test_feature_extraction),
        ("Output Creation", test_output_creation)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


def main():
    """Main test function."""
    success = run_all_tests()
    
    if success:
        print("\n🎯 Next steps:")
        print("   1. Run the complete pipeline: python run_pipeline.py")
        print("   2. Check the results in the output directory")
        print("   3. Review the generated CSV and JSON files")
        sys.exit(0)
    else:
        print("\n❌ Please fix the failing tests before running the pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
