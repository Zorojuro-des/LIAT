#!/usr/bin/env python3
"""
Complete Player Tracking Pipeline Runner

This script runs the complete pipeline for tracking players across multiple camera views:
1. Detects players in both videos using YOLOv11
2. Extracts rich features (appearance, position, motion)
3. Matches players across views using multi-feature similarity
4. Ensures temporal consistency across frames
5. Exports results in multiple formats

Usage:
    python run_pipeline.py
    python run_pipeline.py --broadcast custom_broadcast.mp4 --tacticam custom_tacticam.mp4
"""

import os
import sys
import argparse
import time
from datetime import datetime
import json
import shutil

# Import our custom modules
from player_tracker import PlayerTracker
from temporal_tracker import TemporalPlayerTracker


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'cv2', 'numpy', 'pandas', 'scipy', 'sklearn', 
        'ultralytics', 'tqdm', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True


def check_input_files(broadcast_path: str, tacticam_path: str, model_path: str):
    """Check if all input files exist."""
    print("📁 Checking input files...")
    
    files_to_check = [
        (broadcast_path, "Broadcast video"),
        (tacticam_path, "Tacticam video"),
        (model_path, "YOLOv11 model")
    ]
    
    missing_files = []
    for file_path, description in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append(f"{description}: {file_path}")
            print(f"❌ Missing: {description} ({file_path})")
        else:
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ Found: {description} ({file_size:.1f} MB)")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True


def create_output_directory(output_dir: str):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"player_tracking_{timestamp}")
    
    os.makedirs(output_path, exist_ok=True)
    print(f"📂 Created output directory: {output_path}")
    
    return output_path


def run_basic_tracking(broadcast_path: str, tacticam_path: str, 
                      model_path: str, output_dir: str):
    """Run the basic player tracking pipeline."""
    print("\n" + "="*60)
    print("🎯 BASIC PLAYER TRACKING PIPELINE")
    print("="*60)
    
    # Initialize tracker
    tracker = PlayerTracker(model_path)
    
    # Run the complete pipeline
    results = tracker.process_videos(broadcast_path, tacticam_path, output_dir)
    
    return results


def run_temporal_tracking(output_dir: str):
    """Run the temporal tracking pipeline."""
    print("\n" + "="*60)
    print("🎯 TEMPORAL TRACKING PIPELINE")
    print("="*60)
    
    # Check if detection files exist
    broadcast_detections = os.path.join(output_dir, "broadcast_detections.json")
    tacticam_detections = os.path.join(output_dir, "tacticam_detections.json")
    
    if not os.path.exists(broadcast_detections):
        print(f"❌ Broadcast detections not found: {broadcast_detections}")
        return None
    
    if not os.path.exists(tacticam_detections):
        print(f"❌ Tacticam detections not found: {tacticam_detections}")
        return None
    
    # Change to output directory to run temporal tracking
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Initialize temporal tracker
        tracker = TemporalPlayerTracker()
        
        # Track each view separately first
        print("\n1️⃣ Tracking players in broadcast view...")
        broadcast_results = tracker.track_single_view("broadcast_detections.json", "broadcast")
        
        # Reset tracker for second view
        tracker = TemporalPlayerTracker()
        
        print("\n2️⃣ Tracking players in tacticam view...")
        tacticam_results = tracker.track_single_view("tacticam_detections.json", "tacticam")
        
        print("\n3️⃣ Aligning player IDs across views...")
        aligned_results = tracker.align_cross_view_ids(broadcast_results, tacticam_results)
        
        print("\n4️⃣ Exporting temporal tracking results...")
        tracker.export_results(aligned_results)
        
        print("\n🎉 Temporal tracking completed!")
        
        return aligned_results
        
    finally:
        os.chdir(original_dir)


def generate_summary_report(basic_results, temporal_results, output_dir: str):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("📊 GENERATING SUMMARY REPORT")
    print("="*60)
    
    report = {
        "pipeline_info": {
            "timestamp": datetime.now().isoformat(),
            "output_directory": output_dir,
            "pipeline_version": "1.0"
        },
        "basic_tracking": {},
        "temporal_tracking": {},
        "comparison": {}
    }
    
    # Basic tracking stats
    if basic_results:
        report["basic_tracking"] = {
            "total_detections": len(basic_results),
            "unique_players": len(set(r['player_id'] for r in basic_results)),
            "frames_processed": len(set(r['frame'] for r in basic_results)),
            "broadcast_detections": len([r for r in basic_results if r['view'] == 'broadcast']),
            "tacticam_detections": len([r for r in basic_results if r['view'] == 'tacticam'])
        }
    
    # Temporal tracking stats
    if temporal_results:
        report["temporal_tracking"] = {
            "total_detections": len(temporal_results),
            "unique_players": len(set(r['player_id'] for r in temporal_results)),
            "frames_processed": len(set(r['frame'] for r in temporal_results)),
            "broadcast_detections": len([r for r in temporal_results if r['view'] == 'broadcast']),
            "tacticam_detections": len([r for r in temporal_results if r['view'] == 'tacticam'])
        }
    
    # Save report
    report_path = os.path.join(output_dir, "pipeline_summary.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Summary report saved to: {report_path}")
    
    # Print summary
    print("\n📋 PIPELINE SUMMARY:")
    print("-" * 40)
    
    if basic_results:
        print("Basic Tracking Results:")
        print(f"   - Total detections: {report['basic_tracking']['total_detections']}")
        print(f"   - Unique players: {report['basic_tracking']['unique_players']}")
        print(f"   - Frames processed: {report['basic_tracking']['frames_processed']}")
    
    if temporal_results:
        print("Temporal Tracking Results:")
        print(f"   - Total detections: {report['temporal_tracking']['total_detections']}")
        print(f"   - Unique players: {report['temporal_tracking']['unique_players']}")
        print(f"   - Frames processed: {report['temporal_tracking']['frames_processed']}")
    
    return report


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete Player Tracking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py
    python run_pipeline.py --broadcast broadcast.mp4 --tacticam tacticam.mp4
    python run_pipeline.py --model custom_model.pt --output results/
    python run_pipeline.py --skip-temporal
        """
    )
    
    parser.add_argument("--broadcast", default="broadcast.mp4", 
                       help="Path to broadcast video file")
    parser.add_argument("--tacticam", default="tacticam.mp4", 
                       help="Path to tacticam video file")
    parser.add_argument("--model", default="best.pt", 
                       help="Path to YOLOv11 model file")
    parser.add_argument("--output", default="results", 
                       help="Output directory for results")
    parser.add_argument("--skip-temporal", action="store_true",
                       help="Skip temporal tracking step")
    parser.add_argument("--skip-checks", action="store_true",
                       help="Skip dependency and file checks")
    
    args = parser.parse_args()
    
    print("🚀 PLAYER TRACKING PIPELINE STARTED")
    print("=" * 60)
    print(f"Broadcast video: {args.broadcast}")
    print(f"Tacticam video: {args.tacticam}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        # Step 2: Check input files
        if not check_input_files(args.broadcast, args.tacticam, args.model):
            sys.exit(1)
    
    # Step 3: Create output directory
    output_dir = create_output_directory(args.output)
    
    # Step 4: Run basic tracking
    try:
        basic_results = run_basic_tracking(args.broadcast, args.tacticam, args.model, output_dir)
    except Exception as e:
        print(f"❌ Basic tracking failed: {e}")
        basic_results = None
    
    # Step 5: Run temporal tracking
    temporal_results = None
    if not args.skip_temporal and basic_results:
        try:
            temporal_results = run_temporal_tracking(output_dir)
        except Exception as e:
            print(f"❌ Temporal tracking failed: {e}")
            temporal_results = None
    
    # Step 6: Generate summary report
    report = generate_summary_report(basic_results, temporal_results, output_dir)
    
    # Final summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED!")
    print("="*60)
    print(f"⏱️  Total execution time: {elapsed_time:.2f} seconds")
    print(f"📂 Results saved to: {output_dir}")
    
    if basic_results and temporal_results:
        print("✅ Both basic and temporal tracking completed successfully!")
    elif basic_results:
        print("✅ Basic tracking completed successfully!")
        print("⚠️  Temporal tracking was skipped or failed.")
    else:
        print("❌ Pipeline failed - no results generated.")
        sys.exit(1)
    
    print("\n📋 Output files:")
    for file in os.listdir(output_dir):
        if file.endswith(('.json', '.csv')):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   - {file} ({file_size:.1f} KB)")
    
    print("\n🎯 Next steps:")
    print("   1. Review the generated CSV files for player tracking results")
    print("   2. Use the JSON files for further analysis or visualization")
    print("   3. Check the pipeline summary for detailed statistics")
    
    return basic_results, temporal_results


if __name__ == "__main__":
    main()
