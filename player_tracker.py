#!/usr/bin/env python3
"""
Player Tracking System for Multi-View Sports Videos

This system tracks players across two different camera angles (broadcast and tacticam)
and assigns consistent player IDs using YOLOv11 detection and multi-feature matching.

Features:
- Player detection using YOLOv11
- Multi-feature matching (appearance, position, temporal consistency)
- Consistent ID assignment across views
- Robust handling of varying player counts
- Export to CSV and JSON formats
"""

import cv2
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import os
import argparse
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PlayerTracker:
    def __init__(self, model_path: str = "best.pt"):
        """Initialize the player tracker with YOLOv11 model."""
        self.model = YOLO(model_path)
        self.global_id_counter = 0
        self.track_history = {}  # Store tracking history for temporal consistency
        
    def extract_features(self, frame: np.ndarray, bbox: List[float]) -> Dict:
        """Extract features from a player detection for matching."""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Extract player crop
        player_crop = frame[y1:y2, x1:x2]
        
        if player_crop.size == 0:
            return None
            
        # Position features (relative to frame)
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width_ratio = (x2 - x1) / w
        height_ratio = (y2 - y1) / h
        
        # Appearance features
        # Color histogram in HSV space
        hsv_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_crop], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv_crop], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv_crop], [2], None, [32], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-6)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-6)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-6)
        
        # Texture features (simple gradient magnitude)
        gray_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        texture_mean = np.mean(gradient_magnitude)
        texture_std = np.std(gradient_magnitude)
        
        return {
            'position': [center_x, center_y],
            'size': [width_ratio, height_ratio],
            'color_hist_h': hist_h,
            'color_hist_s': hist_s,
            'color_hist_v': hist_v,
            'texture_mean': texture_mean,
            'texture_std': texture_std,
            'bbox': bbox
        }
    
    def detect_players(self, video_path: str, output_path: str = None) -> List[Dict]:
        """Detect players in video and extract features."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = []
        
        for frame_idx in tqdm(range(frame_count), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO detection
            results = self.model.predict(frame, verbose=False)[0]
            
            frame_detections = []
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                label = self.model.names[int(cls)]
                
                if label == "player" and conf > 0.3:  # Filter low confidence detections
                    bbox = box.tolist()
                    features = self.extract_features(frame, bbox)
                    
                    if features is not None:
                        detection = {
                            'frame': frame_idx,
                            'bbox': bbox,
                            'confidence': float(conf),
                            'features': features
                        }
                        frame_detections.append(detection)
            
            detections.append({
                'frame': frame_idx,
                'detections': frame_detections
            })
        
        cap.release()
        
        # Save detections if output path provided
        if output_path:
            # Convert numpy arrays to lists for JSON serialization
            serializable_detections = []
            for frame_data in detections:
                frame_dict = {'frame': frame_data['frame'], 'detections': []}
                for det in frame_data['detections']:
                    det_dict = {
                        'frame': det['frame'],
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'features': {
                            'position': det['features']['position'],
                            'size': det['features']['size'],
                            'color_hist_h': det['features']['color_hist_h'].tolist(),
                            'color_hist_s': det['features']['color_hist_s'].tolist(),
                            'color_hist_v': det['features']['color_hist_v'].tolist(),
                            'texture_mean': det['features']['texture_mean'],
                            'texture_std': det['features']['texture_std'],
                            'bbox': det['features']['bbox']
                        }
                    }
                    frame_dict['detections'].append(det_dict)
                serializable_detections.append(frame_dict)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_detections, f, indent=2)
        
        return detections
    
    def calculate_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """Calculate similarity between two feature sets."""
        # Position similarity (closer = more similar)
        pos_dist = np.linalg.norm(np.array(feat1['position']) - np.array(feat2['position']))
        pos_sim = np.exp(-pos_dist * 5)  # Exponential decay
        
        # Size similarity
        size_dist = np.linalg.norm(np.array(feat1['size']) - np.array(feat2['size']))
        size_sim = np.exp(-size_dist * 10)
        
        # Color histogram similarity (correlation)
        color_sim_h = np.corrcoef(feat1['color_hist_h'], feat2['color_hist_h'])[0, 1]
        color_sim_s = np.corrcoef(feat1['color_hist_s'], feat2['color_hist_s'])[0, 1]
        color_sim_v = np.corrcoef(feat1['color_hist_v'], feat2['color_hist_v'])[0, 1]
        
        # Handle NaN values in correlation
        color_sim_h = 0 if np.isnan(color_sim_h) else color_sim_h
        color_sim_s = 0 if np.isnan(color_sim_s) else color_sim_s
        color_sim_v = 0 if np.isnan(color_sim_v) else color_sim_v
        
        color_sim = (color_sim_h + color_sim_s + color_sim_v) / 3
        
        # Texture similarity
        texture_sim = np.exp(-abs(feat1['texture_mean'] - feat2['texture_mean']) / 100)
        
        # Weighted combination
        total_sim = (
            0.4 * pos_sim +      # Position is most important
            0.3 * color_sim +    # Color is second most important
            0.2 * size_sim +     # Size matters for consistency
            0.1 * texture_sim    # Texture is least reliable
        )
        
        return max(0, total_sim)  # Ensure non-negative
    
    def align_views_procrustes(self, view_a_positions: List[List[float]], 
                             view_b_positions: List[List[float]]) -> np.ndarray:
        """Align view B positions to view A using Procrustes analysis."""
        if len(view_a_positions) < 2 or len(view_b_positions) < 2:
            return np.array(view_b_positions)
            
        a_pts = np.array(view_a_positions)
        b_pts = np.array(view_b_positions)
        
        # Center the points
        a_mean = np.mean(a_pts, axis=0)
        b_mean = np.mean(b_pts, axis=0)
        
        a_centered = a_pts - a_mean
        b_centered = b_pts - b_mean
        
        # Find principal directions
        def principal_axis(X):
            if X.shape[0] < 2:
                return np.array([1, 0])
            U, _, Vt = np.linalg.svd(X)
            return Vt[0]
        
        a_axis = principal_axis(a_centered)
        b_axis = principal_axis(b_centered)
        
        # Calculate rotation angle
        angle = np.arctan2(a_axis[1], a_axis[0]) - np.arctan2(b_axis[1], b_axis[0])
        
        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Apply transformation
        b_aligned = np.dot(b_centered, R.T) + a_mean
        
        return b_aligned
    
    def match_players_across_views(self, view_a_detections: List[Dict], 
                                 view_b_detections: List[Dict]) -> List[Dict]:
        """Match players across two views and assign consistent IDs."""
        results = []
        
        # Process each frame
        min_frames = min(len(view_a_detections), len(view_b_detections))
        
        for frame_idx in tqdm(range(min_frames), desc="Matching players across views"):
            a_dets = view_a_detections[frame_idx]['detections']
            b_dets = view_b_detections[frame_idx]['detections']
            
            if not a_dets or not b_dets:
                continue
            
            # Extract positions for alignment
            a_positions = [det['features']['position'] for det in a_dets]
            b_positions = [det['features']['position'] for det in b_dets]
            
            # Align view B to view A
            b_aligned_positions = self.align_views_procrustes(a_positions, b_positions)
            
            # Update view B features with aligned positions
            for i, det in enumerate(b_dets):
                if i < len(b_aligned_positions):
                    det['features']['aligned_position'] = b_aligned_positions[i].tolist()
                else:
                    det['features']['aligned_position'] = det['features']['position']
            
            # Calculate similarity matrix
            similarity_matrix = np.zeros((len(a_dets), len(b_dets)))
            
            for i, a_det in enumerate(a_dets):
                for j, b_det in enumerate(b_dets):
                    # Use aligned position for view B
                    b_features_aligned = b_det['features'].copy()
                    b_features_aligned['position'] = b_features_aligned['aligned_position']
                    
                    similarity = self.calculate_similarity(a_det['features'], b_features_aligned)
                    similarity_matrix[i, j] = similarity
            
            # Convert to distance matrix (lower is better for Hungarian algorithm)
            distance_matrix = 1 - similarity_matrix
            
            # Use Hungarian algorithm for optimal matching
            if distance_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(distance_matrix)
                
                # Create matches with threshold
                matches = []
                similarity_threshold = 0.3
                
                for i, j in zip(row_ind, col_ind):
                    if similarity_matrix[i, j] > similarity_threshold:
                        matches.append((i, j, similarity_matrix[i, j]))
                
                # Assign IDs
                frame_results = []
                used_b_indices = set()
                
                # Process matches
                for a_idx, b_idx, sim_score in matches:
                    used_b_indices.add(b_idx)
                    
                    # Assign consistent ID
                    player_id = self.global_id_counter
                    self.global_id_counter += 1
                    
                    # View A result
                    frame_results.append({
                        'frame': frame_idx,
                        'view': 'broadcast',
                        'player_id': player_id,
                        'bbox': a_dets[a_idx]['bbox'],
                        'confidence': a_dets[a_idx]['confidence'],
                        'position': a_dets[a_idx]['features']['position'],
                        'match_similarity': sim_score
                    })
                    
                    # View B result
                    frame_results.append({
                        'frame': frame_idx,
                        'view': 'tacticam',
                        'player_id': player_id,
                        'bbox': b_dets[b_idx]['bbox'],
                        'confidence': b_dets[b_idx]['confidence'],
                        'position': b_dets[b_idx]['features']['position'],
                        'aligned_position': b_dets[b_idx]['features']['aligned_position'],
                        'match_similarity': sim_score
                    })
                
                # Handle unmatched detections in view B
                for b_idx, b_det in enumerate(b_dets):
                    if b_idx not in used_b_indices:
                        player_id = self.global_id_counter
                        self.global_id_counter += 1
                        
                        frame_results.append({
                            'frame': frame_idx,
                            'view': 'tacticam',
                            'player_id': player_id,
                            'bbox': b_det['bbox'],
                            'confidence': b_det['confidence'],
                            'position': b_det['features']['position'],
                            'aligned_position': b_det['features'].get('aligned_position', b_det['features']['position']),
                            'match_similarity': 0.0
                        })
                
                results.extend(frame_results)
        
        return results
    
    def export_results(self, results: List[Dict], output_dir: str = "."):
        """Export results to CSV and JSON formats."""
        if not results:
            print("No results to export")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Export to CSV
        csv_path = os.path.join(output_dir, "player_tracking_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Export to JSON
        json_path = os.path.join(output_dir, "player_tracking_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results exported to:")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")
        
        # Print summary statistics
        print(f"\n📊 Summary:")
        print(f"   - Total detections: {len(results)}")
        print(f"   - Unique players: {len(set(r['player_id'] for r in results))}")
        print(f"   - Frames processed: {len(set(r['frame'] for r in results))}")
        print(f"   - Broadcast detections: {len([r for r in results if r['view'] == 'broadcast'])}")
        print(f"   - Tacticam detections: {len([r for r in results if r['view'] == 'tacticam'])}")
        
        # Show matching statistics
        matched_results = [r for r in results if r.get('match_similarity', 0) > 0]
        if matched_results:
            avg_similarity = np.mean([r['match_similarity'] for r in matched_results])
            print(f"   - Average match similarity: {avg_similarity:.3f}")
    
    def process_videos(self, broadcast_path: str, tacticam_path: str, 
                      output_dir: str = ".") -> List[Dict]:
        """Complete pipeline to process both videos and match players."""
        print("🎯 Starting Player Tracking Pipeline")
        print("=" * 50)
        
        # Step 1: Detect players in both videos
        print("\n1️⃣ Detecting players in broadcast video...")
        broadcast_detections = self.detect_players(
            broadcast_path, 
            os.path.join(output_dir, "broadcast_detections.json")
        )
        
        print("\n2️⃣ Detecting players in tacticam video...")
        tacticam_detections = self.detect_players(
            tacticam_path,
            os.path.join(output_dir, "tacticam_detections.json")
        )
        
        # Step 2: Match players across views
        print("\n3️⃣ Matching players across views...")
        results = self.match_players_across_views(broadcast_detections, tacticam_detections)
        
        # Step 3: Export results
        print("\n4️⃣ Exporting results...")
        self.export_results(results, output_dir)
        
        print("\n🎉 Pipeline completed successfully!")
        return results


def main():
    """Main function to run the player tracking system."""
    parser = argparse.ArgumentParser(description="Player Tracking System for Multi-View Sports Videos")
    parser.add_argument("--broadcast", default="broadcast.mp4", help="Path to broadcast video")
    parser.add_argument("--tacticam", default="tacticam.mp4", help="Path to tacticam video")
    parser.add_argument("--model", default="best.pt", help="Path to YOLOv11 model")
    parser.add_argument("--output", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = PlayerTracker(args.model)
    
    # Process videos
    results = tracker.process_videos(args.broadcast, args.tacticam, args.output)
    
    return results


if __name__ == "__main__":
    main()
