#!/usr/bin/env python3
"""
Enhanced Temporal Player Tracking System

This system improves upon the basic player tracking by adding temporal consistency
across frames, ensuring that player IDs remain stable over time.
"""

import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import cv2
from collections import defaultdict, deque
import os


class TemporalPlayerTracker:
    def __init__(self, max_missing_frames: int = 5, similarity_threshold: float = 0.4):
        """
        Initialize temporal tracker.
        
        Args:
            max_missing_frames: Maximum frames a player can be missing before being considered lost
            similarity_threshold: Minimum similarity score for matching players
        """
        self.max_missing_frames = max_missing_frames
        self.similarity_threshold = similarity_threshold
        self.global_id_counter = 0
        self.active_tracks = {}  # player_id -> track_info
        self.track_history = defaultdict(list)  # player_id -> list of frame data
        
    def load_detections(self, json_path: str) -> List[Dict]:
        """Load detections from JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def extract_appearance_features(self, detection: Dict) -> np.ndarray:
        """Extract appearance features from detection."""
        features = detection.get('features', {})
        
        # Combine color histograms
        color_hist = np.concatenate([
            features.get('color_hist_h', []),
            features.get('color_hist_s', []),
            features.get('color_hist_v', [])
        ])
        
        # Add texture features
        texture_features = [
            features.get('texture_mean', 0),
            features.get('texture_std', 0)
        ]
        
        # Add size features
        size_features = features.get('size', [0, 0])
        
        # Combine all features
        all_features = np.concatenate([color_hist, texture_features, size_features])
        
        # Normalize
        norm = np.linalg.norm(all_features)
        if norm > 0:
            all_features = all_features / norm
            
        return all_features
    
    def calculate_motion_consistency(self, track_history: List[Dict], 
                                   current_position: List[float]) -> float:
        """Calculate motion consistency score based on trajectory."""
        if len(track_history) < 2:
            return 0.5  # Neutral score for insufficient history
        
        # Get recent positions
        recent_positions = [item['position'] for item in track_history[-3:]]
        recent_positions.append(current_position)
        
        # Calculate velocity vectors
        velocities = []
        for i in range(1, len(recent_positions)):
            vel = np.array(recent_positions[i]) - np.array(recent_positions[i-1])
            velocities.append(vel)
        
        if len(velocities) < 2:
            return 0.5
        
        # Calculate velocity consistency (how similar are the velocity vectors)
        velocity_similarities = []
        for i in range(1, len(velocities)):
            v1, v2 = velocities[i-1], velocities[i]
            # Normalize velocities
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                velocity_similarities.append(max(0, cos_sim))
        
        if velocity_similarities:
            return np.mean(velocity_similarities)
        return 0.5
    
    def match_detections_to_tracks(self, detections: List[Dict], 
                                 frame_idx: int) -> List[Tuple[int, int, float]]:
        """Match current detections to existing tracks."""
        if not detections or not self.active_tracks:
            return []
        
        # Extract features for current detections
        detection_features = []
        detection_positions = []
        
        for det in detections:
            features = self.extract_appearance_features(det)
            position = det.get('features', {}).get('position', [0, 0])
            
            detection_features.append(features)
            detection_positions.append(position)
        
        # Calculate similarity matrix
        track_ids = list(self.active_tracks.keys())
        similarity_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, (det_features, det_pos) in enumerate(zip(detection_features, detection_positions)):
            for j, track_id in enumerate(track_ids):
                track_info = self.active_tracks[track_id]
                track_history = self.track_history[track_id]
                
                # Appearance similarity
                if len(track_info['appearance_features']) > 0:
                    app_sim = cosine_similarity(
                        [det_features], 
                        [track_info['appearance_features']]
                    )[0, 0]
                else:
                    app_sim = 0.0
                
                # Position similarity
                last_pos = track_info['last_position']
                pos_dist = np.linalg.norm(np.array(det_pos) - np.array(last_pos))
                pos_sim = np.exp(-pos_dist * 5)  # Exponential decay
                
                # Motion consistency
                motion_sim = self.calculate_motion_consistency(track_history, det_pos)
                
                # Time gap penalty
                frames_since_last = frame_idx - track_info['last_frame']
                time_penalty = np.exp(-frames_since_last * 0.1)
                
                # Combined similarity
                combined_sim = (0.4 * app_sim + 0.3 * pos_sim + 0.2 * motion_sim + 0.1 * time_penalty)
                similarity_matrix[i, j] = combined_sim
        
        # Use Hungarian algorithm for optimal matching
        if similarity_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-similarity_matrix)  # Negative for maximization
            
            matches = []
            for i, j in zip(row_ind, col_ind):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    matches.append((i, track_ids[j], similarity_matrix[i, j]))
            
            return matches
        
        return []
    
    def update_tracks(self, detections: List[Dict], matches: List[Tuple[int, int, float]], 
                     frame_idx: int) -> List[Dict]:
        """Update existing tracks and create new ones."""
        results = []
        matched_detection_indices = set()
        
        # Update matched tracks
        for det_idx, track_id, similarity in matches:
            matched_detection_indices.add(det_idx)
            detection = detections[det_idx]
            
            # Update track info
            track_info = self.active_tracks[track_id]
            track_info['last_frame'] = frame_idx
            track_info['last_position'] = detection['features']['position']
            track_info['frames_since_detection'] = 0
            
            # Update appearance features (exponential moving average)
            new_features = self.extract_appearance_features(detection)
            if len(track_info['appearance_features']) > 0:
                alpha = 0.2  # Learning rate
                track_info['appearance_features'] = (
                    alpha * new_features + (1 - alpha) * track_info['appearance_features']
                )
            else:
                track_info['appearance_features'] = new_features
            
            # Add to history
            self.track_history[track_id].append({
                'frame': frame_idx,
                'position': detection['features']['position'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            })
            
            # Create result
            result = {
                'frame': frame_idx,
                'player_id': track_id,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'position': detection['features']['position'],
                'match_similarity': similarity,
                'track_length': len(self.track_history[track_id])
            }
            results.append(result)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                # Create new track
                track_id = self.global_id_counter
                self.global_id_counter += 1
                
                self.active_tracks[track_id] = {
                    'last_frame': frame_idx,
                    'last_position': detection['features']['position'],
                    'frames_since_detection': 0,
                    'appearance_features': self.extract_appearance_features(detection),
                    'created_frame': frame_idx
                }
                
                # Add to history
                self.track_history[track_id].append({
                    'frame': frame_idx,
                    'position': detection['features']['position'],
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox']
                })
                
                # Create result
                result = {
                    'frame': frame_idx,
                    'player_id': track_id,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'position': detection['features']['position'],
                    'match_similarity': 0.0,  # New track
                    'track_length': 1
                }
                results.append(result)
        
        return results
    
    def cleanup_inactive_tracks(self, current_frame: int):
        """Remove tracks that have been inactive for too long."""
        inactive_tracks = []
        
        for track_id, track_info in self.active_tracks.items():
            frames_since_last = current_frame - track_info['last_frame']
            if frames_since_last > self.max_missing_frames:
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.active_tracks[track_id]
    
    def track_single_view(self, detections_path: str, view_name: str) -> List[Dict]:
        """Track players in a single view across all frames."""
        detections = self.load_detections(detections_path)
        all_results = []
        
        for frame_data in detections:
            frame_idx = frame_data['frame']
            frame_detections = frame_data['detections']
            
            # Match detections to existing tracks
            matches = self.match_detections_to_tracks(frame_detections, frame_idx)
            
            # Update tracks
            frame_results = self.update_tracks(frame_detections, matches, frame_idx)
            
            # Add view information
            for result in frame_results:
                result['view'] = view_name
            
            all_results.extend(frame_results)
            
            # Cleanup inactive tracks
            self.cleanup_inactive_tracks(frame_idx)
        
        return all_results
    
    def align_cross_view_ids(self, broadcast_results: List[Dict], 
                           tacticam_results: List[Dict]) -> List[Dict]:
        """Align player IDs across two views."""
        # Group results by frame
        broadcast_by_frame = defaultdict(list)
        tacticam_by_frame = defaultdict(list)
        
        for result in broadcast_results:
            broadcast_by_frame[result['frame']].append(result)
        
        for result in tacticam_results:
            tacticam_by_frame[result['frame']].append(result)
        
        # Create ID mapping from tacticam to broadcast
        id_mapping = {}
        aligned_results = []
        
        # Process frames in order
        common_frames = sorted(set(broadcast_by_frame.keys()) & set(tacticam_by_frame.keys()))
        
        for frame_idx in common_frames:
            b_results = broadcast_by_frame[frame_idx]
            t_results = tacticam_by_frame[frame_idx]
            
            if not b_results or not t_results:
                continue
            
            # Calculate position similarity matrix
            b_positions = [r['position'] for r in b_results]
            t_positions = [r['position'] for r in t_results]
            
            # Align tacticam positions to broadcast (simple alignment)
            if len(b_positions) >= 2 and len(t_positions) >= 2:
                # Use centroid alignment
                b_centroid = np.mean(b_positions, axis=0)
                t_centroid = np.mean(t_positions, axis=0)
                
                # Adjust tacticam positions
                t_positions_aligned = [
                    (np.array(pos) - t_centroid + b_centroid).tolist()
                    for pos in t_positions
                ]
            else:
                t_positions_aligned = t_positions
            
            # Calculate distance matrix
            dist_matrix = np.zeros((len(b_results), len(t_results)))
            for i, b_pos in enumerate(b_positions):
                for j, t_pos in enumerate(t_positions_aligned):
                    dist = np.linalg.norm(np.array(b_pos) - np.array(t_pos))
                    dist_matrix[i, j] = dist
            
            # Match using Hungarian algorithm
            if dist_matrix.size > 0:
                row_ind, col_ind = linear_sum_assignment(dist_matrix)
                
                # Create matches with distance threshold
                for i, j in zip(row_ind, col_ind):
                    if dist_matrix[i, j] < 0.3:  # Distance threshold
                        b_id = b_results[i]['player_id']
                        t_id = t_results[j]['player_id']
                        
                        # Update mapping
                        if t_id not in id_mapping:
                            id_mapping[t_id] = b_id
        
        # Apply mapping to tacticam results
        for result in tacticam_results:
            original_id = result['player_id']
            if original_id in id_mapping:
                result['player_id'] = id_mapping[original_id]
                result['original_tacticam_id'] = original_id
        
        # Combine results
        aligned_results = broadcast_results + tacticam_results
        
        return aligned_results
    
    def export_results(self, results: List[Dict], output_dir: str = "."):
        """Export results to CSV and JSON formats."""
        if not results:
            print("No results to export")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Export to CSV
        csv_path = os.path.join(output_dir, "temporal_tracking_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Export to JSON
        json_path = os.path.join(output_dir, "temporal_tracking_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Temporal tracking results exported to:")
        print(f"   - {csv_path}")
        print(f"   - {json_path}")
        
        # Print summary statistics
        print(f"\n📊 Temporal Tracking Summary:")
        print(f"   - Total detections: {len(results)}")
        print(f"   - Unique players: {len(set(r['player_id'] for r in results))}")
        print(f"   - Frames processed: {len(set(r['frame'] for r in results))}")
        print(f"   - Broadcast detections: {len([r for r in results if r['view'] == 'broadcast'])}")
        print(f"   - Tacticam detections: {len([r for r in results if r['view'] == 'tacticam'])}")
        
        # Track length statistics
        track_lengths = defaultdict(int)
        for result in results:
            track_lengths[result['player_id']] += 1
        
        if track_lengths:
            avg_track_length = np.mean(list(track_lengths.values()))
            max_track_length = max(track_lengths.values())
            print(f"   - Average track length: {avg_track_length:.1f} frames")
            print(f"   - Longest track: {max_track_length} frames")


def main():
    """Main function to run temporal tracking."""
    # Initialize tracker
    tracker = TemporalPlayerTracker()
    
    # Track each view separately first
    print("🎯 Starting Temporal Player Tracking")
    print("=" * 50)
    
    print("\n1️⃣ Tracking players in broadcast view...")
    broadcast_results = tracker.track_single_view("broadcast_detections.json", "broadcast")
    
    # Reset tracker for second view
    tracker = TemporalPlayerTracker()
    
    print("\n2️⃣ Tracking players in tacticam view...")
    tacticam_results = tracker.track_single_view("tacticam_detections.json", "tacticam")
    
    print("\n3️⃣ Aligning player IDs across views...")
    aligned_results = tracker.align_cross_view_ids(broadcast_results, tacticam_results)
    
    print("\n4️⃣ Exporting results...")
    tracker.export_results(aligned_results)
    
    print("\n🎉 Temporal tracking completed!")
    
    return aligned_results


if __name__ == "__main__":
    main()
