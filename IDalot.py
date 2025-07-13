import json
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

class PlayerTracker:
    """Enhanced player tracker with temporal consistency."""
    
    def __init__(self, max_missing_frames: int = 5, position_threshold: float = 0.15):
        self.max_missing_frames = max_missing_frames
        self.position_threshold = position_threshold
        self.global_id_counter = 0
        self.active_tracks = {}  # track_id -> track_info
        self.track_history = defaultdict(deque)  # track_id -> deque of positions
        
    def predict_position(self, track_id: int, current_frame: int) -> Optional[List[float]]:
        """Predict next position based on track history."""
        if track_id not in self.track_history:
            return None
            
        history = self.track_history[track_id]
        if len(history) < 2:
            return list(history[-1]['position']) if history else None
            
        # Use last two positions to predict next position
        last_pos = np.array(history[-1]['position'])
        second_last_pos = np.array(history[-2]['position'])
        
        # Calculate velocity
        velocity = last_pos - second_last_pos
        
        # Predict next position
        predicted_pos = last_pos + velocity
        
        return predicted_pos.tolist()
    
    def calculate_track_confidence(self, track_id: int) -> float:
        """Calculate confidence score for a track based on its history."""
        if track_id not in self.track_history:
            return 0.0
            
        history = self.track_history[track_id]
        if len(history) < 2:
            return 0.5
            
        # Calculate motion consistency
        positions = [np.array(h['position']) for h in history]
        velocities = [positions[i] - positions[i-1] for i in range(1, len(positions))]
        
        if len(velocities) < 2:
            return 0.7
            
        # Calculate velocity consistency (lower variance = higher confidence)
        velocity_magnitudes = [np.linalg.norm(v) for v in velocities]
        velocity_consistency = 1.0 / (1.0 + np.var(velocity_magnitudes))
        
        # Track length bonus
        length_bonus = min(len(history) / 10.0, 1.0)
        
        return 0.5 * velocity_consistency + 0.5 * length_bonus
    
    def update_track(self, track_id: int, position: List[float], frame_idx: int):
        """Update track with new position."""
        self.active_tracks[track_id] = {
            'last_position': position,
            'last_frame': frame_idx,
            'frames_since_update': 0
        }
        
        # Add to history (keep last 10 positions)
        self.track_history[track_id].append({
            'position': position,
            'frame': frame_idx
        })
        
        # Keep only recent history
        if len(self.track_history[track_id]) > 10:
            self.track_history[track_id].popleft()
    
    def cleanup_inactive_tracks(self, current_frame: int):
        """Remove tracks that have been inactive for too long."""
        inactive_tracks = []
        
        for track_id, track_info in self.active_tracks.items():
            frames_since_update = current_frame - track_info['last_frame']
            if frames_since_update > self.max_missing_frames:
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            del self.active_tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]

def load_relative_positions(json_path):
    """Load relative positions from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    all_positions = []
    for frame_data in data:
        positions = [
            det["relative_position"]
            for det in frame_data["detections"]
            if det["class"] == "player"
        ]
        all_positions.append(positions)
    return all_positions


def procrustes_align_partial(src_pts, dst_pts):
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    if len(src_pts) < 2 or len(dst_pts) < 2:
        return src_pts  # not enough points to align

    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    def principal_axis(X):
        U, _, Vt = np.linalg.svd(X)
        return Vt[0]

    src_axis = principal_axis(src_centered)
    dst_axis = principal_axis(dst_centered)

    angle = np.arctan2(dst_axis[1], dst_axis[0]) - np.arctan2(src_axis[1], src_axis[0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    aligned_src = np.dot(src_centered, R.T) + dst_mean
    return aligned_src


def match_points(current_pts, tracked_pts, max_dist=0.1):
    if len(current_pts) == 0 or len(tracked_pts) == 0:
        return [-1] * len(current_pts)

    current_pts = np.array(current_pts)
    tracked_pts = np.array(tracked_pts)

    dist_matrix = np.linalg.norm(current_pts[:, None, :] - tracked_pts[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matched = [-1] * len(current_pts)
    for i, j in zip(row_ind, col_ind):
        if dist_matrix[i, j] < max_dist:
            matched[i] = j
    return matched


def match_and_label(view_a_positions, view_b_positions):
    global_id_counter = 0
    tracked_players = {}  # global_id: position
    tracked_ids = []      # list of current global IDs
    tracked_positions = []  # list of current positions

    csv_rows = []
    json_rows = []

    for frame_idx in range(min(len(view_a_positions), len(view_b_positions))):
        a_pts = view_a_positions[frame_idx]
        b_pts = view_b_positions[frame_idx]
        if len(a_pts) == 0 or len(b_pts) == 0:
            continue  # skip empty frames
        # Match View A to tracked players (temporal matching)
        match_ids = match_points(a_pts, tracked_positions, max_dist=0.1)
        ids_a = []
        for i, match_idx in enumerate(match_ids):
            if match_idx == -1:
                ids_a.append(global_id_counter)
                tracked_ids.append(global_id_counter)
                tracked_positions.append(a_pts[i])
                global_id_counter += 1
            else:
                ids_a.append(tracked_ids[match_idx])
                tracked_positions[match_idx] = a_pts[i]  # update position

        # Align B to A and match spatially
        b_aligned = procrustes_align_partial(b_pts, a_pts)
        if len(a_pts) == 0 or len(b_aligned) == 0:
            continue  # extra safety
        dist_matrix = np.linalg.norm(np.array(a_pts)[:, None, :] - np.array(b_aligned)[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        ids_b = [-1] * len(b_pts)
        used = set()
        for i, j in zip(row_ind, col_ind):
            if dist_matrix[i, j] < 0.1 and j not in used:
                ids_b[j] = ids_a[i]
                used.add(j)

        # Assign new IDs to unmatched B players
        for i in range(len(ids_b)):
            if ids_b[i] == -1:
                ids_b[i] = global_id_counter
                tracked_ids.append(global_id_counter)
                tracked_positions.append(b_pts[i])
                global_id_counter += 1

        # Log output
        for i, (gid, pos) in enumerate(zip(ids_a, a_pts)):
            csv_rows.append([frame_idx, "A", i, gid, pos[0], pos[1]])
            json_rows.append({
                "frame": frame_idx,
                "view": "A",
                "player_index": i,
                "global_id": gid,
                "relative_position": pos
            })

        for i, (gid, pos) in enumerate(zip(ids_b, b_pts)):
            csv_rows.append([frame_idx, "B", i, gid, pos[0], pos[1]])
            json_rows.append({
                "frame": frame_idx,
                "view": "B",
                "player_index": i,
                "global_id": gid,
                "relative_position": pos
            })

    return csv_rows, json_rows


def main():
    view_a_positions = load_relative_positions("detections_view_a.json")
    view_b_positions = load_relative_positions("detections_view_b.json")

    csv_data, json_data = match_and_label(view_a_positions, view_b_positions)

    # Write CSV
    with open("player_ids.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "view", "player_idx", "global_id", "relative_x", "relative_y"])
        writer.writerows(csv_data)

    # Write JSON
    with open("player_ids.json", "w") as f:
        json.dump(json_data, f, indent=2)

    print("✅ Exported: player_ids.csv and player_ids.json")


if __name__ == "__main__":
    main()
