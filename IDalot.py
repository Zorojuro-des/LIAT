import json
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_relative_positions(json_path):
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
