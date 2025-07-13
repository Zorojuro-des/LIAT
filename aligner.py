import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def load_relative_positions(json_path):
    """Load player relative positions from JSON detections."""
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
    """
    Align src_pts (View B) to dst_pts (View A) even if lengths differ.
    Uses centroids and PCA-based orientation alignment.
    """
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    # Center the points
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    # PCA to find dominant direction
    def principal_axis(X):
        U, S, Vt = np.linalg.svd(X)
        return Vt[0]  # first principal component

    src_axis = principal_axis(src_centered)
    dst_axis = principal_axis(dst_centered)

    # Compute rotation angle between principal directions
    angle = np.arctan2(dst_axis[1], dst_axis[0]) - np.arctan2(src_axis[1], src_axis[0])
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    # Apply rotation and translate
    aligned_src = np.dot(src_centered, R.T) + dst_mean
    return aligned_src

def assign_ids_nearest(view_a_pts, view_b_pts_aligned, max_dist=0.1):
    """Match aligned B to A using nearest neighbor via Hungarian algorithm."""
    n, m = len(view_a_pts), len(view_b_pts_aligned)
    dist_matrix = np.linalg.norm(
        view_a_pts[:, None, :] - view_b_pts_aligned[None, :, :], axis=2
    )

    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if dist_matrix[i, j] < max_dist:
            matches.append((i, j))

    return matches


def match_frame(view_a_frame_pts, view_b_frame_pts, threshold=0.1):
    """Align and match a single frame of View A and View B."""
    if len(view_a_frame_pts) < 2 or len(view_b_frame_pts) < 2:
        return [], None

    view_a_pts = np.array(view_a_frame_pts)
    view_b_pts = np.array(view_b_frame_pts)

    view_b_aligned = procrustes_align_partial(view_b_pts, view_a_pts)
    matches = assign_ids_nearest(view_a_pts, view_b_aligned, max_dist=threshold)

    return matches, view_b_aligned


def visualize_alignment(view_a_pts, view_b_pts_aligned, matches, frame_idx):
    """Draw the alignment and match visualization for a single frame."""
    plt.figure(figsize=(8, 6))
    plt.scatter(*view_a_pts.T, c='blue', label='View A')
    plt.scatter(*view_b_pts_aligned.T, c='red', marker='x', label='View B (Aligned)')

    for a_idx, b_idx in matches:
        a = view_a_pts[a_idx]
        b = view_b_pts_aligned[b_idx]
        plt.plot([a[0], b[0]], [a[1], b[1]], 'gray', linestyle='--', alpha=0.6)

    plt.legend()
    plt.title(f"Frame {frame_idx} - Player Alignment")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    view_a_positions = load_relative_positions("detections_view_a.json")
    view_b_positions = load_relative_positions("detections_view_b.json")

    # Iterate over frames
    for frame_idx in range(min(len(view_a_positions), len(view_b_positions))):
        a_pts = view_a_positions[frame_idx]
        b_pts = view_b_positions[frame_idx]

        matches, b_aligned = match_frame(a_pts, b_pts)

        print(f"Frame {frame_idx}: matched {len(matches)} players")

        # if matches and b_aligned is not None:
        #     visualize_alignment(np.array(a_pts), b_aligned, matches, frame_idx)

        # Uncomment to only visualize first few
        # if frame_idx > 10:
        #     break


if __name__ == "__main__":
    main()
