"""
estimate_poses.py
-----------------
Stage 2 of the video-to-3D pipeline.

Runs COLMAP Structure-from-Motion on the extracted keyframes to recover:
  - Camera intrinsics (focal length, principal point)
  - Per-frame camera extrinsics [R | t]  (world-to-camera 4×4 pose)
  - A sparse 3D point cloud (COLMAP reconstruction)

All results are exported to a clean poses.json so downstream stages
don't need to parse COLMAP's binary format directly.

Outputs
-------
outputs/colmap/          : raw COLMAP workspace (sparse/, database.db, …)
outputs/poses.json       : {intrinsics K, per-frame [R|t] as flat list}
outputs/viz/             : sparse point cloud visualisation (Open3D screenshot)
                           + camera trajectory plot

Usage
-----
    python scripts/estimate_poses.py \
        --frames_dir outputs/frames/ \
        --output_dir outputs/ \
        --camera_model SIMPLE_RADIAL \
        --viz

Requirements
------------
    pip install pycolmap open3d matplotlib numpy
    (or use system COLMAP binary — this script supports both)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# COLMAP wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_colmap_binary(frames_dir: str, colmap_dir: str,
                      camera_model: str = "SIMPLE_RADIAL") -> bool:
    """
    Run COLMAP via the command-line binary (most compatible).
    Returns True on success.
    """
    db_path      = os.path.join(colmap_dir, "database.db")
    sparse_dir   = os.path.join(colmap_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    colmap_bin = shutil.which("colmap")
    if colmap_bin is None:
        print("[WARN] 'colmap' binary not found in PATH.")
        print("       Install with:  sudo apt install colmap")
        print("       Or via conda:  conda install -c conda-forge colmap")
        return False

    print(f"[INFO] Using COLMAP binary: {colmap_bin}")

    # Step 1: Feature extraction
    print("\n[COLMAP] Step 1/3 — Feature extraction...")
    ret = subprocess.run([
        colmap_bin, "feature_extractor",
        "--database_path",   db_path,
        "--image_path",      frames_dir,
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",   # assume one camera for whole video
        "--SiftExtraction.use_gpu", "1",
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] Feature extraction failed:\n{ret.stderr[-2000:]}")
        return False
    print("        Done.")

    # Step 2: Sequential matching (video frames → sequential is much faster than exhaustive)
    print("[COLMAP] Step 2/3 — Sequential feature matching...")
    ret = subprocess.run([
        colmap_bin, "sequential_matcher",
        "--database_path", db_path,
        "--SequentialMatching.overlap", "10",   # match each frame with ±10 neighbours
        "--SiftMatching.use_gpu", "1",
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] Matching failed:\n{ret.stderr[-2000:]}")
        return False
    print("        Done.")

    # Step 3: Sparse reconstruction (incremental SfM)
    print("[COLMAP] Step 3/3 — Sparse reconstruction (SfM)...")
    ret = subprocess.run([
        colmap_bin, "mapper",
        "--database_path", db_path,
        "--image_path",    frames_dir,
        "--output_path",   sparse_dir,
        "--Mapper.num_threads", "8",
    ], capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"[ERROR] Mapper failed:\n{ret.stderr[-2000:]}")
        return False
    print("        Done.")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Parse COLMAP binary output
# ─────────────────────────────────────────────────────────────────────────────

def parse_colmap_cameras_bin(path: str) -> dict:
    """Parse COLMAP cameras.bin → dict of camera_id → {model, params, w, h}"""
    import struct
    cameras = {}
    with open(path, "rb") as f:
        n_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_cameras):
            cam_id  = struct.unpack("<i", f.read(4))[0]
            model   = struct.unpack("<i", f.read(4))[0]
            width   = struct.unpack("<Q", f.read(8))[0]
            height  = struct.unpack("<Q", f.read(8))[0]
            # Number of params depends on model
            # SIMPLE_PINHOLE=1, PINHOLE=2, SIMPLE_RADIAL=3, RADIAL=4 params
            model_params = {0: 3, 1: 1, 2: 4, 3: 2, 4: 4, 5: 5}
            n_params = model_params.get(model, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {
                "model": model, "width": int(width), "height": int(height),
                "params": list(params)
            }
    return cameras


def parse_colmap_images_bin(path: str) -> dict:
    """Parse COLMAP images.bin → dict of image_id → {name, qvec, tvec, camera_id}"""
    import struct
    images = {}
    with open(path, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            image_id  = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))   # quaternion (qw, qx, qy, qz)
            tvec = struct.unpack("<3d", f.read(24))   # translation
            cam_id = struct.unpack("<i", f.read(4))[0]
            # Read name (null-terminated string)
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            # Skip 2D points
            n_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n_points2d)  # each point2d is (x, y, point3d_id) = 3×8 bytes
            images[image_id] = {
                "name":      name.decode("utf-8"),
                "qvec":      list(qvec),
                "tvec":      list(tvec),
                "camera_id": cam_id,
            }
    return images


def parse_colmap_points_bin(path: str) -> np.ndarray:
    """Parse COLMAP points3D.bin → Nx6 array (x,y,z,r,g,b)"""
    import struct
    points = []
    with open(path, "rb") as f:
        n_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_points):
            f.read(8)   # point3d_id
            xyz  = struct.unpack("<3d", f.read(24))
            rgb  = struct.unpack("<3B", f.read(3))
            f.read(8)   # error (reprojection)
            n_tracks = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * n_tracks)  # track elements
            points.append(list(xyz) + list(rgb))
    return np.array(points, dtype=np.float64) if points else np.zeros((0, 6))


def qvec_to_rotmat(qvec) -> np.ndarray:
    """Convert quaternion [qw, qx, qy, qz] to 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),    2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),    1 - 2*(qx**2 + qy**2)],
    ])
    return R


def build_intrinsics_matrix(camera: dict) -> np.ndarray:
    """
    Build 3×3 intrinsics matrix K from COLMAP camera params.
    Supports SIMPLE_RADIAL, SIMPLE_PINHOLE, PINHOLE models.
    """
    params = camera["params"]
    model  = camera["model"]
    if model == 0:    # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    elif model == 1:  # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    elif model == 2:  # SIMPLE_RADIAL: f, cx, cy, k
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    else:             # Fallback: use first param as f, use image centre
        f = params[0]
        cx, cy = camera["width"] / 2.0, camera["height"] / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    return K


def export_poses_json(colmap_sparse_dir: str, output_path: str) -> dict:
    """
    Parse COLMAP sparse reconstruction and export clean poses.json.
    Returns the poses dict.
    """
    # Find the first (and usually only) reconstruction subfolder: 0/, 1/, etc.
    recon_dirs = sorted([
        d for d in os.listdir(colmap_sparse_dir)
        if os.path.isdir(os.path.join(colmap_sparse_dir, d))
    ])
    if not recon_dirs:
        raise RuntimeError(f"No COLMAP reconstruction found in {colmap_sparse_dir}")

    recon_dir = os.path.join(colmap_sparse_dir, recon_dirs[0])
    print(f"[INFO] Parsing COLMAP reconstruction: {recon_dir}/")

    cameras_bin = os.path.join(recon_dir, "cameras.bin")
    images_bin  = os.path.join(recon_dir, "images.bin")
    points_bin  = os.path.join(recon_dir, "points3D.bin")

    cameras = parse_colmap_cameras_bin(cameras_bin)
    images  = parse_colmap_images_bin(images_bin)
    points  = parse_colmap_points_bin(points_bin)

    print(f"       Cameras : {len(cameras)}")
    print(f"       Images  : {len(images)}")
    print(f"       3D pts  : {len(points)}")

    # Build intrinsics for the first (shared) camera
    cam0 = cameras[list(cameras.keys())[0]]
    K    = build_intrinsics_matrix(cam0)

    # Build per-frame pose list, sorted by image name
    frame_poses = []
    for img_id, img_data in sorted(images.items(), key=lambda x: x[1]["name"]):
        R = qvec_to_rotmat(img_data["qvec"])
        t = np.array(img_data["tvec"])

        # COLMAP gives world-to-camera transform.
        # We store both W2C (for projection) and C2W (camera centre in world)
        W2C = np.eye(4)
        W2C[:3, :3] = R
        W2C[:3,  3] = t

        C2W = np.linalg.inv(W2C)
        camera_centre_world = C2W[:3, 3].tolist()

        frame_poses.append({
            "name":                  img_data["name"],
            "camera_id":             img_data["camera_id"],
            "W2C":                   W2C.flatten().tolist(),   # 16 floats
            "C2W":                   C2W.flatten().tolist(),   # 16 floats
            "camera_centre_world":   camera_centre_world,
        })

    poses_data = {
        "n_frames":        len(frame_poses),
        "image_width":     cam0["width"],
        "image_height":    cam0["height"],
        "intrinsics_K":    K.flatten().tolist(),      # 9 floats: [fx,0,cx,0,fy,cy,0,0,1]
        "colmap_model":    cam0["model"],
        "colmap_params":   cam0["params"],
        "frames":          frame_poses,
        "sparse_points":   points.tolist(),            # for scale alignment later
    }

    with open(output_path, "w") as f:
        json.dump(poses_data, f, indent=2)

    print(f"[OK] poses.json → {output_path}")
    return poses_data


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def visualise_sparse_cloud(poses_data: dict, viz_dir: str) -> None:
    """
    Save a matplotlib scatter plot of:
      - Sparse 3D point cloud (gray dots)
      - Camera centres (blue dots connected by trajectory line)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        points = np.array(poses_data.get("sparse_points", []))
        frames = poses_data["frames"]
        centres = np.array([f["camera_centre_world"] for f in frames])

        fig = plt.figure(figsize=(12, 5))

        # Top-down view
        ax1 = fig.add_subplot(121)
        if len(points) > 0:
            pts = points[::max(1, len(points)//5000)]  # subsample for speed
            ax1.scatter(pts[:, 0], pts[:, 2], s=0.5, c="#cccccc", alpha=0.5)
        ax1.plot(centres[:, 0], centres[:, 2], "-o", color="#378ADD",
                 markersize=3, linewidth=1.0, label="camera trajectory")
        ax1.scatter(centres[0, 0], centres[0, 2], c="green", s=50, zorder=5,
                    label="start")
        ax1.scatter(centres[-1, 0], centres[-1, 2], c="red", s=50, zorder=5,
                    label="end")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.set_title("Top-down view (X-Z plane)")
        ax1.legend(fontsize=7)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # Side view
        ax2 = fig.add_subplot(122)
        if len(points) > 0:
            ax2.scatter(pts[:, 0], pts[:, 1], s=0.5, c="#cccccc", alpha=0.5)
        ax2.plot(centres[:, 0], centres[:, 1], "-o", color="#378ADD",
                 markersize=3, linewidth=1.0)
        ax2.scatter(centres[0, 0], centres[0, 1], c="green", s=50, zorder=5)
        ax2.scatter(centres[-1, 0], centres[-1, 1], c="red", s=50, zorder=5)
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_title("Side view (X-Y plane)")
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"COLMAP sparse reconstruction  |  "
                     f"{len(frames)} cameras  |  {len(points)} points",
                     fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(viz_dir, "sparse_reconstruction.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[VIZ] Sparse reconstruction plot → {out_path}")

    except Exception as e:
        print(f"[WARN] Visualisation failed: {e}")


def print_pose_summary(poses_data: dict) -> None:
    """Print a human-readable summary of the recovered poses."""
    K = np.array(poses_data["intrinsics_K"]).reshape(3, 3)
    frames = poses_data["frames"]
    centres = np.array([f["camera_centre_world"] for f in frames])

    print("\n=== Pose Summary ===")
    print(f"  Frames registered  : {len(frames)} / {poses_data['n_frames']}")
    print(f"  Image size         : {poses_data['image_width']} × {poses_data['image_height']}")
    print(f"  Focal length fx    : {K[0,0]:.1f} px")
    print(f"  Focal length fy    : {K[1,1]:.1f} px")
    print(f"  Principal point    : ({K[0,2]:.1f}, {K[1,2]:.1f})")

    if len(centres) > 1:
        # Estimate scene scale from camera baseline
        trajectory_len = np.sum(np.linalg.norm(np.diff(centres, axis=0), axis=1))
        print(f"  Camera path length : {trajectory_len:.3f} (COLMAP units)")
        bbox = centres.max(axis=0) - centres.min(axis=0)
        print(f"  Camera bbox        : {bbox[0]:.3f} × {bbox[1]:.3f} × {bbox[2]:.3f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def estimate_poses(
    frames_dir: str,
    output_dir: str,
    camera_model: str = "SIMPLE_RADIAL",
    viz: bool = True,
    skip_if_exists: bool = True,
) -> dict:

    colmap_dir = os.path.join(output_dir, "colmap")
    viz_dir    = os.path.join(output_dir, "viz")
    poses_path = os.path.join(output_dir, "poses.json")
    os.makedirs(colmap_dir, exist_ok=True)
    if viz:
        os.makedirs(viz_dir, exist_ok=True)

    sparse_dir = os.path.join(colmap_dir, "sparse")

    # Skip COLMAP if we already have a reconstruction (for re-runs)
    recon_exists = (
        os.path.isdir(sparse_dir) and
        any(os.path.isdir(os.path.join(sparse_dir, d))
            for d in os.listdir(sparse_dir))
    )
    if skip_if_exists and recon_exists and os.path.isfile(poses_path):
        print("[INFO] COLMAP reconstruction already exists — skipping rerun.")
        print("       Delete outputs/colmap/ to force re-reconstruction.")
        with open(poses_path) as f:
            return json.load(f)

    # Run COLMAP
    success = run_colmap_binary(frames_dir, colmap_dir, camera_model)
    if not success:
        print("\n[ERROR] COLMAP failed. Possible fixes:")
        print("  1. Install COLMAP:     sudo apt install colmap")
        print("  2. Reduce frame count: lower --max_frames in extract_frames.py")
        print("  3. Check frame quality: open outputs/viz/keyframes_contact_sheet.png")
        sys.exit(1)

    # Parse results → poses.json
    poses_data = export_poses_json(sparse_dir, poses_path)
    print_pose_summary(poses_data)

    # Visualise
    if viz:
        visualise_sparse_cloud(poses_data, viz_dir)

    return poses_data


def parse_args():
    p = argparse.ArgumentParser(description="Run COLMAP SfM on extracted frames.")
    p.add_argument("--frames_dir",    required=True,
                   help="Directory containing keyframe JPEGs")
    p.add_argument("--output_dir",    default="outputs",
                   help="Root output directory (default: outputs/)")
    p.add_argument("--camera_model",  default="SIMPLE_RADIAL",
                   choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL"],
                   help="COLMAP camera model (default: SIMPLE_RADIAL)")
    p.add_argument("--viz",           action="store_true", default=True)
    p.add_argument("--no-viz",        action="store_false", dest="viz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"[ERROR] Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    n_frames = len([f for f in os.listdir(args.frames_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"[INFO] Found {n_frames} frames in {args.frames_dir}")
    if n_frames < 10:
        print("[WARN] Very few frames — COLMAP may struggle. "
              "Consider lowering --flow_threshold in extract_frames.py")

    poses = estimate_poses(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        camera_model=args.camera_model,
        viz=args.viz,
    )

    print("Next step:")
    print("  python scripts/estimate_depth.py "
          "--frames_dir outputs/frames/ --output_dir outputs/")
