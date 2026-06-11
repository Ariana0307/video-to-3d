"""
fuse_pointcloud.py
------------------
Stage 4 of the video-to-3D pipeline.

Fuses per-frame metric depth maps and RGB images into a single 3D mesh using
TSDF (Truncated Signed Distance Function) volumetric integration, implemented
via Open3D's ScalableTSDFVolume.

Key design choices:
  - Uses Open3D's HASH-MAP based sparse TSDF (not a dense grid) → memory efficient
  - Applies a scale alignment step between Depth Pro metric depths and
    COLMAP poses (they may have different absolute scales)
  - Marching cubes extracts a clean, coloured triangle mesh
  - Outputs both a dense mesh (.ply) and a coloured point cloud (.ply)

Outputs
-------
outputs/mesh.ply              : coloured triangle mesh (view in MeshLab / Open3D)
outputs/pointcloud.ply        : coloured point cloud  (view in CloudCompare)
outputs/viz/reconstruction.png: rendered screenshot of the mesh

Usage
-----
    python scripts/fuse_pointcloud.py \
        --output_dir outputs/ \
        --voxel_length 0.03 \
        --sdf_trunc 0.12 \
        --viz

Requirements
------------
    pip install open3d numpy
"""

import argparse
import json
import os
import sys
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Scale alignment
# ─────────────────────────────────────────────────────────────────────────────

def compute_scale_factor(poses_data: dict, depth_infos: list,
                          frames_dir: str) -> float:
    """
    Align the metric scale of Depth Pro depths to COLMAP's coordinate system.

    Strategy: COLMAP produces a sparse 3D point cloud. For each sparse point,
    we know which frames see it (via reprojection). We compare:
      - The depth from COLMAP's sparse cloud (ground truth geometry)
      - The depth from Depth Pro for the same pixel

    The ratio gives a global scale factor.

    If the sparse cloud is too small (< 20 points), falls back to 1.0
    (no scale correction), which is still valid since Depth Pro is metric.
    """
    sparse_pts = np.array(poses_data.get("sparse_points", []))

    if len(sparse_pts) < 20:
        print("[WARN] Sparse point cloud too small for scale alignment — using scale=1.0")
        print("       Depth Pro metric depths will be used as-is.")
        return 1.0

    frames = poses_data["frames"]

    K_flat = poses_data["intrinsics_K"]
    K = np.array(K_flat).reshape(3, 3)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    W = poses_data["image_width"]
    H = poses_data["image_height"]

    # Build a lookup from frame name to depth info
    depth_lookup = {d["filename"]: d for d in depth_infos}

    ratios = []
    for frame in frames[:30]:  # check first 30 frames for speed
        name = frame["name"]
        if name not in depth_lookup:
            continue
        depth_file = depth_lookup[name]["depth_file"]
        depth_path = os.path.join(os.path.dirname(
            depth_infos[0].get("_depth_dir", "")), depth_file)
        if not os.path.isfile(depth_path):
            continue

        W2C = np.array(frame["W2C"]).reshape(4, 4)
        depth_map = np.load(depth_path)

        for pt in sparse_pts[::max(1, len(sparse_pts)//200)]:
            xyz_world = pt[:3]
            xyz_cam   = (W2C[:3, :3] @ xyz_world) + W2C[:3, 3]
            if xyz_cam[2] <= 0:
                continue
            # Project to image
            u = fx * xyz_cam[0] / xyz_cam[2] + cx
            v = fy * xyz_cam[1] / xyz_cam[2] + cy
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            depth_colmap = xyz_cam[2]
            depth_depthpro = float(depth_map[vi, ui])
            if depth_depthpro > 0.1:
                ratios.append(depth_colmap / depth_depthpro)

    if not ratios:
        print("[WARN] Could not compute scale ratio — using 1.0")
        return 1.0

    # Robust median
    scale = float(np.median(ratios))
    print(f"[INFO] Scale alignment: {len(ratios)} correspondences, "
          f"scale factor = {scale:.4f}  "
          f"(std={np.std(ratios):.4f}, "
          f"range [{min(ratios):.3f}, {max(ratios):.3f}])")

    # Sanity check: if scale is wildly off, fall back
    if scale < 0.01 or scale > 100:
        print(f"[WARN] Scale factor {scale:.4f} looks unreliable — using 1.0")
        return 1.0

    return scale


# ─────────────────────────────────────────────────────────────────────────────
# TSDF Fusion
# ─────────────────────────────────────────────────────────────────────────────

def fuse_tsdf(
    poses_data: dict,
    depth_infos: list,
    frames_dir: str,
    depth_dir: str,
    output_dir: str,
    voxel_length: float = 0.03,
    sdf_trunc: float = 0.12,
    depth_scale: float = 1000.0,   # Open3D expects depth in mm if scale=1000
    depth_max: float = 5.0,        # clip depths beyond this (metres)
    scale_factor: float = 1.0,
) -> object:
    """
    Integrate all frames into a TSDF volume and extract a mesh.

    Parameters
    ----------
    poses_data   : loaded poses.json
    depth_infos  : loaded depth_info.json frames list
    frames_dir   : directory of RGB JPEG frames
    depth_dir    : directory of .npy depth maps
    output_dir   : root output directory
    voxel_length : TSDF voxel size in metres (3cm for indoor)
    sdf_trunc    : truncation distance (typically 4–5× voxel_length)
    depth_scale  : Open3D internal depth unit (keep at 1000)
    depth_max    : ignore depth values beyond this many metres
    scale_factor : Depth Pro → COLMAP scale correction

    Returns
    -------
    Open3D TriangleMesh
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[ERROR] open3d not installed.  pip install open3d")
        sys.exit(1)

    K_flat = poses_data["intrinsics_K"]
    K = np.array(K_flat).reshape(3, 3)
    W = poses_data["image_width"]
    H = poses_data["image_height"]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H,
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2]
    )

    # Scalable TSDF: sparse hash-map of voxel blocks
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    # Build lookup: frame name → depth info
    depth_lookup = {d["filename"]: d for d in depth_infos}
    frames       = poses_data["frames"]

    print(f"[INFO] TSDF fusion: voxel={voxel_length*100:.1f}cm  "
          f"trunc={sdf_trunc*100:.1f}cm  scale={scale_factor:.4f}")
    print(f"[INFO] Integrating {len(frames)} frames...")

    integrated = 0
    skipped    = 0

    for i, frame in enumerate(frames):
        fname = frame["name"]

        # Load RGB
        rgb_path = os.path.join(frames_dir, fname)
        if not os.path.isfile(rgb_path):
            skipped += 1
            continue

        # Load depth
        if fname not in depth_lookup:
            skipped += 1
            continue
        stem       = os.path.splitext(fname)[0]
        depth_path = os.path.join(depth_dir, f"{stem}.npy")
        if not os.path.isfile(depth_path):
            skipped += 1
            continue

        depth_m = np.load(depth_path).astype(np.float32)

        # Apply scale correction
        depth_m = depth_m * scale_factor

        # Clip invalid depths
        depth_m[depth_m > depth_max] = 0.0
        depth_m[depth_m < 0.1]       = 0.0

        # Convert depth to uint16 in mm (Open3D convention with depth_scale=1000)
        depth_mm = (depth_m * depth_scale).astype(np.uint16)

        # Load as Open3D images
        color_o3d = o3d.io.read_image(rgb_path)
        depth_o3d = o3d.geometry.Image(depth_mm)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_max,
            convert_rgb_to_intensity=False,
        )

        # Camera pose: Open3D integrate expects extrinsic = C2W (camera-to-world)
        # COLMAP gives us W2C, so we use C2W = inv(W2C) — already stored in poses.json
        C2W = np.array(frame["C2W"]).reshape(4, 4)

        volume.integrate(rgbd, intrinsic, np.linalg.inv(C2W))
        integrated += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(frames)}] integrated {integrated} frames...", end="\r")

    print(f"\n[INFO] Integrated: {integrated}  Skipped: {skipped}")

    # Extract mesh via marching cubes
    print("[INFO] Running marching cubes to extract mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    n_verts  = len(np.asarray(mesh.vertices))
    n_tris   = len(np.asarray(mesh.triangles))
    print(f"[INFO] Mesh: {n_verts:,} vertices, {n_tris:,} triangles")

    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud export
# ─────────────────────────────────────────────────────────────────────────────

def mesh_to_pointcloud(mesh, n_points: int = 500_000):
    """Sample a dense coloured point cloud from the mesh surface."""
    try:
        import open3d as o3d
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        return pcd
    except Exception as e:
        print(f"[WARN] Could not sample point cloud: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def render_mesh_screenshot(mesh, output_path: str) -> None:
    """
    Save a headless screenshot of the mesh from multiple viewpoints.
    Uses Open3D's offscreen rendering.
    """
    try:
        import open3d as o3d

        # Try headless rendering (requires Open3D >= 0.13)
        vis = o3d.visualization.rendering.OffscreenRenderer(1280, 720)
        vis.scene.set_background([0.15, 0.15, 0.15, 1.0])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        vis.scene.add_geometry("mesh", mesh, mat)

        # Fit camera to scene
        bounds = mesh.get_axis_aligned_bounding_box()
        vis.setup_camera(60.0, bounds, bounds.get_center())

        img = vis.render_to_image()
        o3d.io.write_image(output_path, img)
        print(f"[VIZ] Mesh screenshot → {output_path}")

    except Exception as e:
        print(f"[WARN] Headless render failed ({e})")
        print("       View mesh manually: python -c \"import open3d as o3d; "
              f"o3d.visualization.draw_geometries([o3d.io.read_triangle_mesh('{output_path.replace('.png', '.ply')}', True)])\"")


def make_reconstruction_summary(mesh, poses_data: dict, output_dir: str) -> None:
    """Print and save a reconstruction quality summary."""
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)

    bbox   = mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound)

    summary = {
        "n_vertices":      int(len(verts)),
        "n_triangles":     int(len(tris)),
        "bbox_metres":     extent.tolist(),
        "volume_m3":       float(extent[0] * extent[1] * extent[2]),
        "has_vertex_colors": mesh.has_vertex_colors(),
        "n_input_frames":  poses_data["n_frames"],
    }

    print("\n=== Reconstruction Summary ===")
    print(f"  Vertices   : {summary['n_vertices']:,}")
    print(f"  Triangles  : {summary['n_triangles']:,}")
    print(f"  Scene bbox : {extent[0]:.2f}m × {extent[1]:.2f}m × {extent[2]:.2f}m")
    print(f"  Has colour : {summary['has_vertex_colors']}")

    with open(os.path.join(output_dir, "reconstruction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_fusion(
    output_dir: str,
    voxel_length: float = 0.03,
    sdf_trunc:    float = None,
    depth_max:    float = 5.0,
    viz: bool = True,
) -> None:

    # Load metadata from previous stages
    poses_path = os.path.join(output_dir, "poses.json")
    depth_info_path = os.path.join(output_dir, "depth_info.json")
    frames_dir = os.path.join(output_dir, "frames")
    depth_dir  = os.path.join(output_dir, "depth")
    viz_dir    = os.path.join(output_dir, "viz")

    if not os.path.isfile(poses_path):
        print(f"[ERROR] poses.json not found: {poses_path}")
        print("        Run estimate_poses.py first.")
        sys.exit(1)
    if not os.path.isfile(depth_info_path):
        print(f"[ERROR] depth_info.json not found: {depth_info_path}")
        print("        Run estimate_depth.py first.")
        sys.exit(1)

    with open(poses_path) as f:
        poses_data = json.load(f)
    with open(depth_info_path) as f:
        depth_data = json.load(f)
    depth_infos = depth_data["frames"]

    # Inject depth_dir into info for scale alignment
    for d in depth_infos:
        d["_depth_dir"] = depth_dir

    os.makedirs(viz_dir, exist_ok=True)

    # Auto-set sdf_trunc if not specified
    if sdf_trunc is None:
        sdf_trunc = voxel_length * 4

    print(f"[INFO] Loaded {len(poses_data['frames'])} poses, "
          f"{len(depth_infos)} depth maps")

    # Scale alignment
    scale_factor = compute_scale_factor(poses_data, depth_infos, frames_dir)

    # Run TSDF fusion
    mesh = fuse_tsdf(
        poses_data=poses_data,
        depth_infos=depth_infos,
        frames_dir=frames_dir,
        depth_dir=depth_dir,
        output_dir=output_dir,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_max=depth_max,
        scale_factor=scale_factor,
    )

    # Save outputs
    mesh_path = os.path.join(output_dir, "mesh.ply")
    pcd_path  = os.path.join(output_dir, "pointcloud.ply")

    try:
        import open3d as o3d
        o3d.io.write_triangle_mesh(mesh_path, mesh, write_vertex_colors=True)
        print(f"[OK] Mesh → {mesh_path}")

        pcd = mesh_to_pointcloud(mesh)
        if pcd is not None:
            o3d.io.write_point_cloud(pcd_path, pcd)
            print(f"[OK] Point cloud → {pcd_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save mesh: {e}")
        sys.exit(1)

    # Summary
    make_reconstruction_summary(mesh, poses_data, output_dir)

    # Visualisations
    if viz:
        render_mesh_screenshot(mesh, os.path.join(viz_dir, "reconstruction.png"))

    print(f"\n=== Done! ===")
    print(f"  Mesh saved to   : {mesh_path}")
    print(f"  View with Open3D:")
    print(f"    python -c \"import open3d as o3d; "
          f"o3d.visualization.draw_geometries(["
          f"o3d.io.read_triangle_mesh('{mesh_path}', True)])\"")
    print(f"  Or open in MeshLab: meshlab {mesh_path}")
    print(f"\nOptional next step:")
    print(f"  python scripts/segment_labels.py --output_dir {output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="TSDF fusion → mesh.")
    p.add_argument("--output_dir",    default="outputs",
                   help="Root output directory (default: outputs/)")
    p.add_argument("--voxel_length",  type=float, default=0.03,
                   help="Voxel size in metres (default: 0.03 = 3cm)")
    p.add_argument("--sdf_trunc",     type=float, default=None,
                   help="TSDF truncation distance (default: 4× voxel_length)")
    p.add_argument("--depth_max",     type=float, default=5.0,
                   help="Maximum depth to integrate in metres (default: 5.0)")
    p.add_argument("--viz",           action="store_true", default=True)
    p.add_argument("--no-viz",        action="store_false", dest="viz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fusion(
        output_dir=args.output_dir,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        depth_max=args.depth_max,
        viz=args.viz,
    )
