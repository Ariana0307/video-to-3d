#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Master script: runs the full video-to-3D pipeline end-to-end.

    python run_pipeline.py --video path/to/video.mp4

Stages
------
  1. extract_frames.py  — adaptive keyframe extraction
  2. estimate_poses.py  — COLMAP SfM → camera poses
  3. estimate_depth.py  — Depth Pro → metric depth maps
  4. fuse_pointcloud.py — TSDF fusion → coloured mesh
  [5. segment_labels.py — SAM2 semantic labels (--semantic flag)]

All intermediate results and visualisations are saved to --output_dir.
"""

import argparse
import os
import sys
import time
import subprocess

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "scripts")


def run_stage(script: str, args_list: list, stage_name: str) -> None:
    script_path = os.path.join(SCRIPTS_DIR, script)
    cmd = [sys.executable, script_path] + args_list
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name}")
    print(f"{'='*60}")
    t0 = time.time()
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"\n[ERROR] Stage '{stage_name}' failed (exit code {ret.returncode})")
        sys.exit(ret.returncode)
    print(f"\n[OK] {stage_name} complete in {time.time()-t0:.0f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Video-to-3D full pipeline.")
    p.add_argument("--video",          required=True,   help="Input video path")
    p.add_argument("--output_dir",     default="outputs", help="Output directory")
    p.add_argument("--flow_threshold", type=float, default=3.0)
    p.add_argument("--blur_threshold", type=float, default=80.0)
    p.add_argument("--max_frames",     type=int,   default=150)
    p.add_argument("--voxel_length",   type=float, default=0.03)
    p.add_argument("--depth_max",      type=float, default=5.0)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--semantic",       action="store_true",
                   help="Run SAM2 semantic labelling after reconstruction")
    p.add_argument("--no_viz",         action="store_true",
                   help="Skip visualisation outputs (faster)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    viz_flag = [] if args.no_viz else ["--viz"]
    frames_dir = os.path.join(args.output_dir, "frames")

    t_start = time.time()

    # Stage 1
    run_stage("extract_frames.py", [
        "--video",          args.video,
        "--output_dir",     args.output_dir,
        "--flow_threshold", str(args.flow_threshold),
        "--blur_threshold", str(args.blur_threshold),
        "--max_frames",     str(args.max_frames),
    ] + viz_flag, "Frame extraction")

    # Stage 2
    run_stage("estimate_poses.py", [
        "--frames_dir",  frames_dir,
        "--output_dir",  args.output_dir,
    ] + viz_flag, "COLMAP pose estimation")

    # Stage 3
    run_stage("estimate_depth.py", [
        "--frames_dir",  frames_dir,
        "--output_dir",  args.output_dir,
        "--device",      args.device,
    ] + viz_flag, "Depth Pro inference")

    # Stage 4
    run_stage("fuse_pointcloud.py", [
        "--output_dir",   args.output_dir,
        "--voxel_length", str(args.voxel_length),
        "--depth_max",    str(args.depth_max),
    ] + viz_flag, "TSDF fusion")

    # Stage 5 (optional)
    if args.semantic:
        run_stage("segment_labels.py", [
            "--output_dir",  args.output_dir,
            "--frames_dir",  frames_dir,
            "--device",      args.device,
        ] + viz_flag, "SAM2 semantic labelling")

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {total/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"  Outputs:")
    print(f"    Mesh         : {args.output_dir}/mesh.ply")
    print(f"    Point cloud  : {args.output_dir}/pointcloud.ply")
    print(f"    Viz          : {args.output_dir}/viz/")
    if args.semantic:
        print(f"    Labelled mesh: {args.output_dir}/mesh_labelled.ply")
    print(f"\n  View mesh:")
    print(f"    python -c \"import open3d as o3d; "
          f"o3d.visualization.draw_geometries(["
          f"o3d.io.read_triangle_mesh('{args.output_dir}/mesh.ply', True)])\"")
