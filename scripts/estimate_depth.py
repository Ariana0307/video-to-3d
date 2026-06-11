"""
estimate_depth.py
-----------------
Stage 3 of the video-to-3D pipeline.

Runs Apple's Depth Pro model on each keyframe to produce metric depth maps
(values in real-world metres). This is the "virtual RGB-D sensor" at the
heart of our approach — no stereo rig, no LiDAR, no calibration target.

Depth Pro key properties:
  - Outputs metric depth (absolute metres, not relative)
  - Also estimates per-image focal length (useful if COLMAP didn't converge)
  - Sharp boundaries, good on indoor scenes
  - ~1–3s per frame on a 12GB GPU

Outputs
-------
outputs/depth/           : per-frame depth maps as float32 .npy files
outputs/depth_viz/       : colourised depth PNG for visual inspection
outputs/depth_info.json  : per-frame {filename, min_d, max_d, median_d, focal_px}
outputs/viz/             : summary visualisation grid

Usage
-----
    python scripts/estimate_depth.py \
        --frames_dir outputs/frames/ \
        --output_dir outputs/ \
        --viz

Requirements
------------
    # Install Depth Pro (Apple ML Research):
    pip install git+https://github.com/apple/ml-depth-pro.git
    # Or from the cloned repo:
    cd ml-depth-pro && pip install -e .
    # Download weights:
    python -c "import depth_pro; depth_pro.create_model_and_transforms()"
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Depth Pro loader
# ─────────────────────────────────────────────────────────────────────────────

def load_depth_pro(device: str = "cuda"):
    """
    Load Depth Pro model and preprocessing transforms.
    Falls back to CPU if CUDA unavailable.
    """
    try:
        import torch
        import depth_pro
    except ImportError:
        print("[ERROR] depth_pro not installed.")
        print("        Install: pip install git+https://github.com/apple/ml-depth-pro.git")
        sys.exit(1)

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — using CPU (will be slow)")
        device = "cpu"

    print(f"[INFO] Loading Depth Pro on {device}...")
    t0 = time.time()
    model, transform = depth_pro.create_model_and_transforms(device=device)
    model.eval()
    print(f"[INFO] Depth Pro loaded in {time.time()-t0:.1f}s")
    return model, transform, device


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_depth(model, transform, image_path: str, device: str) -> tuple:
    """
    Run Depth Pro on a single image.

    Returns
    -------
    depth_m   : np.ndarray (H, W) float32, values in metres
    focal_px  : float — estimated focal length in pixels (Depth Pro output)
    """
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    w_orig, h_orig = image.size

    # Depth Pro expects a specific transform (resize + normalise)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model.infer(image_tensor, f_px=None)  # let model estimate focal

    depth = prediction["depth"].squeeze().cpu().numpy()      # (H', W') float32
    focal_px = float(prediction["focallength_px"].item())

    # Resize depth back to original resolution
    depth = cv2.resize(depth, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    return depth.astype(np.float32), focal_px


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_colormap(depth: np.ndarray,
                       vmin: float = None, vmax: float = None) -> np.ndarray:
    """
    Convert depth array to a colourised BGR image using the 'plasma' colormap.
    Clamps depth to [vmin, vmax] then applies the colormap.
    """
    if vmin is None:
        vmin = np.percentile(depth, 2)
    if vmax is None:
        vmax = np.percentile(depth, 98)

    depth_clipped = np.clip(depth, vmin, vmax)
    depth_norm    = ((depth_clipped - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)

    # Apply plasma colormap (perceptually uniform, good for depth)
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    return colored


def save_depth_viz(rgb_path: str, depth: np.ndarray,
                   focal_px: float, out_path: str) -> None:
    """
    Save a side-by-side RGB | depth colourmap image with a colour bar.
    """
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return

    h, w = rgb.shape[:2]
    depth_color = depth_to_colormap(depth)
    depth_color = cv2.resize(depth_color, (w, h))

    # Add text overlays
    d_med = float(np.median(depth))
    d_max = float(np.percentile(depth, 95))
    cv2.putText(rgb, "RGB", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(depth_color, f"Depth  median={d_med:.2f}m  p95={d_max:.2f}m",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(depth_color, f"focal={focal_px:.0f}px",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)

    side_by_side = np.hstack([rgb, depth_color])
    cv2.imwrite(out_path, side_by_side)


def make_depth_summary_grid(viz_pairs: list, output_path: str,
                             cols: int = 4) -> None:
    """
    Create a grid of (RGB | depth) thumbnail pairs for the README.
    viz_pairs: list of paths to side_by_side images.
    """
    thumbs = []
    thumb_w = 480
    for p in viz_pairs:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / w
        thumbs.append(cv2.resize(img, (thumb_w, int(h * scale))))

    if not thumbs:
        return

    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        ph = max_h - t.shape[0]
        if ph > 0:
            t = np.vstack([t, np.zeros((ph, t.shape[1], 3), dtype=np.uint8)])
        padded.append(t)

    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i:i+cols]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))

    grid = np.vstack(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, grid)
    print(f"[VIZ] Depth summary grid → {output_path}")


def plot_depth_distribution(depth_infos: list, output_path: str) -> None:
    """
    Plot per-frame median depth over time — useful to spot scale drift.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        medians = [d["median_depth_m"] for d in depth_infos]
        focals  = [d["focal_px"]       for d in depth_infos]
        x = list(range(len(medians)))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        ax1.plot(x, medians, color="#1D9E75", linewidth=1.2)
        ax1.fill_between(x, medians, alpha=0.15, color="#1D9E75")
        ax1.set_ylabel("Median depth (m)")
        ax1.set_title("Per-frame depth statistics")
        ax1.grid(True, alpha=0.3)

        ax2.plot(x, focals, color="#BA7517", linewidth=1.2)
        ax2.set_ylabel("Estimated focal length (px)")
        ax2.set_xlabel("Frame index")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"[VIZ] Depth distribution plot → {output_path}")
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def estimate_depth(
    frames_dir: str,
    output_dir: str,
    device: str = "cuda",
    viz: bool = True,
    viz_every: int = 5,     # save side-by-side viz for every N-th frame
    skip_existing: bool = True,
) -> list:
    """
    Run Depth Pro on all keyframes in frames_dir.

    Parameters
    ----------
    frames_dir    : directory of JPEG keyframes
    output_dir    : root output directory
    device        : 'cuda' or 'cpu'
    viz           : produce visualisation outputs
    viz_every     : save RGB|depth comparison for every N-th frame
    skip_existing : skip frames whose .npy already exists (for resuming)

    Returns
    -------
    List of per-frame depth info dicts
    """
    depth_dir   = os.path.join(output_dir, "depth")
    depviz_dir  = os.path.join(output_dir, "depth_viz")
    viz_dir     = os.path.join(output_dir, "viz")
    os.makedirs(depth_dir,  exist_ok=True)
    if viz:
        os.makedirs(depviz_dir, exist_ok=True)
        os.makedirs(viz_dir,    exist_ok=True)

    # Collect frame paths (sorted)
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not frame_files:
        print(f"[ERROR] No images found in {frames_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(frame_files)} frames to process")

    # Load model (once, then loop)
    model, transform, device = load_depth_pro(device)

    depth_infos   = []
    viz_sbs_paths = []  # side-by-side paths for summary grid

    t_start = time.time()
    for i, fname in enumerate(frame_files):
        fpath     = os.path.join(frames_dir, fname)
        stem      = os.path.splitext(fname)[0]
        npy_path  = os.path.join(depth_dir, f"{stem}.npy")
        sbs_path  = os.path.join(depviz_dir, f"{stem}_depth.jpg")

        # Resume support: skip if already done
        if skip_existing and os.path.isfile(npy_path):
            # Load existing to build info dict
            depth = np.load(npy_path)
            info  = {
                "filename":      fname,
                "depth_file":    f"{stem}.npy",
                "min_depth_m":   float(np.min(depth)),
                "max_depth_m":   float(np.max(depth)),
                "median_depth_m":float(np.median(depth)),
                "focal_px":      None,
            }
            depth_infos.append(info)
            if viz and i % viz_every == 0 and os.path.isfile(sbs_path):
                viz_sbs_paths.append(sbs_path)
            continue

        # Infer
        t0 = time.time()
        depth, focal_px = predict_depth(model, transform, fpath, device)
        elapsed = time.time() - t0

        # Save raw depth
        np.save(npy_path, depth)

        info = {
            "filename":       fname,
            "depth_file":     f"{stem}.npy",
            "min_depth_m":    round(float(np.min(depth)),    4),
            "max_depth_m":    round(float(np.max(depth)),    4),
            "median_depth_m": round(float(np.median(depth)), 4),
            "p5_depth_m":     round(float(np.percentile(depth, 5)), 4),
            "p95_depth_m":    round(float(np.percentile(depth, 95)), 4),
            "focal_px":       round(focal_px, 2),
            "inference_time_s": round(elapsed, 2),
        }
        depth_infos.append(info)

        # Per-frame visualisation
        if viz and i % viz_every == 0:
            save_depth_viz(fpath, depth, focal_px, sbs_path)
            viz_sbs_paths.append(sbs_path)

        # Progress
        elapsed_total = time.time() - t_start
        eta = (elapsed_total / (i + 1)) * (len(frame_files) - i - 1)
        print(f"  [{i+1:3d}/{len(frame_files)}]  {fname}  "
              f"median={info['median_depth_m']:.2f}m  "
              f"focal={focal_px:.0f}px  "
              f"({elapsed:.1f}s)  ETA: {eta:.0f}s")

    # Save depth info JSON
    info_path = os.path.join(output_dir, "depth_info.json")
    with open(info_path, "w") as f:
        json.dump({"n_frames": len(depth_infos), "frames": depth_infos}, f, indent=2)
    print(f"\n[OK] Depth maps → {depth_dir}/  ({len(depth_infos)} files)")
    print(f"[OK] Metadata  → {info_path}")

    # Summary visualisations
    if viz:
        if viz_sbs_paths:
            make_depth_summary_grid(
                viz_sbs_paths[:16],  # show up to 16 pairs in the grid
                os.path.join(viz_dir, "depth_overview.png"),
                cols=4
            )
        plot_depth_distribution(
            [d for d in depth_infos if d["focal_px"] is not None],
            os.path.join(viz_dir, "depth_statistics.png")
        )

    # Summary stats
    medians = [d["median_depth_m"] for d in depth_infos if d["median_depth_m"]]
    print(f"\n=== Depth Summary ===")
    print(f"  Frames processed  : {len(depth_infos)}")
    print(f"  Median depth range: {min(medians):.2f}m – {max(medians):.2f}m")
    focals = [d["focal_px"] for d in depth_infos if d.get("focal_px")]
    if focals:
        print(f"  Focal length est. : mean={np.mean(focals):.1f}px  "
              f"std={np.std(focals):.1f}px")

    total_time = time.time() - t_start
    print(f"  Total time        : {total_time:.0f}s  "
          f"({total_time/len(depth_infos):.1f}s/frame)")
    print(f"\nNext step:")
    print(f"  python scripts/fuse_pointcloud.py --output_dir outputs/")

    return depth_infos


def parse_args():
    p = argparse.ArgumentParser(description="Run Depth Pro on all keyframes.")
    p.add_argument("--frames_dir",    required=True,
                   help="Directory of JPEG keyframes")
    p.add_argument("--output_dir",    default="outputs",
                   help="Root output directory (default: outputs/)")
    p.add_argument("--device",        default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--viz",           action="store_true", default=True)
    p.add_argument("--no-viz",        action="store_false", dest="viz")
    p.add_argument("--viz_every",     type=int, default=5,
                   help="Save RGB|depth comparison every N frames (default: 5)")
    p.add_argument("--no_skip",       action="store_false", dest="skip_existing",
                   help="Re-run inference even if .npy files exist")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"[ERROR] Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    estimate_depth(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        device=args.device,
        viz=args.viz,
        viz_every=args.viz_every,
        skip_existing=args.skip_existing,
    )
