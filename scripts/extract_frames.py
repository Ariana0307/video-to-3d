"""
extract_frames.py
-----------------
Stage 1 of the video-to-3D pipeline.

Takes a raw video file and extracts a set of sharp, well-spaced keyframes
using optical flow magnitude as a motion signal — rather than naive uniform
sampling. Only keeps a frame when the scene has moved "enough" AND the frame
is sharp (Laplacian variance > threshold).

Outputs
-------
outputs/frames/          : JPEG keyframes (frame_XXXXX.jpg)
outputs/frame_info.json  : {filename, original_index, timestamp_s, flow_score}
outputs/viz/             : contact sheet PNG for quick visual inspection

Usage
-----
    python scripts/extract_frames.py \
        --video "path/to/video.mp4" \
        --output_dir outputs/ \
        --flow_threshold 3.0 \
        --blur_threshold 80.0 \
        --max_frames 150 \
        --viz
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute mean optical flow magnitude between two grayscale frames
    using Farneback dense optical flow.
    Returns a single float — higher = more motion.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def compute_sharpness(gray: np.ndarray) -> float:
    """
    Laplacian variance as a proxy for image sharpness.
    A blurry frame has low variance in its Laplacian.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_contact_sheet(image_paths: list, output_path: str,
                        thumb_w: int = 240, cols: int = 6) -> None:
    """
    Assemble a grid of thumbnails for quick visual inspection.
    Saves to output_path as a PNG.
    """
    thumbs = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_w / w
        thumb = cv2.resize(img, (thumb_w, int(h * scale)))
        # Pad to uniform height
        thumbs.append(thumb)

    if not thumbs:
        print("[WARN] No images to make contact sheet from.")
        return

    # Uniform height = max thumb height
    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        ph = max_h - t.shape[0]
        if ph > 0:
            t = np.vstack([t, np.zeros((ph, t.shape[1], 3), dtype=np.uint8)])
        padded.append(t)

    # Build rows
    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i+cols]
        # Pad last row if needed
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))
    sheet = np.vstack(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, sheet)
    print(f"[VIZ] Contact sheet saved → {output_path}")


def make_flow_plot(frame_indices: list, flow_scores: list,
                   kept_mask: list, output_path: str) -> None:
    """
    Save a simple matplotlib plot of optical flow over time,
    highlighting which frames were kept vs discarded.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(frame_indices, flow_scores, color="#888", linewidth=0.8,
                label="flow magnitude")

        kept_idx = [frame_indices[i] for i in range(len(kept_mask)) if kept_mask[i]]
        kept_flow = [flow_scores[i] for i in range(len(kept_mask)) if kept_mask[i]]
        ax.scatter(kept_idx, kept_flow, color="#378ADD", s=18, zorder=5,
                   label=f"kept frames ({sum(kept_mask)})")

        ax.set_xlabel("Video frame index")
        ax.set_ylabel("Mean optical flow (px)")
        ax.set_title("Optical flow — adaptive frame selection")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"[VIZ] Flow plot saved → {output_path}")
    except ImportError:
        print("[WARN] matplotlib not found — skipping flow plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction logic
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    output_dir: str,
    flow_threshold: float = 3.0,
    blur_threshold: float = 80.0,
    max_frames: int = 150,
    viz: bool = True,
) -> list:
    """
    Extract keyframes from a video using adaptive optical-flow sampling.

    Parameters
    ----------
    video_path      : path to input .mp4 / .mov
    output_dir      : root output directory
    flow_threshold  : minimum mean optical flow (pixels) to keep a frame
    blur_threshold  : minimum Laplacian variance; blurry frames are dropped
    max_frames      : hard cap on number of output frames
    viz             : whether to produce visualisation outputs

    Returns
    -------
    List of dicts with frame metadata (filename, index, timestamp, flow_score)
    """

    frames_dir = os.path.join(output_dir, "frames")
    viz_dir    = os.path.join(output_dir, "viz")
    os.makedirs(frames_dir, exist_ok=True)
    if viz:
        os.makedirs(viz_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps if fps > 0 else 0

    print(f"[INFO] Video: {os.path.basename(video_path)}")
    print(f"       {total_frames} frames @ {fps:.1f} fps  ({duration_s:.1f}s)")
    print(f"[INFO] Flow threshold={flow_threshold}  Blur threshold={blur_threshold}")
    print(f"[INFO] Max output frames={max_frames}")
    print()

    # ── First pass: read every frame, compute flow ──────────────────────────
    all_frame_indices = []
    all_flow_scores   = []
    all_sharpness     = []

    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = compute_sharpness(gray)

        if prev_gray is not None:
            flow_score = compute_flow_magnitude(prev_gray, gray)
        else:
            flow_score = 0.0

        all_frame_indices.append(frame_idx)
        all_flow_scores.append(flow_score)
        all_sharpness.append(sharpness)

        prev_gray = gray
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  [scan] {frame_idx}/{total_frames} frames analysed...", end="\r")

    cap.release()
    print(f"  [scan] {frame_idx}/{total_frames} frames analysed.    ")

    # ── Second pass: select keyframes ───────────────────────────────────────
    # Adaptive: keep a frame if (flow >= threshold) AND (sharp enough)
    # Then apply max_frames cap by keeping highest-flow frames if over budget.

    kept_mask = [False] * len(all_frame_indices)

    # Always keep the very first and last frame
    kept_mask[0] = True
    kept_mask[-1] = True

    for i in range(1, len(all_frame_indices) - 1):
        if (all_flow_scores[i] >= flow_threshold and
                all_sharpness[i] >= blur_threshold):
            kept_mask[i] = True

    kept_indices = [i for i, k in enumerate(kept_mask) if k]
    print(f"[INFO] {len(kept_indices)} frames pass flow+blur filter")

    # If over budget, thin by keeping highest-flow frames
    if len(kept_indices) > max_frames:
        # Always keep first and last, thin the rest by flow score
        interior = kept_indices[1:-1]
        interior_scores = [all_flow_scores[i] for i in interior]
        sorted_interior = sorted(interior, key=lambda i: all_flow_scores[i],
                                  reverse=True)
        keep_interior = set(sorted_interior[:max_frames - 2])
        kept_indices = (
            [kept_indices[0]] +
            sorted([i for i in interior if i in keep_interior]) +
            [kept_indices[-1]]
        )
        print(f"[INFO] Thinned to {len(kept_indices)} frames (budget cap)")

    # ── Third pass: extract and save selected frames ─────────────────────────
    cap = cv2.VideoCapture(video_path)
    frame_metadata = []
    saved_paths    = []
    prev_idx       = -1

    for seq_num, frame_idx in enumerate(kept_indices):
        # Seek: if not sequential, use grab() loop for efficiency
        if frame_idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Could not read frame {frame_idx}, skipping.")
            prev_idx = frame_idx
            continue

        filename = f"frame_{frame_idx:06d}.jpg"
        out_path = os.path.join(frames_dir, filename)
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        meta = {
            "filename":       filename,
            "frame_index":    frame_idx,
            "timestamp_s":    round(frame_idx / fps, 4) if fps > 0 else 0,
            "flow_score":     round(all_flow_scores[frame_idx], 4),
            "sharpness":      round(all_sharpness[frame_idx], 2),
        }
        frame_metadata.append(meta)
        saved_paths.append(out_path)
        prev_idx = frame_idx

    cap.release()

    # Save metadata JSON
    info_path = os.path.join(output_dir, "frame_info.json")
    with open(info_path, "w") as f:
        json.dump({
            "video":          os.path.basename(video_path),
            "fps":            fps,
            "total_frames":   total_frames,
            "duration_s":     round(duration_s, 2),
            "flow_threshold": flow_threshold,
            "blur_threshold": blur_threshold,
            "n_keyframes":    len(frame_metadata),
            "frames":         frame_metadata,
        }, f, indent=2)

    print(f"\n[OK] Saved {len(frame_metadata)} keyframes → {frames_dir}/")
    print(f"[OK] Metadata → {info_path}")

    # ── Visualisations ───────────────────────────────────────────────────────
    if viz:
        # Contact sheet (subsample if many frames)
        viz_paths = saved_paths
        if len(viz_paths) > 60:
            step = len(viz_paths) // 60
            viz_paths = viz_paths[::step]
        make_contact_sheet(
            viz_paths,
            os.path.join(viz_dir, "keyframes_contact_sheet.png")
        )

        # Optical flow plot
        make_flow_plot(
            all_frame_indices, all_flow_scores, kept_mask,
            os.path.join(viz_dir, "flow_scores.png")
        )

    return frame_metadata


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Extract keyframes from video.")
    p.add_argument("--video",            required=True,
                   help="Path to input video file")
    p.add_argument("--output_dir",       default="outputs",
                   help="Root output directory (default: outputs/)")
    p.add_argument("--flow_threshold",   type=float, default=3.0,
                   help="Min mean optical flow magnitude to keep a frame (default: 3.0)")
    p.add_argument("--blur_threshold",   type=float, default=80.0,
                   help="Min Laplacian variance; drop blurry frames (default: 80.0)")
    p.add_argument("--max_frames",       type=int,   default=150,
                   help="Hard cap on output frame count (default: 150)")
    p.add_argument("--viz",              action="store_true", default=True,
                   help="Produce visualisation outputs (default: True)")
    p.add_argument("--no-viz",           action="store_false", dest="viz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    metadata = extract_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        flow_threshold=args.flow_threshold,
        blur_threshold=args.blur_threshold,
        max_frames=args.max_frames,
        viz=args.viz,
    )

    print(f"\n=== Summary ===")
    print(f"  Keyframes extracted : {len(metadata)}")
    if metadata:
        ts = [m['timestamp_s'] for m in metadata]
        print(f"  Time span           : {ts[0]:.2f}s → {ts[-1]:.2f}s")
        flow_vals = [m['flow_score'] for m in metadata]
        print(f"  Flow scores         : min={min(flow_vals):.2f}  "
              f"max={max(flow_vals):.2f}  "
              f"mean={np.mean(flow_vals):.2f}")
    print(f"\nNext step:")
    print(f"  python scripts/estimate_poses.py --frames_dir outputs/frames/ --output_dir outputs/")
