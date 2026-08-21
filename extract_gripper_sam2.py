"""
Step 2 of gripper segmentation: run SAM2 on the trial's RGB video.

Reproduces CAP's method (Cui et al., App. A.1.1): segment the left and right
fingers, take each mask's centroid, and map the centroid distance to a scalar
aperture in [0, 1] (0 = closed, 1 = open).

Outputs into the trial folder:
    gripper_state.csv    frame_idx, aperture, state(OPEN/CLOSED)
    gripper_masks.npz    masks: (N, H, W) uint8  (0 bg, 1 left finger, 2 right finger)
    gripper_preview.mp4  overlay for a quick quality check

Usage:
    python extract_gripper_sam2.py <trial_folder>

Requires: sam2, torch, opencv-python, numpy, and prompts.json from pick_points.py
Set the checkpoint / config via env vars if they aren't at the defaults:
    export SAM2_CKPT=/path/to/sam2.1_hiera_tiny.pt
    export SAM2_CFG=configs/sam2.1/sam2.1_hiera_t.yaml
"""
import os
import sys
import glob
import json
import shutil
import tempfile
import contextlib
import numpy as np
import cv2
import torch

SAM2_CHECKPOINT = os.environ.get("SAM2_CKPT", "sam2/checkpoints/sam2.1_hiera_tiny.pt")
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_t.yaml")
OPEN_THRESHOLD = 0.5     # aperture above this => "OPEN"
WRITE_PREVIEW = True


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    # SAM2 on Apple MPS can produce degraded masks; default to CPU for correctness.
    return "cpu"


def find_video(folder):
    for ext in ("mp4", "MP4", "mov", "MOV"):
        hits = sorted(glob.glob(os.path.join(folder, f"RGB_*.{ext}")))
        if hits:
            return hits[0]
    return None


def extract_frames(video, out_dir, long_side):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, size = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = long_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), frame)
        size = frame.shape[:2]
        i += 1
    cap.release()
    return i, fps, size


def centroid(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()])


def fill_nans(a):
    a = a.copy()
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() == 0:
        return np.ones_like(a)
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def write_preview(video, masks, aperture, long_side, out_path, fps):
    cap = cv2.VideoCapture(video)
    colors = {1: (0, 255, 0), 2: (0, 180, 255)}
    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(masks):
            break
        h, w = frame.shape[:2]
        scale = long_side / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        over = frame.copy()
        for cls, color in colors.items():
            over[masks[i] == cls] = color
        blend = cv2.addWeighted(frame, 0.6, over, 0.4, 0)
        cv2.putText(blend, f"aperture={aperture[i]:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (blend.shape[1], blend.shape[0]))
        writer.write(blend)
        i += 1
    cap.release()
    if writer:
        writer.release()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python extract_gripper_sam2.py <trial_folder>")
    folder = sys.argv[1]
    video = find_video(folder)
    if not video:
        raise SystemExit(f"No RGB_* video found in {folder}")
    prompts_path = os.path.join(folder, "prompts.json")
    if not os.path.exists(prompts_path):
        raise SystemExit(f"Missing {prompts_path} — run pick_points.py first.")
    with open(prompts_path) as f:
        prompts = json.load(f)
    long_side = prompts.get("proc_long_side", 512)

    frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        n, fps, (H, W) = extract_frames(video, frames_dir, long_side)
        print(f"Extracted {n} frames at {W}x{H}, fps~{fps:.1f}")

        device = pick_device()
        print("Device:", device, "(set CUDA machine for speed; CPU works but is slow)")
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CHECKPOINT, device=device)

        cast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())
        masks_by_frame = {}
        with torch.inference_mode(), cast:
            infer = predictor.init_state(video_path=frames_dir)
            for oid_str, d in prompts["objects"].items():
                if not d["points"]:
                    print(f"WARNING: object {oid_str} has no points; skipping.")
                    continue
                predictor.add_new_points_or_box(
                    inference_state=infer,
                    frame_idx=0,
                    obj_id=int(oid_str),
                    points=np.array(d["points"], dtype=np.float32),
                    labels=np.array(d["labels"], dtype=np.int32),
                )
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(infer):
                masks_by_frame[fidx] = {}
                for k, oid in enumerate(obj_ids):
                    m = (mask_logits[k] > 0.0).squeeze().cpu().numpy().astype(bool)
                    masks_by_frame[fidx][int(oid)] = m

        # Reduce to an integer mask stack + centroid distance per frame.
        masks = np.zeros((n, H, W), dtype=np.uint8)
        dist = np.full(n, np.nan, dtype=np.float32)
        for fidx in range(n):
            fr = masks_by_frame.get(fidx, {})
            mL, mR = fr.get(1), fr.get(2)
            if mL is not None:
                masks[fidx][mL] = 1
            if mR is not None:
                masks[fidx][mR] = 2
            cL = centroid(mL) if mL is not None else None
            cR = centroid(mR) if mR is not None else None
            if cL is not None and cR is not None:
                dist[fidx] = float(np.linalg.norm(cL - cR))

        dist = fill_nans(dist)
        lo, hi = np.nanpercentile(dist, 1), np.nanpercentile(dist, 99)
        aperture = np.clip((dist - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        csv_path = os.path.join(folder, "gripper_state.csv")
        with open(csv_path, "w") as f:
            f.write("frame_idx,aperture,state\n")
            for i in range(n):
                st = "OPEN" if aperture[i] > OPEN_THRESHOLD else "CLOSED"
                f.write(f"{i},{aperture[i]:.5f},{st}\n")
        print("Wrote", csv_path)

        npz_path = os.path.join(folder, "gripper_masks.npz")
        np.savez_compressed(npz_path, masks=masks, fps=np.float32(fps))
        print("Wrote", npz_path)

        if WRITE_PREVIEW:
            write_preview(video, masks, aperture, long_side,
                          os.path.join(folder, "gripper_preview.mp4"), fps)
            print("Wrote gripper_preview.mp4 (open it to check the masks)")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
