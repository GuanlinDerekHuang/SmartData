"""
Track the target object through the video with SAM2, seeded by the point that
Moondream (or MolmoPoint) predicted on frame 0.

Why this exists: the phone is mounted on the gripper, so the *camera* moves while
the object stays put. A single frame-0 pixel is therefore only correct for frame 0.
Tracking gives the object's pixel position on every frame.

Inputs (in the trial folder):
    RGB_*.mp4            the video
    contact_points.json  from ground_targets*.py (frame-0 point per prompt)

Outputs:
    target_track.csv     frame_idx,label,cx,cy   (blank cx/cy where lost)
    target_preview.mp4   overlay for a quick quality check

Usage:
    python extract_target_track.py <trial_folder>

Env:
    SAM2_CKPT / SAM2_CFG   same as extract_gripper_sam2.py
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
PROC_LONG_SIDE = 512      # must match extract_gripper_sam2.py so coords line up
WRITE_PREVIEW = True


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    return float(xs.mean()), float(ys.mean())


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python extract_target_track.py <trial_folder>")
    folder = sys.argv[1]

    video = find_video(folder)
    if not video:
        raise SystemExit(f"No RGB_* video found in {folder}")
    cp_path = os.path.join(folder, "contact_points.json")
    if not os.path.exists(cp_path):
        raise SystemExit(f"Missing {cp_path} -- run ground_targets_moondream.py first.")

    with open(cp_path) as f:
        cp = json.load(f)
    src_w, src_h = cp["frame_size"]

    # Build the SAM2 prompts: one object id per named target.
    seeds = []   # (obj_id, label, x, y) in ORIGINAL frame coords
    oid = 1
    for label, pts in cp["points"].items():
        for (x, y) in pts:
            seeds.append((oid, label, float(x), float(y)))
            oid += 1
    if not seeds:
        raise SystemExit("contact_points.json contains no points.")
    print(f"Seeding {len(seeds)} target(s) from frame 0:")
    for o, lab, x, y in seeds:
        print(f"  obj {o}: '{lab}' at ({x:.0f}, {y:.0f})")

    frames_dir = tempfile.mkdtemp(prefix="sam2_target_")
    try:
        n, fps, (H, W) = extract_frames(video, frames_dir, PROC_LONG_SIDE)
        sx, sy = W / src_w, H / src_h
        print(f"Extracted {n} frames at {W}x{H}; scaling seed points by "
              f"({sx:.3f}, {sy:.3f})")

        device = pick_device()
        print(f"Device: {device}  (this walks every frame; expect ~30 min on CPU)")
        from sam2.build_sam import build_sam2_video_predictor
        predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CHECKPOINT, device=device)

        cast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device == "cuda" else contextlib.nullcontext())
        tracks = {o: {} for o, _, _, _ in seeds}
        with torch.inference_mode(), cast:
            infer = predictor.init_state(video_path=frames_dir)
            for o, lab, x, y in seeds:
                predictor.add_new_points_or_box(
                    inference_state=infer,
                    frame_idx=0,
                    obj_id=o,
                    points=np.array([[x * sx, y * sy]], dtype=np.float32),
                    labels=np.array([1], dtype=np.int32),   # positive click
                )
            for fidx, obj_ids, mask_logits in predictor.propagate_in_video(infer):
                for k, o in enumerate(obj_ids):
                    m = (mask_logits[k] > 0.0).squeeze().cpu().numpy().astype(bool)
                    c = centroid(m)
                    if c is not None:
                        tracks[int(o)][fidx] = (c[0], c[1], m)

        labels = {o: lab for o, lab, _, _ in seeds}
        csv_path = os.path.join(folder, "target_track.csv")
        found = 0
        with open(csv_path, "w") as f:
            f.write("frame_idx,label,cx,cy\n")
            for fidx in range(n):
                for o in tracks:
                    rec = tracks[o].get(fidx)
                    if rec is None:
                        f.write(f"{fidx},{labels[o]},,\n")
                    else:
                        f.write(f"{fidx},{labels[o]},{rec[0]:.2f},{rec[1]:.2f}\n")
                        found += 1
        print(f"Wrote {csv_path}  ({found} located across {n} frames)")

        if WRITE_PREVIEW:
            cap = cv2.VideoCapture(video)
            writer = None
            palette = [(0, 220, 255), (255, 120, 0), (200, 0, 255)]
            i = 0
            while i < n:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                s = PROC_LONG_SIDE / max(h, w)
                if s < 1:
                    frame = cv2.resize(frame, (int(w * s), int(h * s)))
                for j, o in enumerate(sorted(tracks)):
                    rec = tracks[o].get(i)
                    if rec is None:
                        continue
                    col = palette[j % len(palette)]
                    over = frame.copy()
                    over[rec[2]] = col
                    frame = cv2.addWeighted(frame, 0.7, over, 0.3, 0)
                    cv2.drawMarker(frame, (int(rec[0]), int(rec[1])), col,
                                   cv2.MARKER_CROSS, 22, 2)
                if writer is None:
                    writer = cv2.VideoWriter(
                        os.path.join(folder, "target_preview.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                i += 1
            cap.release()
            if writer:
                writer.release()
            print("Wrote target_preview.mp4  <- check the target stays locked on")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
