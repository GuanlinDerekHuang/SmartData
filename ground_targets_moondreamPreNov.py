"""
Visual grounding with Moondream 2 (lightweight alternative to MolmoPoint): find the pixel location of named target
objects in a trial's RGB video, for use as CAP-style contact prompts.

This is the "contact prompting" step from Contact-Anchored Policies: instead of
clicking the object manually, we ask a VLM to point at it from a text prompt.

Usage:
    python ground_targets.py <trial_folder> "small bouncing ball" "pill bottle"
    python ground_targets.py 2026-07-10-07_07_48 "water bottle" "vitamin bottle"

Options via env vars:
    MD_MODEL      default vikhyatk/moondream2 (~2B, runs fine on CPU)
    FRAME_IDX     which video frame to ground on (default 0)

Outputs into the trial folder:
    contact_points.json     {"<prompt>": [[x, y], ...], ...}  pixel coords in ORIGINAL frame
    contact_points.png      the frame with the points drawn, for a quick visual check

Environment (per Ai2's model card):
    conda create --name molmo python=3.11 && conda activate molmo
    pip install transformers==4.57.1 torch pillow einops torchvision accelerate decord2
"""
import os
import sys
import glob
import json
import contextlib
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_ID = os.environ.get("MD_MODEL", "vikhyatk/moondream2")
# Moondream updates often; pin a revision for reproducibility if you care.
REVISION = os.environ.get("MD_REVISION")
FRAME_IDX = int(os.environ.get("FRAME_IDX", "0"))
# The checkpoint ships as F32 (~36GB for 8B). Load in half precision instead.
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
         "fp32": torch.float32}[os.environ.get("MD_DTYPE", "fp16")]


def find_video(folder):
    for ext in ("mp4", "MP4", "mov", "MOV"):
        hits = sorted(glob.glob(os.path.join(folder, f"RGB_*.{ext}")))
        if hits:
            return hits[0]
    return None


def grab_frame(video, idx):
    cap = cv2.VideoCapture(video)
    if idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Couldn't read frame {idx} of {video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def pick_device():
    forced = os.environ.get("MD_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def point_at(model, pil_img, prompt):
    """Ask Moondream to point at `prompt`; return list of (x, y) in pixels."""
    W, H = pil_img.size
    result = model.point(pil_img, prompt)
    pts = []
    # Moondream returns normalized coords in 0..1 (dict or list form).
    raw = result["points"] if isinstance(result, dict) and "points" in result else result
    for p in raw:
        if isinstance(p, dict):
            x, y = float(p["x"]), float(p["y"])
        else:
            x, y = float(p[0]), float(p[1])
        pts.append((x * W, y * H))   # scale to pixels
    return pts


def main():
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python ground_targets.py <trial_folder> "object one" ["object two" ...]')
    folder = sys.argv[1]
    prompts = sys.argv[2:]

    video = find_video(folder)
    if not video:
        raise SystemExit(f"No RGB_* video found in {folder}")

    frame = grab_frame(video, FRAME_IDX)
    H, W = frame.shape[:2]
    pil = Image.fromarray(frame)
    print(f"Grounding on frame {FRAME_IDX} of {os.path.basename(video)} ({W}x{H})")

    device = pick_device()
    if device == "cpu":
        # Moondream auto-detects MPS internally and moves tensors there, which
        # clashes with CPU-loaded weights ("Passed CPU tensor to MPS op").
        # Hide MPS so its detection returns False and everything stays on CPU.
        torch.backends.mps.is_available = lambda: False
        torch.backends.mps.is_built = lambda: False
    print(f"Loading {MODEL_ID} on {device} as {DTYPE} "
          f"(first run downloads the weights; this is large)")
    # NOTE: no device_map="auto". That lets accelerate offload layers to disk,
    # which produces meta tensors that MPS cannot materialize.
    kw = dict(trust_remote_code=True, dtype=DTYPE)
    if REVISION:
        kw["revision"] = REVISION
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    model = model.to(device).eval()

    results = {}
    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    palette = [(0, 220, 120), (0, 150, 255), (255, 80, 200), (60, 220, 255)]

    for i, prompt in enumerate(prompts):
        pts = point_at(model, pil, prompt)
        results[prompt] = pts
        print(f"  '{prompt}' -> {pts if pts else 'NO POINT FOUND'}")
        color = palette[i % len(palette)]
        for (x, y) in pts:
            cv2.drawMarker(vis, (int(round(x)), int(round(y))), color,
                           cv2.MARKER_CROSS, 26, 3)
            cv2.putText(vis, prompt, (int(x) + 12, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    json_path = os.path.join(folder, "contact_points.json")
    with open(json_path, "w") as f:
        json.dump({"frame_idx": FRAME_IDX, "frame_size": [W, H],
                   "points": results}, f, indent=2)
    png_path = os.path.join(folder, "contact_points.png")
    cv2.imwrite(png_path, vis)
    print(f"Wrote {json_path}\nWrote {png_path}  <- open this to check the points landed correctly")


if __name__ == "__main__":
    main()
