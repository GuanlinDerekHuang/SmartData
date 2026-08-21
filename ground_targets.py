"""
Visual grounding with MolmoPoint: find the pixel location of named target
objects in a trial's RGB video, for use as CAP-style contact prompts.

This is the "contact prompting" step from Contact-Anchored Policies: instead of
clicking the object manually, we ask a VLM to point at it from a text prompt.

Usage:
    python ground_targets.py <trial_folder> "small bouncing ball" "pill bottle"
    python ground_targets.py 2026-07-10-07_07_48 "water bottle" "vitamin bottle"

Options via env vars:
    MOLMO_MODEL   default allenai/MolmoPoint-8B  (try MolmoPoint-Vid-4B if RAM-limited)
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
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = os.environ.get("MOLMO_MODEL", "allenai/MolmoPoint-8B")
FRAME_IDX = int(os.environ.get("FRAME_IDX", "0"))
# The checkpoint ships as F32 (~36GB for 8B). Load in half precision instead.
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16,
         "fp32": torch.float32}[os.environ.get("MOLMO_DTYPE", "fp16")]


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
    forced = os.environ.get("MOLMO_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def point_at(model, processor, pil_img, prompt, device):
    """Ask the model to point at `prompt`; return list of (x, y) in pixels."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Point to the {prompt}"},
            {"type": "image", "image": pil_img},
        ],
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
        return_pointing_metadata=True,   # needed to decode point tokens
    )
    metadata = inputs.pop("metadata")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    cast = (torch.autocast(device, dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext())
    with torch.inference_mode(), cast:
        output = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=200,
        )

    gen = output[:, inputs["input_ids"].size(1):]
    text = processor.post_process_image_text_to_text(
        gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    pts = model.extract_image_points(
        text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["image_sizes"],
    )
    # points come back as (object_id, image_num, x, y)
    out = []
    for p in np.array(pts).reshape(-1, 4):
        out.append((float(p[2]), float(p[3])))
    return out


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
    print(f"Loading {MODEL_ID} on {device} as {DTYPE} "
          f"(first run downloads the weights; this is large)")
    # NOTE: no device_map="auto". That lets accelerate offload layers to disk,
    # which produces meta tensors that MPS cannot materialize.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=DTYPE, low_cpu_mem_usage=True)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="left")

    results = {}
    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    palette = [(0, 220, 120), (0, 150, 255), (255, 80, 200), (60, 220, 255)]

    for i, prompt in enumerate(prompts):
        pts = point_at(model, processor, pil, prompt, device)
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
