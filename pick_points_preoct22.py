"""
Step 1 of gripper segmentation: pick SAM2 prompt points on frame 0.

Usage:
    python pick_points.py <trial_folder>
    e.g.  python pick_points.py 2026-07-10-07_07_48

A window opens showing the first RGB frame. Controls:
    press 1  -> then click the LEFT gripper finger
    press 2  -> then click the RIGHT gripper finger
    left-click  = positive point (this pixel IS the finger)
    right-click = negative point (this pixel is NOT the finger)
    s = save prompts.json and quit
    q = quit without saving

Tip: 1-2 positive clicks per finger plus a couple of negatives on the
background is usually enough. Because your rig is fixed, the same points
often work for every trial.
"""
import os
import sys
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["keymap.save"] = []



PROC_LONG_SIDE = 512  # resize longer side to this (kept consistent in extract step)


def find_video(folder):
    for ext in ("mp4", "MP4", "mov", "MOV"):
        hits = sorted(glob.glob(os.path.join(folder, f"RGB_*.{ext}")))
        if hits:
            return hits[0]
    return None


def load_frame0(video, long_side):
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Couldn't read first frame of {video}")
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pick_points.py <trial_folder>")
    folder = sys.argv[1]
    video = find_video(folder)
    if not video:
        raise SystemExit(f"No RGB_* video found in {folder}")

    img = load_frame0(video, PROC_LONG_SIDE)
    H, W = img.shape[:2]

    state = {"1": {"points": [], "labels": []},
             "2": {"points": [], "labels": []}}
    current = {"obj": "1"}
    colors = {"1": "lime", "2": "cyan"}

    fig, ax = plt.subplots(figsize=(9, 7))

    def draw():
        ax.clear()
        ax.imshow(img)
        for oid, d in state.items():
            for (x, y), lab in zip(d["points"], d["labels"]):
                ax.plot(x, y, marker=("+" if lab == 1 else "x"),
                        color=colors[oid], markersize=13, markeredgewidth=2)
        ax.set_title(f"Object {current['obj']} "
                     f"(press 1=LEFT finger, 2=RIGHT finger)\n"
                     f"left-click=positive  right-click=negative   s=save  q=quit")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        lab = 1 if event.button == 1 else 0
        state[current["obj"]]["points"].append([float(event.xdata), float(event.ydata)])
        state[current["obj"]]["labels"].append(lab)
        draw()

    def on_key(event):
        if event.key in ("1", "2"):
            current["obj"] = event.key
            draw()
        elif event.key == "enter":
            out = {"frame_size": [W, H],
                   "proc_long_side": PROC_LONG_SIDE,
                   "objects": state}
            path = os.path.join(folder, "prompts.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2)
            print("Saved", path)
            plt.close(fig)
        elif event.key == "q":
            print("Quit without saving.")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
