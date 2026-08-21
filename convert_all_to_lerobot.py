"""
Batch-convert every trial folder into ONE multi-episode LeRobotDataset.

Each subfolder that contains AR_Pose_*.txt + RGB_*.mp4 + gripper_state.csv
becomes one episode. A per-trial task label can be supplied via a tasks.json
map; otherwise --default-task is used for all.

Usage:
    python convert_all_to_lerobot.py --repo-id GuanlinDerekH/gripper-demos
    python convert_all_to_lerobot.py --repo-id GuanlinDerekH/gripper-demos --overwrite
    # optional per-trial labels: a JSON like
    #   { "2026-07-10-07_07_48": "pick up the object",
    #     "2026-07-10-07_09_01": "pick up the bottle" }
    python convert_all_to_lerobot.py --repo-id ... --tasks-json tasks.json

Run from inside rerun_atomic. Requires: lerobot, numpy, opencv-python.
"""
import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import cv2


def find(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


def is_trial(folder):
    return (find(folder, "AR_Pose_*.txt")
            and (find(folder, "RGB_*.mp4") or find(folder, "RGB_*.mov"))
            and find(folder, "gripper_state.csv"))


def parse_pose(path):
    ts, quats, trans = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            ts.append(int(p[0].strip().strip('"').strip("<>")) / 1000.0)
            qx, qy, qz, qw, tx, ty, tz = (float(x) for x in p[1:8])
            quats.append([qx, qy, qz, qw])
            trans.append([tx, ty, tz])
    return np.array(ts), np.array(quats, np.float32), np.array(trans, np.float32)


def load_aperture(path, n):
    ap = []
    with open(path) as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                ap.append(float(line.split(",")[1]))
    ap = np.array(ap, np.float32)
    if len(ap) < n:
        ap = np.concatenate([ap, np.full(n - len(ap), ap[-1] if len(ap) else 0.0)])
    return ap[:n]


def quat_delta_smallangle(q0, q1):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    cx, cy, cz, cw = -x0, -y0, -z0, w0
    rx = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    ry = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    rz = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    return (2.0 * np.array([rx, ry, rz], np.float32)).astype(np.float32)


def read_frames(video, size):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(fr, (size, size)))
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=".", help="folder containing the trial subfolders")
    ap.add_argument("--default-task", default="manipulate the object")
    ap.add_argument("--tasks-json", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tasks_map = {}
    if args.tasks_json and os.path.exists(args.tasks_json):
        tasks_map = json.load(open(args.tasks_json))

    trials = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                    if os.path.isdir(d) and is_trial(d))
    if not trials:
        raise SystemExit(f"No trial folders found under {args.root!r}.")
    print(f"Found {len(trials)} trial(s):")
    for t in trials:
        print("  ", os.path.basename(t))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.overwrite:
        cache = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
        if os.path.isdir(cache):
            shutil.rmtree(cache)
            print(f"Removed existing dataset at {cache}")

    features = {
        "observation.images.gripper_cam": {
            "dtype": "video", "shape": (args.img_size, args.img_size, 3),
            "names": ["height", "width", "channel"]},
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]},
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, use_videos=True)

    total = 0
    for t in trials:
        name = os.path.basename(t)
        task = tasks_map.get(name, args.default_task)
        pose_txt = find(t, "AR_Pose_*.txt")
        rgb = find(t, "RGB_*.mp4") or find(t, "RGB_*.mov")
        state_csv = find(t, "gripper_state.csv")

        ts, quats, trans = parse_pose(pose_txt)
        frames = read_frames(rgb, args.img_size)
        n = min(len(ts), len(frames))
        if n < 2:
            print(f"  SKIP {name}: too few frames")
            continue
        aperture = load_aperture(state_csv, n)

        for i in range(n - 1):
            dpos = (trans[i + 1] - trans[i]).astype(np.float32)
            drot = quat_delta_smallangle(quats[i], quats[i + 1])
            action = np.concatenate([dpos, drot, [aperture[i + 1]]]).astype(np.float32)
            state = np.concatenate([trans[i], quats[i]]).astype(np.float32)
            dataset.add_frame({
                "observation.images.gripper_cam": frames[i],
                "observation.state": state,
                "action": action,
                "task": task,
            })
        dataset.save_episode()
        total += n - 1
        print(f"  + {name}: {n-1} frames  (task: '{task}')")

    dataset.finalize()
    print(f"\nDone. {len(trials)} episodes, {total} frames total -> '{args.repo_id}'.")


if __name__ == "__main__":
    main()
