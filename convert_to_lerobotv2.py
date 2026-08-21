"""
Convert an AnySense trial into a LeRobotDataset (v3) episode for behavior cloning.

Observation:  RGB frame + camera/gripper pose (7-dim: tx,ty,tz, qx,qy,qz,qw)
Action:       next-step pose DELTA (6-dim: dx,dy,dz + small-angle rotation)
              + gripper aperture (1-dim)  ->  7-dim action, CAP-style.

This is the plain behavior-cloning version: no 3D contact anchor yet (that needs
metric depth + intrinsics). The schema is a strict subset of the CAP version, so
a contact channel can be added later without rebuilding.

Usage:
    python convert_to_lerobot.py <trial_folder> --repo-id <user>/<name> [--task "pick up the object"]

Requires: lerobot (>=0.4 for v3), numpy, opencv-python
    pip install lerobot
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2


def find(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


def parse_pose(path):
    """Return (timestamps_s, quats Nx4 xyzw, trans Nx3)."""
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
    if len(ap) < n:                       # pad if slightly short
        ap = np.concatenate([ap, np.full(n - len(ap), ap[-1])])
    return ap[:n]


def quat_delta_smallangle(q0, q1):
    """Approximate rotation from q0 to q1 as a 3-vector (axis*angle), xyzw quats."""
    # relative quaternion q_rel = q1 * conj(q0)
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    # conj(q0) = (-x0,-y0,-z0,w0)
    cx, cy, cz, cw = -x0, -y0, -z0, w0
    # Hamilton product q1 * conj(q0)
    rw = w1 * cw - x1 * cx - y1 * cy - z1 * cz
    rx = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    ry = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    rz = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    # small-angle: rotation vector ~ 2 * (rx,ry,rz) for unit quats near identity
    v = np.array([rx, ry, rz], np.float32)
    return (2.0 * v).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial")
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/gripper-pick")
    ap.add_argument("--task", default="manipulate the object")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=256, help="square resize for training")
    args = ap.parse_args()

    trial = args.trial
    pose_txt = find(trial, "AR_Pose_*.txt")
    rgb_mp4 = find(trial, "RGB_*.mp4") or find(trial, "RGB_*.mov")
    state_csv = find(trial, "gripper_state.csv")
    if not (pose_txt and rgb_mp4 and state_csv):
        raise SystemExit("Need AR_Pose_*.txt, RGB_*.mp4, and gripper_state.csv in the trial folder.")

    ts, quats, trans = parse_pose(pose_txt)
    n_pose = len(ts)

    # read RGB frames
    cap = cv2.VideoCapture(rgb_mp4)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (args.img_size, args.img_size))
        frames.append(fr)
    cap.release()

    n = min(n_pose, len(frames))
    print(f"pose rows={n_pose}, frames={len(frames)} -> using {n}")
    frames = frames[:n]
    quats, trans, ts = quats[:n], trans[:n], ts[:n]
    aperture = load_aperture(state_csv, n)

    # build 7-dim state and 7-dim action (action = NEXT-step delta + aperture)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.images.gripper_cam": {
            "dtype": "video", "shape": (args.img_size, args.img_size, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": ["tx", "ty", "tz", "qx", "qy", "qz", "qw"],
        },
        "action": {
            "dtype": "float32", "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        use_videos=True,
    )

    for i in range(n - 1):   # last frame has no "next" -> no action
        dpos = (trans[i + 1] - trans[i]).astype(np.float32)
        drot = quat_delta_smallangle(quats[i], quats[i + 1])
        action = np.concatenate([dpos, drot, [aperture[i + 1]]]).astype(np.float32)
        state = np.concatenate([trans[i], quats[i]]).astype(np.float32)

        dataset.add_frame({
            "observation.images.gripper_cam": frames[i],
            "observation.state": state,
            "action": action,
            "task": args.task,
        })

    dataset.save_episode()
    dataset.finalize()
    print(f"Done. Episode written for repo_id '{args.repo_id}'.")
    print("Visualize/train locally without pushing; push_to_hub() is optional.")


if __name__ == "__main__":
    main()
