"""
Convert AnySense trials into a single Diffusion-Policy / UMI-style .zarr store.

Output layout (the convention Diffusion Policy and UMI expect):
    <out>.zarr/
      data/
        img     (total_steps, H, W, 3)  uint8
        state   (total_steps, 7)        float32   tx,ty,tz, qx,qy,qz,qw
        action  (total_steps, 7)        float32   dx,dy,dz, drx,dry,drz, gripper
      meta/
        episode_ends (num_episodes,)    int64     cumulative end index per episode

All episodes are concatenated; episode_ends marks the boundaries.

Usage:
    python convert_to_zarr.py --root . --out gripper_dataset.zarr
    python convert_to_zarr.py --root . --out gripper_dataset.zarr --img-size 256

Requires: zarr, numcodecs, numpy, opencv-python
    pip install zarr numpy opencv-python
"""
import os
import glob
import argparse
import numpy as np
import cv2
import zarr


def find(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


def is_trial(folder):
    return (find(folder, "AR_Pose_*.txt")
            and (find(folder, "RGB_*.mp4") or find(folder, "RGB_*.mov"))
            and find(folder, "gripper_state.csv"))


def parse_pose(path):
    quats, trans = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            qx, qy, qz, qw, tx, ty, tz = (float(x) for x in p[1:8])
            quats.append([qx, qy, qz, qw])
            trans.append([tx, ty, tz])
    return np.array(quats, np.float32), np.array(trans, np.float32)


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


def _make(group, name, data, chunks):
    """Create an array under `group`, compatible with zarr v2 and v3
    (uses each version's default compression)."""
    if hasattr(group, "create_array"):        # zarr >= 3
        arr = group.create_array(name, shape=data.shape, dtype=data.dtype, chunks=chunks)
        arr[:] = data
        return arr
    return group.create_dataset(name, data=data, chunks=chunks)  # zarr 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder containing trial subfolders")
    ap.add_argument("--out", default="gripper_dataset.zarr")
    ap.add_argument("--img-size", type=int, default=256)
    args = ap.parse_args()

    trials = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                    if os.path.isdir(d) and is_trial(d))
    if not trials:
        raise SystemExit(f"No processed trials found under {args.root!r}.")
    print(f"Found {len(trials)} trial(s).")

    # --- accumulate every step of every episode ---
    all_img, all_state, all_action, episode_ends = [], [], [], []
    running = 0
    for t in trials:
        name = os.path.basename(t)
        quats, trans = parse_pose(find(t, "AR_Pose_*.txt"))
        frames = read_frames(find(t, "RGB_*.mp4") or find(t, "RGB_*.mov"), args.img_size)
        n = min(len(quats), len(frames))
        if n < 2:
            print(f"  SKIP {name}: too few frames")
            continue
        aperture = load_aperture(find(t, "gripper_state.csv"), n)

        for i in range(n - 1):            # last frame has no "next" -> no action
            dpos = (trans[i + 1] - trans[i]).astype(np.float32)
            drot = quat_delta_smallangle(quats[i], quats[i + 1])
            all_img.append(frames[i])
            all_state.append(np.concatenate([trans[i], quats[i]]).astype(np.float32))
            all_action.append(np.concatenate([dpos, drot, [aperture[i + 1]]]).astype(np.float32))

        running += (n - 1)
        episode_ends.append(running)       # cumulative end index
        print(f"  + {name}: {n-1} steps  (ends at {running})")

    img = np.asarray(all_img, dtype=np.uint8)          # (N, H, W, 3)
    state = np.asarray(all_state, dtype=np.float32)     # (N, 7)
    action = np.asarray(all_action, dtype=np.float32)   # (N, 7)
    ends = np.asarray(episode_ends, dtype=np.int64)
    print(f"\nTotal steps: {len(img)}   episodes: {len(ends)}")

    # --- write the zarr store ---
    store = zarr.open(args.out, mode="w")
    data = store.create_group("data")
    meta = store.create_group("meta")

    # chunk images per-frame (chunk size 1 along time) so training can random-access
    _make(data, "img", img, (1, args.img_size, args.img_size, 3))
    _make(data, "state", state, (min(1024, len(state)), 7))
    _make(data, "action", action, (min(1024, len(action)), 7))
    _make(meta, "episode_ends", ends, (len(ends),))

    print(f"\nWrote {args.out}")
    print("Structure:")
    try:
        print(store.tree())
    except Exception:
        print("  data/img   ", img.shape)
        print("  data/state ", state.shape)
        print("  data/action", action.shape)
        print("  meta/episode_ends", ends.tolist())


if __name__ == "__main__":
    main()
