"""
Sanity-check a Diffusion-Policy-style .zarr store (the Zarr equivalent of
inspect_dataset.py for LeRobot).

Usage:
    python inspect_zarr.py combined.zarr
    python inspect_zarr.py 2026-07-10-07_07_34/2026-07-10-07_07_34.zarr
"""
import sys
import numpy as np
import zarr

path = sys.argv[1] if len(sys.argv) > 1 else "combined.zarr"
store = zarr.open(path, mode="r")

img = store["data/img"]
state = store["data/state"]
action = store["data/action"]
ends = store["meta/episode_ends"][:]

n = img.shape[0]
print(f"store: {path}")
print(f"total steps : {n}")
print(f"episodes    : {len(ends)}")
print(f"img   shape : {tuple(img.shape)}  {img.dtype}")
print(f"state shape : {tuple(state.shape)}  {state.dtype}")
print(f"action shape: {tuple(action.shape)}  {action.dtype}")

# episode boundaries -> per-episode step counts
starts = np.concatenate([[0], ends[:-1]])
print("\nepisodes (from episode_ends):")
for k, (a, b) in enumerate(zip(starts, ends)):
    print(f"  episode {k}: steps [{a}:{b}]  ({b - a} steps)")

# sanity totals
if ends[-1] != n:
    print(f"\n!! WARNING: last episode_end ({ends[-1]}) != total steps ({n})")
else:
    print(f"\nok: episode_ends[-1] == total steps ({n})")

# frame 0
print("\n--- frame 0 ---")
print("state :", np.asarray(state[0]))
print("action:", np.asarray(action[0]))

# gripper channel (last action dim) sampled across the whole store
grip = np.asarray(action[::25, -1])
print("\ngripper action, every 25th step (should swing between ~0 and ~1):")
print(np.round(grip, 2))

# position-delta magnitudes (first 3 action dims)
step = max(1, n // 400)
dpos = np.linalg.norm(np.asarray(action[::step, :3]), axis=1)
print(f"\nper-frame position-delta magnitude: "
      f"mean={dpos.mean():.4f}  max={dpos.max():.4f}  (expect small, e.g. < ~0.1)")
