"""
Sanity-check a converted LeRobotDataset before training.

Usage:
    python inspect_dataset.py GuanlinDerekH/gripper-pick
"""
import sys
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = sys.argv[1] if len(sys.argv) > 1 else "GuanlinDerekH/gripper-pick"
ds = LeRobotDataset(repo_id)

print("frames:", ds.num_frames, " episodes:", ds.num_episodes)

s = ds[0]
print("\n--- frame 0 ---")
print("state :", np.asarray(s["observation.state"]))
print("action:", np.asarray(s["action"]))
img = s["observation.images.gripper_cam"]
print("image :", tuple(img.shape), img.dtype)

# Look at the gripper channel (last action value) across the episode to confirm
# it opens/closes like your aperture signal (near 1 open, near 0 at grasps).
grip = np.array([np.asarray(ds[i]["action"])[-1] for i in range(0, ds.num_frames, 25)])
print("\ngripper action, every 25th frame (should swing between ~0 and ~1):")
print(np.round(grip, 2))

# Position deltas should be small per frame (cm-scale at 30 Hz), not huge jumps.
dpos = np.array([np.linalg.norm(np.asarray(ds[i]["action"])[:3])
                 for i in range(min(ds.num_frames, 200))])
print(f"\nper-frame position-delta magnitude: "
      f"mean={dpos.mean():.4f}  max={dpos.max():.4f}  (expect small, e.g. < ~0.1)")
