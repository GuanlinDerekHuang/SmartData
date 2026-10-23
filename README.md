# SmartData

**A pipeline that turns iPhone recordings from a handheld gripper into labeled, training-ready robot-manipulation data.**

SmartData takes raw recordings from an iPhone mounted on a handheld two-finger gripper (RGB video, depth, and 6-DoF camera pose) and processes them into a synchronized, labeled dataset for training imitation-learning policies. One command (`main.py`) runs the whole chain:

- gripper segmentation and an open/closed signal
- contact (grasp) labeling
- optional target-object grounding and tracking
- a synchronized Rerun visualization
- export to **both** LeRobot and Zarr dataset formats

The project reproduces the data-collection and label-extraction half of the **Contact-Anchored Policies (CAP)** pipeline: AnySense capture, SAM 2 gripper-aperture extraction, and hindsight contact labeling. It adds an interactive [Rerun](https://rerun.io) visualization layer, a one-command pipeline controller, and dual-format dataset export.

---

## Quick start: analyzing a brand-new trial

Each step says which machine it runs on.

```bash
# 0. (One time per terminal) point SAM 2 at its checkpoint
export SAM2_CKPT=/path/to/sam2/checkpoints/sam2.1_hiera_tiny.pt
export SAM2_CFG=configs/sam2.1/sam2.1_hiera_t.yaml

# 1. Put the new trial folder inside the repo
#    (the folder AnySense exported, with RGB_*.mp4, Depth_*.mp4, AR_Pose_*.txt)
cd /path/to/SmartData
ls <TRIAL>/

# 2. Mark the two gripper fingers on frame 0 (manual: opens a window)
conda run -n base python pick_points.py <TRIAL>

# 3. Preview what the pipeline will do
python main.py <TRIAL> --to rrd --dry-run

# 4. Label and visualize the trial (SAM 2 takes ~10–30 min on CPU, a few min on a CUDA GPU)
python main.py <TRIAL> --to rrd
#    optional: also find and track target objects in the scene
#    python main.py <TRIAL> --to rrd --ground "football" "water bottle"

# 5. Check the labels
open <TRIAL>/gripper_preview.mp4              # finger masks stay separate; aperture dips at grasps
rerun <TRIAL>/phone_gripper_capture.rrd       # everything on one timeline

# 6. Rebuild the combined training datasets with the new trial included
python main.py --all --dry-run
python main.py --all

# 7. Validate both dataset formats
conda run -n lerobot python inspect_dataset.py SmartData/combined
conda run -n lerobot python inspect_zarr.py combined.zarr

# 8. Commit the new trial's raw files and labels
#    (.gitignore keeps large generated files out)
git add .
git commit -m "Add trial <TRIAL>"
git push
```

`<TRIAL>` is the trial folder name, for example `2026-07-10-07_07_34`. Details for each step follow.

### Step 2: clicking the fingers

A window opens on the trial's first frame. The phone moves between trials, so this step can't be automated.

| Key / click | Action |
|---|---|
| `1` | Start marking the **left** finger |
| `2` | Start marking the **right** finger |
| Left-click | Positive point: this pixel **is** the current finger |
| Right-click | Negative point: this pixel is **not** the current finger |
| `Enter` | Save `prompts.json` and close |
| `q` | Quit **without** saving |

Tips:
- Put 1–2 positive clicks in the middle of each finger pad.
- While marking one finger, put a negative click on the **other** finger. The two fingers look alike, so this keeps their masks from merging.
- Add a negative click or two on the background between the fingers.
- A successful save prints `Saved <TRIAL>/prompts.json` in the terminal. No dialog appears.

### Step 4: why `--to rrd`

`--to rrd` runs only the labeling stages for this trial. Step 6 then packages **all** labeled trials, including this one, into the combined datasets. That avoids building a single-trial dataset you won't train on. If you do want a standalone dataset for just this trial, run `python main.py <TRIAL>` without `--to`.

### Step 5: what to look for

- **`gripper_preview.mp4`:** the green and blue finger masks should stay on their own fingers and never merge. The printed aperture should drop toward 0 when the gripper closes.
- **Rerun:** dragging the timeline should move the video, depth, 3D trajectory, masks, and gripper plots together.
- If the masks are wrong, re-run `pick_points.py`, then redo only SAM 2 with `python main.py <TRIAL> --to rrd --force sam2`.

---

## Overview

Each trial is a folder recorded by the [AnySense](https://github.com/NYU-robot-learning/AnySense) iOS app on an iPhone rigidly mounted to a handheld two-finger gripper:

- `RGB_*.mp4`: the camera view
- `Depth_*.mp4`: a depth visualization video
- `AR_Pose_*.txt`: timestamped 6-DoF camera/gripper pose (quaternion + translation) from ARKit visual-inertial odometry at about 30 Hz

### Pipeline

```
AnySense trial (RGB + Depth + Pose)
        │
        ├─ pick_points ───────► prompts.json                (manual finger clicks)
        ├─ SAM 2 ─────────────► finger masks + aperture     (gripper open/closed signal)
        ├─ contact labeling ──► grasp moments + contact points
        ├─ Moondream 2 ───────► target object point         (optional, --ground)
        │     └─ SAM 2 ───────► target tracked over frames
        ├─ Rerun ─────────────► synchronized .rrd visualization
        └─ packaging ─────────► LeRobot dataset + Zarr store (combined across trials)
```

### Pipeline order and scripts

| Order | Script | Env | What it does | Output (in trial folder unless noted) |
|---|---|---|---|---|
| — | `main.py` | any | Controller: runs the stages below in their correct conda envs, skips finished work, single-trial or `--all` | — |
| 1 | `pick_points.py` | base | **Manual:** click the two gripper fingers on frame 0 | `prompts.json` |
| 2 | `extract_gripper_sam2.py` | base | SAM 2 tracks both fingers; the distance between their centers becomes a 0–1 gripper aperture | `gripper_state.csv`, `gripper_masks.npz`, `gripper_preview.mp4` |
| 3 | `extract_contact_anchor.py` | base | CAP hindsight labeling: contact = the frame where the aperture stops decreasing; anchor = point between the fingers | `contact_anchors.json`, `grasp_center.csv`, `contact_preview.png` |
| 4 | `ground_targets_moondream.py` | molmo | *Optional:* Moondream 2 points to named objects on frame 0 | `contact_points.json`, `contact_points.png` |
| 5 | `extract_target_track.py` | base | *Optional:* SAM 2 tracks each grounded object across all frames | `target_track.csv`, `target_preview.mp4` |
| 6 | `build_gripper_rrd.py` | base | Builds the Rerun recording: video, depth, 3D pose, masks, gripper state, contacts, targets | `phone_gripper_capture.rrd` |
| 7 | `convert_to_lerobot.py` / `convert_all_to_lerobot.py` | lerobot | One trial / all labeled trials → LeRobotDataset v3 | `~/.cache/huggingface/lerobot/SmartData/...` |
| 8 | `convert_to_zarr.py` | lerobot | One trial / all labeled trials → Diffusion Policy-style Zarr | `<TRIAL>/<TRIAL>.zarr` or `combined.zarr` |
| — | `inspect_dataset.py` / `inspect_zarr.py` | lerobot | Validate each dataset format | printed report |
| — | `ground_targets.py` | molmo | MolmoPoint variant of step 4 (needs a CUDA GPU with ~18 GB; see notes) | `contact_points.json` |

---

## Installation

The pipeline uses three conda environments, because the tools have conflicting dependencies. `main.py` switches between them automatically with `conda run`, so you never activate them by hand. The names below must match the `ENVS` dictionary at the top of `main.py`.

**`base`** (SAM 2, Rerun, contact labeling):
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd checkpoints && ./download_ckpts.sh && cd ../..
pip install rerun-sdk numpy opencv-python matplotlib
```

**`molmo`** (Moondream 2 grounding, optional):
```bash
conda create -n molmo python=3.11 -y
conda run -n molmo pip install transformers torch pillow einops torchvision accelerate opencv-python numpy
```

**`lerobot`** (dataset export and validation):
```bash
conda create -n lerobot python=3.11 -y
conda run -n lerobot pip install lerobot zarr opencv-python numpy
```

**FFmpeg**, so Rerun can decode the H.264/HEVC videos:
```bash
brew install ffmpeg   # macOS
```

**SAM 2 checkpoint path.** Export it in every new terminal, or add both lines to `~/.zshrc` so you only do it once:
```bash
export SAM2_CKPT=/path/to/sam2/checkpoints/sam2.1_hiera_tiny.pt
export SAM2_CFG=configs/sam2.1/sam2.1_hiera_t.yaml
```

---

## `main.py` reference

```bash
python main.py <TRIAL>                     # label + visualize + package one trial
python main.py --all                       # label every ready trial, then package ALL labeled trials combined
python main.py <TRIAL> --dry-run           # print the plan and commands, run nothing
python main.py <TRIAL> --to rrd            # labeling + visualization only, no datasets
python main.py <TRIAL> --from contact      # start partway (earlier outputs must exist)
python main.py <TRIAL> --force sam2        # redo one stage even though its output exists
python main.py <TRIAL> --force             # redo every stage
python main.py <TRIAL> --ground "football" "water bottle"   # add target grounding + tracking
python main.py <TRIAL> --no-rrd --no-lerobot --no-zarr      # skip specific outputs
python main.py --all --repo-id me/name --zarr out.zarr      # custom output names
```

**Stages, in order:** `sam2 → contact → ground → track → rrd → lerobot → zarr`. `ground` and `track` run only with `--ground`.

**Skip-if-exists:** each labeling stage is skipped when its output file already exists, so re-runs are cheap and the slow SAM 2 pass never repeats by accident. Use `--force` to override.

**`--all` status table:** marks each trial as `labeled` (SAM 2 already done), `ready` (has `prompts.json`), or `NEEDS pick_points`. Trials without `prompts.json` are skipped with a note.

**Default outputs:**

| Mode | LeRobot | Zarr |
|---|---|---|
| Single trial | `SmartData/<TRIAL>` | `<TRIAL>/<TRIAL>.zarr` |
| `--all` | `SmartData/combined` | `combined.zarr` |

---

## Data format

Both formats hold the same data. Each timestep has:

| Field | Shape | Contents |
|---|---|---|
| image | 256×256×3 | the RGB frame |
| state | 7 | position (tx, ty, tz) + orientation quaternion (qx, qy, qz, qw) |
| action | 7 | next-step position change (dx, dy, dz) + rotation change (drx, dry, drz) + gripper aperture |

Actions are **relative** (frame-to-frame changes), following CAP, so they carry over across environments. The gripper channel is the SAM 2-derived aperture: about 1 = open, about 0 = closed.

**LeRobotDataset v3** (`SmartData/combined`): Parquet tables for state/action, AV1-compressed MP4 for images, and JSON metadata for episode boundaries. Load it with `LeRobotDataset("SmartData/combined")`. Visualize it with `lerobot-dataset-viz`.

**Zarr** (`combined.zarr`), the Diffusion Policy / UMI layout:
```
combined.zarr/
  data/img            (N, 256, 256, 3) uint8
  data/state          (N, 7) float32
  data/action         (N, 7) float32
  meta/episode_ends   (E,) int64       cumulative end index of each episode
  .attrs["episode_names"]               trial folder of each episode, in order
```
All episodes are concatenated into one long array. `episode_ends` marks where each one stops: episode *k* covers steps `[episode_ends[k-1], episode_ends[k])`.

---

## What's in git

**Tracked:** raw trial recordings, small label files (`prompts.json`, `gripper_state.csv`, `contact_anchors.json`, `grasp_center.csv`, `contact_points.json`, `target_track.csv`), and code.

**Ignored** (see `.gitignore`): Zarr stores, `.rrd` files, `gripper_masks.npz`, preview videos/images, `.DS_Store`. These are large and can all be regenerated from the tracked files with `main.py`.

---

## Notes and limitations

- **Behavior cloning, not full CAP.** CAP also conditions the policy on a **3D contact anchor**. That requires *metric* depth (AnySense's depth `.bin` frames, not the visualization MP4) and camera intrinsics, neither of which is exported in the current trial folders. The schema can take an added contact-anchor field without a rebuild.
- **Trials that start with the gripper closed.** The contact detector looks for open→closed transitions, so a grasp already underway at frame 0 is not detected (example: `07_07_34` reports 0 contacts).
- **MolmoPoint vs. Moondream 2.** MolmoPoint-8B is about 36 GB at full precision, and its released code has no working half-precision path outside CUDA. Moondream 2 (about 2B parameters, about 4 GB) runs on Apple Silicon and is the default. CAP itself benchmarks several interchangeable pointing models.
- **Compute.** SAM 2 is the bottleneck: about 10–30 min per trial on CPU, a few minutes on a CUDA GPU. Everything else takes seconds to minutes.
- **Troubleshooting.**
  - `ModuleNotFoundError` almost always means the wrong conda env: run inspection scripts with `conda run -n lerobot ...`.
  - `FileNotFoundError: sam2/checkpoints/...` means `SAM2_CKPT` isn't exported in this terminal.
  - Black video panels in Rerun mean FFmpeg isn't on your `PATH`.

---

## Acknowledgments and references

This project builds directly on the following work and tools.

**Contact-Anchored Policies (CAP)** — the method this pipeline reproduces and extends.
> Cui, Z. J., Rayyan, O., Etukuru, H., Tan, B., Andrianarivo, Z., Teng, Z., Zhou, Y., Mehta, K., Wojno, N., Wu, K. Y., Anjaria, M. H., Wu, Z., Mao, M., Zhang, G., Shah, B., Kim, Y., Chintala, S., Pinto, L., & Shafiullah, N. M. M. *Contact-Anchored Policies: Contact Conditioning Creates Strong Robot Utility Models.* Project page: [cap-policy.github.io](https://cap-policy.github.io)

**SAM 2 (Segment Anything in Images and Videos)** — gripper and target segmentation/tracking.
> Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollár, P., & Feichtenhofer, C. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024. [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)

**Moondream 2** — vision-language object grounding (pointing).
> Korrapati, V. *Moondream: a small vision-language model.* [github.com/vikhyat/moondream](https://github.com/vikhyat/moondream) · Model: [huggingface.co/vikhyatk/moondream2](https://huggingface.co/vikhyatk/moondream2)

**Molmo / MolmoPoint** — the initially-selected grounding model (Allen Institute for AI).
> Deitke, M., et al. *Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models.* Allen Institute for AI (Ai2), 2024. · Model: [huggingface.co/allenai/MolmoPoint-8B](https://huggingface.co/allenai/MolmoPoint-8B) · [github.com/allenai/molmo2](https://github.com/allenai/molmo2)

**LeRobot** — dataset format and policy training (Hugging Face).
> Cadene, R., et al. *LeRobot: State-of-the-art machine learning for real-world robotics in PyTorch.* Hugging Face. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

**Rerun** — multimodal visualization.
> Rerun.io. *Rerun: an SDK and viewer for visualizing and interacting with multimodal data streams.* [rerun.io](https://rerun.io) · [github.com/rerun-io/rerun](https://github.com/rerun-io/rerun)

**AnySense** — the iOS data-capture application.
> Bhirangi, R., et al. *AnySense.* NYU Robot Learning Lab, 2024. [github.com/NYU-robot-learning/AnySense](https://github.com/NYU-robot-learning/AnySense)

**Record3D** — the streaming library AnySense's USB streaming derives from.
> Šimoník, M. *Record3D: streaming and recording of RGBD data from iOS devices.* [github.com/marek-simonik/record3d](https://github.com/marek-simonik/record3d)

**VQ-BeT** — the behavior-cloning policy used by CAP and available in LeRobot.
> Lee, S., Wang, Y., Etukuru, H., Kim, H. J., Shafiullah, N. M. M., & Pinto, L. *Behavior Generation with Latent Actions (VQ-BeT).* 2024.

---

## License

The code in this repository is released under the MIT License. Note that the third-party models and libraries above carry their own licenses (e.g. SAM 2 and Molmo are Apache-2.0; check each project). Datasets derived from your own recordings are yours.

*This pipeline was developed as an undergraduate research project. Contributions and issues welcome.*
