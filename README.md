# SmartData

**A pipeline for turning handheld-gripper iPhone recordings into visualized, labeled, training-ready robot-manipulation data.**

SmartData takes raw recordings from an iPhone-mounted handheld gripper — RGB video, depth, and 6-DoF camera pose — and processes them into a fully labeled, synchronized dataset suitable for training imitation-learning policies. It combines video object segmentation, vision-language grounding, contact labeling, interactive visualization, and conversion to the LeRobot dataset format.

The project reproduces the data-collection and labeling half of the **Contact-Anchored Policies (CAP)** pipeline and extends it with an interactive [Rerun](https://rerun.io) visualization layer.

---

## Overview

The data comes from the [AnySense](https://github.com/NYU-robot-learning/AnySense) iOS app running on an iPhone rigidly mounted to a handheld 2-finger gripper. Each trial produces:

- `RGB_*.mp4` — the camera view
- `Depth_*.mp4` — a depth visualization video
- `AR_Pose_*.txt` — timestamped 6-DoF camera/gripper pose (quaternion + translation) from ARKit visual-inertial odometry at ~30 Hz

From these, SmartData derives:

- **Gripper finger segmentation** and a continuous **gripper aperture (open/closed) signal**, using SAM 2.
- **Hindsight contact anchors** — the moment and location where the gripper closed on an object.
- **Target-object grounding** — the object being reached for, located by a vision-language model and tracked through the video.
- A synchronized **Rerun (`.rrd`) recording** tying every stream to one scrubbable timeline.
- A **LeRobotDataset** ready for behavior-cloning policy training.

### Pipeline

```
AnySense capture (RGB + Depth + Pose)
        │
        ├── SAM 2 ─────────► finger masks + aperture signal
        │                          │
        │                    hindsight contact anchors
        │
        ├── Moondream 2 ────► target object point (frame 0)
        │        └── SAM 2 ─► target tracked across all frames
        │
        ├── Rerun ──────────► synchronized visualization (.rrd)
        │
        └── LeRobot conversion ─► training-ready dataset ─► (VQ-BeT policy)
```

---

## Repository structure

| Script | Purpose |
|---|---|
| `pick_points.py` | Interactively mark the two gripper fingers on frame 0 (SAM 2 prompt). |
| `extract_gripper_sam2.py` | Propagate finger masks through the video; compute the aperture signal. |
| `extract_contact_anchor.py` | Hindsight contact labeling — find contact frames and the 2D contact point. |
| `ground_targets_moondream.py` | Locate a named target object with Moondream 2 (frame 0). |
| `ground_targets.py` | MolmoPoint variant of the above (requires CUDA — see notes). |
| `extract_target_track.py` | Track the grounded target object through the video with SAM 2. |
| `build_gripper_rrd.py` | Assemble all streams into a synchronized Rerun `.rrd`. |
| `convert_to_lerobot.py` | Convert one trial into a LeRobotDataset episode. |
| `convert_all_to_lerobot.py` | Batch-convert all trials into one multi-episode dataset. |
| `inspect_dataset.py` | Validate a converted dataset (action scale, gripper signal, image format). |

---

## Installation

The pipeline spans a few tools with conflicting dependencies, so it uses separate conda environments.

**Base environment** (SAM 2, Rerun, contact labeling):
```bash
conda create -n base-env python=3.11 && conda activate base-env
# SAM 2
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd checkpoints && ./download_ckpts.sh && cd ../..
pip install rerun-sdk numpy opencv-python matplotlib
export SAM2_CKPT=/path/to/sam2/checkpoints/sam2.1_hiera_tiny.pt
export SAM2_CFG=configs/sam2.1/sam2.1_hiera_t.yaml
```

**Grounding environment** (Moondream 2):
```bash
conda create -n molmo python=3.11 && conda activate molmo
pip install transformers torch pillow einops torchvision accelerate opencv-python numpy
```

**LeRobot environment** (dataset conversion + training):
```bash
conda create -n lerobot python=3.11 && conda activate lerobot
pip install lerobot opencv-python numpy
```

You will also need **FFmpeg** on your `PATH` for Rerun to decode the H.264/HEVC videos:
```bash
brew install ffmpeg   # macOS
```

---

## Usage

Organize trials as subfolders inside a root directory (e.g. `rerun_atomic/`), each containing that trial's `RGB_*`, `Depth_*`, and `AR_Pose_*` files. Run the scripts from the root, passing the trial folder name.

```bash
# 1. Segment the gripper fingers and derive the aperture signal  (base-env)
python pick_points.py <TRIAL>              # click fingers on frame 0, press Enter
python extract_gripper_sam2.py <TRIAL>     # produces gripper_masks.npz, gripper_state.csv

# 2. Hindsight contact anchors  (base-env)
python extract_contact_anchor.py <TRIAL>

# 3. Ground and track the target object  (molmo env, then base-env)
python ground_targets_moondream.py <TRIAL> "football" "water bottle"
python extract_target_track.py <TRIAL>

# 4. Visualize everything on one timeline  (base-env)
python build_gripper_rrd.py <TRIAL>
rerun <TRIAL>/phone_gripper_capture.rrd

# 5. Convert to a LeRobotDataset  (lerobot env)
python convert_all_to_lerobot.py --repo-id <user>/<name> --overwrite
python inspect_dataset.py <user>/<name>
```

---

## Data format

Each trial is converted into a [LeRobotDataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) episode with:

| Field | Shape | Contents |
|---|---|---|
| `observation.images.gripper_cam` | 256×256×3 (video) | the RGB frame |
| `observation.state` | 7 | position (tx,ty,tz) + quaternion (qx,qy,qz,qw) |
| `action` | 7 | next-step position delta + rotation delta + gripper aperture |

Actions are **relative** (frame-to-frame deltas), following CAP, so they transfer across environments. The gripper channel is the SAM 2-derived aperture.

---

## Notes and limitations

- **This produces a behavior-cloning dataset, not full CAP.** CAP additionally conditions the policy on a **3D contact anchor**, which requires *metric depth* (not the depth visualization video) and *camera intrinsics* — neither is currently exported from the offline AnySense files. The current schema is a strict subset of the CAP version, so a contact-anchor input can be added later without rebuilding.
- **MolmoPoint vs. Moondream 2.** MolmoPoint-8B was the original grounding model but ships in F32 (~36 GB) and its released code lacks a working half-precision path outside CUDA, making it impractical on Apple Silicon. Moondream 2 (~2B) was adopted as a lightweight, pointing-native alternative that runs locally. CAP itself benchmarks multiple interchangeable pointing models, so this is a supported substitution.
- **Compute.** The SAM 2 passes are the bottleneck (~30 min/trial on CPU, a few minutes on a CUDA GPU). Grounding and conversion are lightweight.

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