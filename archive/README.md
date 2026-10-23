# Archive

Scripts that the pipeline no longer uses, kept for reference.
The active scripts live in the repo root, because `main.py` runs them by filename from there.
Every version of every file is also in git history: `git log --follow <file>`.

## Old versions (replaced by a root-level script)

| File | Replaced by | What the newer version adds |
|---|---|---|
| controller/mainv1.py | main.py | `--all` mode with combined packaging |
| controller/main_preoct22.py | main.py | live output (`--no-capture-output`); stops if packaging fails |
| labeling/pick_points_preoct22.py | pick_points.py | help text corrected (Enter saves, not `s`) |
| visualization/build_gripper_rrd_old.py | build_gripper_rrd.py | grasp center, contact anchors, target markers |
| visualization/build_gripper_rrd2.py | build_gripper_rrd.py | per-frame tracked target objects |
| lerobot/convert_to_lerobot_old.py | convert_to_lerobot.py | `task` inside the frame dict; `--overwrite` flag |
| lerobot/convert_to_lerobotv2.py | convert_to_lerobot.py | `--overwrite` flag |
| zarr/convert_to_zarrV1.py | convert_to_zarr.py | accepts a single trial folder as `--root` |
| zarr/convert_to_zarr_preoct22.py | convert_to_zarr.py | records each episode's trial name |
| zarr/inspect_zarr_preoct22.py | inspect_zarr.py | prints trial names next to episodes |
| docs/README_preoct22.md | README.md | new-trial walkthrough, main.py reference, Zarr format |
| docs/README (1).md | README.md | (accidental duplicate of an older README) |

## Alternative not used by main.py

| File | Notes |
|---|---|
| grounding/ground_targets.py | MolmoPoint version of `ground_targets_moondream.py`. Needs a CUDA GPU with ~18 GB of memory. It doesn't depend on any other script, so it still runs from here: `conda run -n molmo python archive/grounding/ground_targets.py <TRIAL> "object"` |
