"""
SmartData pipeline controller.

Runs the full per-trial chain, each stage in its correct conda environment via
`conda run`, from raw AnySense files all the way to LeRobot + Zarr datasets.

Stages (in order):
    sam2      finger masks + aperture     -> gripper_masks.npz, gripper_state.csv   [env: base]
    contact   hindsight contact anchors   -> contact_anchors.json, grasp_center.csv [env: base]
    ground    Moondream target point      -> contact_points.json     (only with --ground) [env: molmo]
    track     SAM2 target tracking        -> target_track.csv         (only with --ground) [env: base]
    rrd       synchronized Rerun file     -> phone_gripper_capture.rrd [env: base]
    lerobot   LeRobotDataset episode      -> ~/.cache/.../<repo-id>    [env: lerobot]
    zarr      Diffusion-Policy .zarr      -> <out>.zarr               [env: lerobot]

pick_points is intentionally NOT automated: it needs you to click the fingers,
and the phone position varies per trial. Run it yourself first:
    conda run -n base python pick_points.py <trial>

Examples:
    python main.py 2026-07-10-07_07_48
    python main.py 2026-07-10-07_07_48 --ground "football" --repo-id me/grip --zarr grip.zarr
    python main.py 2026-07-10-07_07_48 --force sam2
    python main.py 2026-07-10-07_07_48 --from contact --to rrd
"""
import os
import sys
import glob
import argparse
import subprocess

# --- which conda env each stage runs in (edit names to match your machine) ---
ENVS = {"base": "base", "molmo": "molmo", "lerobot": "lerobot"}

STAGE_ORDER = ["sam2", "contact", "ground", "track", "rrd", "lerobot", "zarr"]


def find(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


def exists(folder, name):
    return os.path.exists(os.path.join(folder, name))


def run(env, script, args, dry):
    cmd = ["conda", "run", "-n", ENVS[env], "python", script] + args
    print(f"    $ {' '.join(cmd)}")
    if dry:
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial", help="trial folder, e.g. 2026-07-10-07_07_48")
    ap.add_argument("--ground", metavar="OBJECT", nargs="+", default=None,
                    help='run grounding + tracking for one or more targets, e.g. '
                         '--ground "football" "water bottle". Omit to skip grounding.')
    ap.add_argument("--repo-id", default=None,
                    help="LeRobot repo id (default: SmartData/<trial>)")
    ap.add_argument("--zarr", default=None,
                    help="output .zarr path (default: <trial>/<trial>.zarr)")
    ap.add_argument("--no-rrd", action="store_true", help="skip building the Rerun .rrd")
    ap.add_argument("--no-lerobot", action="store_true", help="skip the LeRobot conversion")
    ap.add_argument("--no-zarr", action="store_true", help="skip the Zarr conversion")
    ap.add_argument("--from", dest="from_stage", choices=STAGE_ORDER, default="sam2")
    ap.add_argument("--to", dest="to_stage", choices=STAGE_ORDER, default="zarr")
    ap.add_argument("--force", nargs="?", const="ALL", default=None,
                    metavar="STAGE",
                    help="rerun even if outputs exist; optionally name one stage")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    trial = args.trial.rstrip("/")
    if not os.path.isdir(trial):
        raise SystemExit(f"Trial folder not found: {trial}")

    # Both formats run by default; derive names if not given.
    tname = os.path.basename(trial)
    if not args.no_lerobot and not args.repo_id:
        args.repo_id = f"SmartData/{tname}"
    if not args.no_zarr and not args.zarr:
        args.zarr = os.path.join(trial, f"{tname}.zarr")

    def forced(stage):
        return args.force == "ALL" or args.force == stage

    lo, hi = STAGE_ORDER.index(args.from_stage), STAGE_ORDER.index(args.to_stage)
    selected = STAGE_ORDER[lo:hi + 1]

    # decide, per stage, whether it will run / skip / be excluded
    plan = []
    for stage in selected:
        if stage in ("ground", "track") and not args.ground:
            plan.append((stage, "excluded (no --ground)")); continue
        if stage == "rrd" and args.no_rrd:
            plan.append((stage, "excluded (--no-rrd)")); continue
        if stage == "lerobot" and args.no_lerobot:
            plan.append((stage, "excluded (--no-lerobot)")); continue
        if stage == "zarr" and args.no_zarr:
            plan.append((stage, "excluded (--no-zarr)")); continue

        # skip-if-exists check (unless forced)
        done = {
            "sam2": exists(trial, "gripper_state.csv"),
            "contact": exists(trial, "contact_anchors.json"),
            "ground": exists(trial, "contact_points.json"),
            "track": exists(trial, "target_track.csv"),
            "rrd": exists(trial, "phone_gripper_capture.rrd"),
            "lerobot": False,   # dataset lives outside the trial folder; always attempt
            "zarr": args.zarr and os.path.exists(args.zarr),
        }.get(stage, False)
        if done and not forced(stage):
            plan.append((stage, "skip (output exists; --force to redo)"))
        else:
            plan.append((stage, "RUN"))

    print(f"\nPipeline plan for '{trial}':")
    for stage, status in plan:
        print(f"  {stage:8s} -> {status}")
    print()

    # sanity: sam2 needs prompts.json from the manual pick_points step
    if any(s == "sam2" and st == "RUN" for s, st in plan) and not exists(trial, "prompts.json"):
        print("!! Missing prompts.json — run the manual step first:")
        print(f"     conda run -n {ENVS['base']} python pick_points.py {trial}")
        if not args.dry_run:
            raise SystemExit(1)

    # execute
    for stage, status in plan:
        if status != "RUN":
            continue
        print(f"==> {stage}")
        if stage == "sam2":
            rc = run("base", "extract_gripper_sam2.py", [trial], args.dry_run)
        elif stage == "contact":
            rc = run("base", "extract_contact_anchor.py", [trial], args.dry_run)
        elif stage == "ground":
            rc = run("molmo", "ground_targets_moondream.py", [trial] + args.ground, args.dry_run)
        elif stage == "track":
            rc = run("base", "extract_target_track.py", [trial], args.dry_run)
        elif stage == "rrd":
            rc = run("base", "build_gripper_rrd.py", [trial], args.dry_run)
        elif stage == "lerobot":
            rc = run("lerobot", "convert_to_lerobot.py",
                     [trial, "--repo-id", args.repo_id, "--overwrite"], args.dry_run)
        elif stage == "zarr":
            rc = run("lerobot", "convert_to_zarr.py",
                     ["--root", trial, "--out", args.zarr], args.dry_run)
        else:
            rc = 0
        if rc != 0:
            raise SystemExit(f"Stage '{stage}' failed (exit {rc}). Stopping.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
