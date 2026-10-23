"""
SmartData pipeline controller.

Two modes:

  Single trial   python main.py <trial> [--ground OBJ ...] [--from STAGE --to STAGE]
                 Labels that one trial and packages just that trial.

  All trials     python main.py --all [--root .] [--ground OBJ ...]
                 Labels every trial that has a prompts.json (skip-if-exists),
                 then packages ALL labeled trials into ONE combined dataset.

Stages:
  LABELING (per-trial):   sam2 -> contact -> [ground -> track] -> rrd
  PACKAGING (per scope):  lerobot, zarr

pick_points is manual (the fingers move between trials), so it is never automated.
Run it first for each trial:
    conda run -n base python pick_points.py <trial>
In --all mode, trials without a prompts.json are skipped with a note.

Examples:
    python main.py 2026-07-10-07_07_34
    python main.py 2026-07-10-07_07_34 --ground "football" "water bottle"
    python main.py 2026-07-10-07_07_34 --from contact --to zarr
    python main.py 2026-07-10-07_07_34 --force sam2
    python main.py --all
    python main.py --all --root .
"""
import os
import sys
import glob
import argparse
import subprocess

# --- conda env per stage (edit to match your machine) ---
ENVS = {"base": "base", "molmo": "molmo", "lerobot": "lerobot"}

LABEL_STAGES = ["sam2", "contact", "ground", "track", "rrd"]
ALL_STAGES = LABEL_STAGES + ["lerobot", "zarr"]


# ---------------------------------------------------------------- helpers
def find(folder, pattern):
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits[0] if hits else None


def exists(folder, name):
    return os.path.exists(os.path.join(folder, name))


def is_raw_trial(folder):
    return (find(folder, "AR_Pose_*.txt")
            and (find(folder, "RGB_*.mp4") or find(folder, "RGB_*.mov")))


def run(env, script, argv, dry):
    cmd = ["conda", "run", "-n", ENVS[env], "python", script] + argv
    print(f"    $ {' '.join(cmd)}")
    return 0 if dry else subprocess.call(cmd)


def stage_done(stage, trial):
    return {
        "sam2": exists(trial, "gripper_state.csv"),
        "contact": exists(trial, "contact_anchors.json"),
        "ground": exists(trial, "contact_points.json"),
        "track": exists(trial, "target_track.csv"),
        "rrd": exists(trial, "phone_gripper_capture.rrd"),
    }.get(stage, False)


def run_label_stage(stage, trial, args, dry):
    if stage == "sam2":
        return run("base", "extract_gripper_sam2.py", [trial], dry)
    if stage == "contact":
        return run("base", "extract_contact_anchor.py", [trial], dry)
    if stage == "ground":
        return run("molmo", "ground_targets_moondream.py", [trial] + args.ground, dry)
    if stage == "track":
        return run("base", "extract_target_track.py", [trial], dry)
    if stage == "rrd":
        return run("base", "build_gripper_rrd.py", [trial], dry)
    return 0


def label_one_trial(trial, args, stages):
    """Run the selected labeling stages for one trial. Returns True on success."""
    forced = (lambda s: args.force == "ALL" or args.force == s)
    for stage in stages:
        if stage in ("ground", "track") and not args.ground:
            continue
        if stage == "rrd" and args.no_rrd:
            continue
        if stage == "sam2" and not exists(trial, "prompts.json"):
            print(f"  !! {trial}: no prompts.json — run pick_points.py first; skipping trial.")
            return False
        if stage_done(stage, trial) and not forced(stage):
            print(f"  {stage:8s} skip (exists)")
            continue
        print(f"  ==> {stage}")
        rc = run_label_stage(stage, trial, args, args.dry_run)
        if rc != 0:
            print(f"  !! stage '{stage}' failed on {trial} (exit {rc}).")
            return False
    return True


# ---------------------------------------------------------------- packaging
def package(scope_path, repo_id, zarr_out, args):
    """Package a scope (single trial folder OR a root of trials) into datasets."""
    if not args.no_lerobot:
        print("  ==> lerobot")
        # a single trial uses the single-trial converter; a root uses the batch one
        if is_raw_trial(scope_path):
            run("lerobot", "convert_to_lerobot.py",
                [scope_path, "--repo-id", repo_id, "--overwrite"], args.dry_run)
        else:
            run("lerobot", "convert_all_to_lerobot.py",
                ["--root", scope_path, "--repo-id", repo_id, "--overwrite"], args.dry_run)
    if not args.no_zarr:
        print("  ==> zarr")
        run("lerobot", "convert_to_zarr.py",
            ["--root", scope_path, "--out", zarr_out], args.dry_run)


# ---------------------------------------------------------------- modes
def run_single(args):
    trial = args.trial.rstrip("/")
    if not os.path.isdir(trial):
        raise SystemExit(f"Trial folder not found: {trial}")
    tname = os.path.basename(trial)

    lo, hi = ALL_STAGES.index(args.from_stage), ALL_STAGES.index(args.to_stage)
    selected = ALL_STAGES[lo:hi + 1]
    label_sel = [s for s in selected if s in LABEL_STAGES]
    do_pack = any(s in ("lerobot", "zarr") for s in selected)

    print(f"\nSingle-trial run: {tname}")
    print(f"  labeling stages: {label_sel or '(none)'}")
    print(f"  packaging: {'yes' if do_pack else 'no'}\n")

    if label_sel and not label_one_trial(trial, args, label_sel):
        raise SystemExit("Labeling stopped; not packaging.")

    if do_pack:
        repo_id = args.repo_id or f"SmartData/{tname}"
        zarr_out = args.zarr or os.path.join(trial, f"{tname}.zarr")
        package(trial, repo_id, zarr_out, args)
    print("\nDone.")


def run_all(args):
    root = args.root.rstrip("/")
    trials = sorted(d for d in glob.glob(os.path.join(root, "*"))
                    if os.path.isdir(d) and is_raw_trial(d))
    if not trials:
        raise SystemExit(f"No trial folders found under {root!r}.")

    print(f"\n--all run over {len(trials)} trial(s) under {root!r}:")
    for t in trials:
        state = "labeled" if exists(t, "gripper_state.csv") else \
                ("ready" if exists(t, "prompts.json") else "NEEDS pick_points")
        print(f"  {os.path.basename(t):26s} [{state}]")
    print()

    # 1) label each trial (skip-if-exists inside)
    labeled = []
    for t in trials:
        print(f"--- labeling {os.path.basename(t)} ---")
        ok = label_one_trial(t, args, LABEL_STAGES)
        if exists(t, "gripper_state.csv"):
            labeled.append(t)

    if not labeled:
        raise SystemExit("No labeled trials to package.")

    # 2) package ALL labeled trials into one combined dataset
    print(f"\n--- packaging {len(labeled)} labeled trial(s) combined ---")
    repo_id = args.repo_id or "SmartData/combined"
    zarr_out = args.zarr or os.path.join(root, "combined.zarr")
    package(root, repo_id, zarr_out, args)
    print("\nDone.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial", nargs="?", help="trial folder (single-trial mode)")
    ap.add_argument("--all", action="store_true", help="process every trial under --root")
    ap.add_argument("--root", default=".", help="root of trial folders (--all mode)")
    ap.add_argument("--ground", metavar="OBJECT", nargs="+", default=None,
                    help='grounding target(s), e.g. --ground "football" "water bottle"')
    ap.add_argument("--repo-id", default=None, help="LeRobot repo id (override default)")
    ap.add_argument("--zarr", default=None, help="output .zarr path (override default)")
    ap.add_argument("--no-rrd", action="store_true", help="skip the Rerun .rrd")
    ap.add_argument("--no-lerobot", action="store_true", help="skip LeRobot conversion")
    ap.add_argument("--no-zarr", action="store_true", help="skip Zarr conversion")
    ap.add_argument("--from", dest="from_stage", choices=ALL_STAGES, default="sam2")
    ap.add_argument("--to", dest="to_stage", choices=ALL_STAGES, default="zarr")
    ap.add_argument("--force", nargs="?", const="ALL", default=None, metavar="STAGE",
                    help="rerun even if outputs exist; optionally name one stage")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = ap.parse_args()

    if args.all:
        run_all(args)
    elif args.trial:
        run_single(args)
    else:
        ap.error("give a trial folder, or use --all")


if __name__ == "__main__":
    main()
