"""
Build a synchronized Rerun recording (.rrd) for a phone/gripper trial.

USAGE (from inside rerun_atomic):
    python build_gripper_rrd.py 2026-07-10-07_07_48
    rerun 2026-07-10-07_07_48/phone_gripper_capture.rrd

Auto-discovers, per trial folder:
    AR_Pose_*.txt        -> 6-DoF pose, trajectory, pose signals   (REAL)
    RGB_*.(mp4|mov)      -> RGB video                              (REAL)
    Depth_*.(mp4|mov)    -> depth visualization video              (REAL)
    gripper_masks.npz    -> SAM2 finger masks (SegmentationImage)  (REAL if present)
    gripper_state.csv    -> aperture / open-close signal           (REAL if present)
If the SAM2 outputs are missing, segmentation + state fall back to synthetic
placeholders so the file still builds.
"""
import os
import sys
import glob
import json
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

TRIAL_DIR = sys.argv[1] if len(sys.argv) > 1 else "."


def first(patterns):
    hits = []
    for p in patterns:
        hits += glob.glob(os.path.join(TRIAL_DIR, p))
    hits = sorted(hits)
    return hits[0] if hits else None


POSE_TXT = first(["AR_Pose_*.txt"])
RGB_VIDEO = first([f"RGB_*.{e}" for e in ("mp4", "MP4", "mov", "MOV")])
DEPTH_VIDEO = first([f"Depth_*.{e}" for e in ("mp4", "MP4", "mov", "MOV")])
MASKS_NPZ = first(["gripper_masks.npz"])
GRASP_CSV = first(["grasp_center.csv"])
ANCHORS_JSON = first(["contact_anchors.json"])
MOLMO_JSON = first(["contact_points.json"])
STATE_CSV = first(["gripper_state.csv"])
OUT_RRD = os.path.join(TRIAL_DIR, "phone_gripper_capture.rrd")

if not POSE_TXT:
    raise SystemExit(f"No AR_Pose_*.txt in {TRIAL_DIR!r} (pass the trial folder as the argument).")

print(f"pose  : {POSE_TXT}")
print(f"rgb   : {RGB_VIDEO}")
print(f"depth : {DEPTH_VIDEO}")
print(f"masks : {MASKS_NPZ if MASKS_NPZ else '(none - synthetic segmentation)'}")
print(f"state : {STATE_CSV if STATE_CSV else '(none - synthetic square wave)'}")
print(f"grasp : {GRASP_CSV if GRASP_CSV else '(none)'}")
print(f"anchor: {ANCHORS_JSON if ANCHORS_JSON else '(none)'}")
print(f"molmo : {MOLMO_JSON if MOLMO_JSON else '(none)'}")

# ---- config ----
VIDEO_OFFSET_S = 0.0
QUAT_ORDER = "xyzw"        # flip to "wxyz" if orientation looks wrong
IMG_H, IMG_W = 120, 160    # synthetic fallback resolution only

RGB_ENTITY = "world/phone_gripper/camera/rgb"
DEPTH_ENTITY = "world/phone_gripper/camera/depth"
SEG_ENTITY = "world/phone_gripper/camera/segmentation"


def parse_pose_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            ts_ms = int(p[0].strip().strip('"').strip("<>"))
            a, b, c, d, tx, ty, tz = (float(x) for x in p[1:8])
            quat = (a, b, c, d) if QUAT_ORDER == "xyzw" else (b, c, d, a)
            rows.append((ts_ms, quat, (tx, ty, tz)))
    return rows


def load_state_csv(path):
    ap, st = [], []
    with open(path) as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, a, s = line.split(",")
            ap.append(float(a))
            st.append(s)
    return ap, st



def load_grasp_csv(path):
    pts = []
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3 or parts[1] == "":
                pts.append(None)
            else:
                pts.append((float(parts[1]), float(parts[2])))
    return pts


def rgb_native_size(path):
    import cv2
    cap = cv2.VideoCapture(path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return (int(w), int(h)) if w and h else None


def log_video_assetvideo(entity, path, offset_s):
    video = rr.AssetVideo(path=path)
    rr.log(entity, video, static=True)
    ts_ns = np.asarray(video.read_frame_timestamps_nanos())
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn("capture_time", duration=1e-9 * ts_ns + offset_s)],
        columns=rr.VideoFrameReference.columns_nanos(ts_ns),
    )
    return len(ts_ns)


def log_video_cv2(entity, path, rows, offset_s):
    import cv2
    cap = cv2.VideoCapture(path)
    t0 = rows[0][0]
    i = 0
    while i < len(rows):
        ok, frame = cap.read()
        if not ok:
            break
        rr.set_time("capture_time", duration=(rows[i][0] - t0) / 1000.0 + offset_s)
        rr.log(entity, rr.Image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
        i += 1
    cap.release()
    return i


def log_video(entity, path, offset_s, rows):
    try:
        n = log_video_assetvideo(entity, path, offset_s)
        print(f"  {entity}: AssetVideo, {n} frame refs")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {entity}: AssetVideo failed ({e}); trying cv2 fallback...")
        try:
            n = log_video_cv2(entity, path, rows, offset_s)
            print(f"  {entity}: cv2 fallback, {n} frames")
            return True
        except Exception as e2:  # noqa: BLE001
            print(f"  {entity}: video failed ({e2})")
            return False


# synthetic fallbacks (only used if the real streams are absent)
def synth_depth_mm(phase):
    ys, xs = np.mgrid[0:IMG_H, 0:IMG_W]
    r = np.sqrt((xs - IMG_W / 2) ** 2 + (ys - IMG_H / 2) ** 2) / max(IMG_W, IMG_H) * 2
    return ((0.5 + 2.0 * r + 0.2 * np.sin(phase)) * 1000).astype(np.uint16)


def synth_seg(bx, by, is_open):
    m = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    half = 12 if is_open else 6
    m[max(0, by - half):by + half, max(0, bx - half):bx + half] = 1
    return m


def main():
    rows = parse_pose_file(POSE_TXT)
    if not rows:
        raise SystemExit("No pose rows parsed.")
    t0_ms = rows[0][0]
    positions = np.array([r[2] for r in rows], dtype=np.float32)
    span = positions.max(0) - positions.min(0)
    span[span == 0] = 1.0
    tn = (positions - positions.min(0)) / span

    # load SAM2 outputs if present
    masks = None
    if MASKS_NPZ:
        masks = np.load(MASKS_NPZ)["masks"]
        print(f"Loaded masks {masks.shape}")
    grasp = load_grasp_csv(GRASP_CSV) if GRASP_CSV else None
    anchors = json.load(open(ANCHORS_JSON)) if ANCHORS_JSON else None
    molmo = json.load(open(MOLMO_JSON)) if MOLMO_JSON else None

    aperture = state_txt = None
    if STATE_CSV:
        aperture, state_txt = load_state_csv(STATE_CSV)
        print(f"Loaded {len(aperture)} gripper-state rows")

    rr.init("phone_gripper_capture")

    blueprint = None
    try:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", name="3D: pose + trajectory"),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin=RGB_ENTITY, name="RGB"),
                        rrb.Spatial2DView(origin=DEPTH_ENTITY, name="Depth"),
                        rrb.Spatial2DView(origin=SEG_ENTITY, name="Gripper segmentation"),
                    ),
                    rrb.TimeSeriesView(origin="signals", name="Pose signals (t, q)"),
                    rrb.TimeSeriesView(origin="gripper", name="Gripper aperture / state"),
                    rrb.TextLogView(origin="gripper/state_text", name="Gripper log"),
                ),
                column_shares=[3, 4],
            ),
            collapse_panels=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"(blueprint skipped: {e})")

    try:
        rr.save(OUT_RRD, default_blueprint=blueprint)
    except TypeError:
        rr.save(OUT_RRD)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log("world/trajectory", rr.LineStrips3D([positions]), static=True)
    rr.log("world/phone_gripper", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[0.12, 0, 0], [0, 0.12, 0], [0, 0, 0.12]],
        colors=[[255, 60, 60], [60, 255, 60], [60, 120, 255]],
    ), static=True)
    rr.log("world/phone_gripper/camera", rr.Pinhole(
        focal_length=IMG_W, width=IMG_W, height=IMG_H), static=True)
    rr.log(SEG_ENTITY, rr.AnnotationContext([
        (0, "background", (0, 0, 0)),
        (1, "left finger", (0, 220, 120)),
        (2, "right finger", (0, 150, 255)),
    ]), static=True)

    print("Logging videos:")
    if RGB_VIDEO:
        log_video(RGB_ENTITY, RGB_VIDEO, VIDEO_OFFSET_S, rows)
    depth_ok = log_video(DEPTH_ENTITY, DEPTH_VIDEO, VIDEO_OFFSET_S, rows) if DEPTH_VIDEO else False

    # Scale factor: masks/anchors are in processed resolution, RGB is native.
    sx = sy = 1.0
    if RGB_VIDEO and masks is not None:
        native = rgb_native_size(RGB_VIDEO)
        if native:
            sx = native[0] / masks.shape[2]
            sy = native[1] / masks.shape[1]
            print(f"point scale: masks {masks.shape[2]}x{masks.shape[1]} -> rgb {native[0]}x{native[1]}"
                  f"  (x{sx:.2f})")

    # MolmoPoint predictions: static markers (they describe the whole episode).
    if molmo:
        pts, labels = [], []
        for prompt, plist in molmo.get("points", {}).items():
            for (x, y) in plist:
                pts.append([x, y])
                labels.append(prompt)
        if pts:
            rr.log(RGB_ENTITY + "/molmo_targets",
                   rr.Points2D(pts, labels=labels, radii=8.0,
                               colors=[[255, 200, 0]] * len(pts)),
                   static=True)
            print(f"Logged {len(pts)} MolmoPoint target(s)")

    # Hindsight contact anchors: one static marker per contact event.
    if anchors and anchors.get("events"):
        apts = [[e["x"] * sx, e["y"] * sy] for e in anchors["events"]
                if e.get("x") is not None]
        if apts:
            rr.log(RGB_ENTITY + "/contact_anchors",
                   rr.Points2D(apts, radii=10.0,
                               colors=[[255, 40, 40]] * len(apts),
                               labels=[f"contact @ {e['contact_frame']}"
                                       for e in anchors["events"]
                                       if e.get("x") is not None]),
                   static=True)
            print(f"Logged {len(apts)} hindsight contact anchor(s)")
    contact_frames = ({e["contact_frame"] for e in anchors["events"]}
                      if anchors else set())

    prev_state = None
    for i, (ts_ms, quat, trans) in enumerate(rows):
        t = (ts_ms - t0_ms) / 1000.0
        rr.set_time("capture_time", duration=t)

        rr.log("world/phone_gripper", rr.Transform3D(
            translation=list(trans), rotation=rr.Quaternion(xyzw=list(quat))))

        tx, ty, tz = trans
        qx, qy, qz, qw = quat
        rr.log("signals/translation/x", rr.Scalars(tx))
        rr.log("signals/translation/y", rr.Scalars(ty))
        rr.log("signals/translation/z", rr.Scalars(tz))
        rr.log("signals/quaternion/x", rr.Scalars(qx))
        rr.log("signals/quaternion/y", rr.Scalars(qy))
        rr.log("signals/quaternion/z", rr.Scalars(qz))
        rr.log("signals/quaternion/w", rr.Scalars(qw))

        # ---- gripper state (REAL from SAM2 if available) ----
        if aperture is not None:
            j = min(i, len(aperture) - 1)
            ap = aperture[j]
            txt = state_txt[j]
            rr.log("gripper/opening_width", rr.Scalars(ap))
            rr.log("gripper/state_numeric", rr.Scalars(1.0 if txt == "OPEN" else 0.0))
        else:
            is_open = (t % 4.0) < 2.0
            ap = 0.08 if is_open else 0.0
            txt = "OPEN" if is_open else "CLOSED"
            rr.log("gripper/opening_width", rr.Scalars(ap))
            rr.log("gripper/state_numeric", rr.Scalars(1.0 if is_open else 0.0))
        if txt != prev_state:
            rr.log("gripper/state_text", rr.TextLog(txt))
            prev_state = txt

        # ---- grasp center (midpoint between the fingers) ----
        if grasp is not None:
            g = grasp[min(i, len(grasp) - 1)]
            if g is not None:
                rr.log(RGB_ENTITY + "/grasp_center",
                       rr.Points2D([[g[0] * sx, g[1] * sy]], radii=6.0,
                                   colors=[[60, 255, 160]]))
        rr.log("gripper/contact_event",
               rr.Scalars(1.0 if i in contact_frames else 0.0))

        # ---- segmentation (REAL from SAM2 if available) ----
        if masks is not None:
            j = min(i, len(masks) - 1)
            rr.log(SEG_ENTITY, rr.SegmentationImage(masks[j]))
        else:
            bx = int(tn[i, 0] * (IMG_W - 1))
            by = int((1 - tn[i, 1]) * (IMG_H - 1))
            rr.log(SEG_ENTITY, rr.SegmentationImage(synth_seg(bx, by, txt == "OPEN")))

        # ---- depth fallback only if no real depth video ----
        if not depth_ok:
            rr.log(DEPTH_ENTITY, rr.DepthImage(synth_depth_mm(t * 2.0), meter=1000.0))

    print(f"Wrote {OUT_RRD}. Open with:  rerun {OUT_RRD}")


if __name__ == "__main__":
    main()
