"""
Hindsight contact labeling from the SAM2 outputs (CAP, Cui et al. Sec. 3.1.4).

CAP defines the contact frame as the moment the gripper aperture *ceases to
decrease* -- the fingers have closed onto the object and halted against its
geometry. The contact anchor is then the point centered between the two
fingers at that instant.

This script reads what extract_gripper_sam2.py already produced and needs no
per-trial configuration:
    gripper_state.csv   -> aperture per frame
    gripper_masks.npz   -> finger masks (1 = left, 2 = right)

Outputs into the trial folder:
    grasp_center.csv     frame_idx, cx, cy   (midpoint between finger centroids)
    contact_anchors.json contact events: {contact_frame, x, y, aperture, start, end}
    contact_preview.png  the contact frame with the anchor drawn, for a visual check

Usage:
    python extract_contact_anchor.py <trial_folder>
"""
import os
import sys
import glob
import json
import numpy as np
import cv2

SMOOTH_WIN = 9        # moving-average window on the aperture (frames)
CLOSE_THRESH = 0.5    # aperture below this counts as "closed"
FLAT_EPS = 0.002      # per-frame change below this counts as "no longer decreasing"
FLAT_RUN = 4          # consecutive flat frames required to call contact
MIN_DROP = 0.10       # ignore closing events with less total aperture drop than this


def find_video(folder):
    for ext in ("mp4", "MP4", "mov", "MOV"):
        hits = sorted(glob.glob(os.path.join(folder, f"RGB_*.{ext}")))
        if hits:
            return hits[0]
    return None


def load_aperture(path):
    ap = []
    with open(path) as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if line:
                ap.append(float(line.split(",")[1]))
    return np.asarray(ap, dtype=np.float32)


def smooth(a, win):
    if win <= 1:
        return a
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(a, k, mode="same")


def centroid(mask_eq):
    ys, xs = np.nonzero(mask_eq)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], dtype=np.float32)


def grasp_centers(masks):
    """Midpoint between the two finger centroids, per frame (NaN where missing)."""
    n = len(masks)
    out = np.full((n, 2), np.nan, dtype=np.float32)
    for i in range(n):
        cL = centroid(masks[i] == 1)
        cR = centroid(masks[i] == 2)
        if cL is not None and cR is not None:
            out[i] = (cL + cR) / 2.0
    return out


def find_contact_frames(ap):
    """
    Locate contact events. For each closing episode (aperture crossing the
    threshold downward), walk forward while the aperture keeps decreasing;
    contact is where the decrease flattens out.
    """
    s = smooth(ap, SMOOTH_WIN)
    d = np.diff(s, prepend=s[0])
    closed = s < CLOSE_THRESH

    events = []
    i = 1
    n = len(s)
    while i < n:
        # downward crossing of the threshold = start of a closing episode
        if closed[i] and not closed[i - 1]:
            start = i
            # walk back to where the decrease actually began
            j = i
            while j > 0 and d[j] < 0:
                j -= 1
            start_decline = j
            # walk forward to where it stops decreasing
            k = i
            flat = 0
            while k < n - 1:
                if d[k] > -FLAT_EPS:
                    flat += 1
                    if flat >= FLAT_RUN:
                        break
                else:
                    flat = 0
                k += 1
            contact = k
            drop = float(s[start_decline] - s[contact])
            # find where it reopens, to bound the event
            r = contact
            while r < n - 1 and closed[r]:
                r += 1
            if drop >= MIN_DROP:
                events.append({
                    "contact_frame": int(contact),
                    "decline_start": int(start_decline),
                    "release_frame": int(r),
                    "aperture_at_contact": float(ap[contact]),
                    "aperture_drop": drop,
                })
            i = max(r, contact + 1)
        else:
            i += 1
    return events


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python extract_contact_anchor.py <trial_folder>")
    folder = sys.argv[1]

    state_csv = os.path.join(folder, "gripper_state.csv")
    masks_npz = os.path.join(folder, "gripper_masks.npz")
    for p in (state_csv, masks_npz):
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p} -- run extract_gripper_sam2.py first.")

    ap = load_aperture(state_csv)
    z = np.load(masks_npz)
    masks = z["masks"]
    H, W = masks.shape[1], masks.shape[2]
    print(f"{len(ap)} aperture samples, masks {masks.shape}")

    centers = grasp_centers(masks)
    good = np.isfinite(centers[:, 0]).sum()
    print(f"Grasp center found on {good}/{len(centers)} frames")

    csv_path = os.path.join(folder, "grasp_center.csv")
    with open(csv_path, "w") as f:
        f.write("frame_idx,cx,cy\n")
        for i, (cx, cy) in enumerate(centers):
            if np.isfinite(cx):
                f.write(f"{i},{cx:.2f},{cy:.2f}\n")
            else:
                f.write(f"{i},,\n")
    print("Wrote", csv_path)

    events = find_contact_frames(ap)
    for e in events:
        c = e["contact_frame"]
        pt = centers[c]
        if np.isfinite(pt[0]):
            e["x"], e["y"] = float(pt[0]), float(pt[1])
        else:
            e["x"] = e["y"] = None
    print(f"Detected {len(events)} contact event(s):")
    for e in events:
        print(f"  frame {e['contact_frame']:5d}  anchor=({e['x']}, {e['y']})  "
              f"drop={e['aperture_drop']:.2f}")

    json_path = os.path.join(folder, "contact_anchors.json")
    with open(json_path, "w") as f:
        json.dump({"frame_size": [int(W), int(H)], "events": events}, f, indent=2)
    print("Wrote", json_path)

    # preview of the first contact frame
    video = find_video(folder)
    if video and events and events[0]["x"] is not None:
        c = events[0]["contact_frame"]
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, c)
        ok, frame = cap.read()
        cap.release()
        if ok:
            fh, fw = frame.shape[:2]
            sx, sy = fw / W, fh / H
            x, y = int(events[0]["x"] * sx), int(events[0]["y"] * sy)
            cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.circle(frame, (x, y), 26, (0, 0, 255), 2)
            cv2.putText(frame, f"contact @ frame {c}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            out = os.path.join(folder, "contact_preview.png")
            cv2.imwrite(out, frame)
            print("Wrote", out, " <- check the anchor sits between the fingers")


if __name__ == "__main__":
    main()
