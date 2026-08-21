# Progress Report: Adding Visual Grounding to the Gripper Data Pipeline

**Date:** July 24, 2026
**Scope:** Extending the AnySense → SAM2 → Rerun pipeline with text-prompted object grounding

---

## Objective

Add a visual grounding step to the data collection pipeline so that a target object can be
located from a natural-language description ("point to the pill bottle") rather than
hand-annotated. This mirrors the *contact prompting* step in Contact-Anchored Policies
(CAP, Cui et al.), where a pixel coordinate on the object of interest is selected either
manually or by querying a vision-language model.

---

## Outcome

Working end-to-end. Text prompt → per-frame tracked object position, visualized in Rerun
alongside the existing RGB, depth, pose, segmentation, and gripper-state streams.
The originally selected model did not work; a substantially smaller alternative did.

---

## Attempt 1: MolmoPoint-8B (Ai2) — abandoned

MolmoPoint was the first choice: released March 2026, it replaces text-coordinate pointing
with a coarse-to-fine architecture using dedicated grounding tokens, which promises better
generalization across resolutions and lower decoding cost.

Four distinct failures, in order:

**1. Download and storage cost.** The checkpoint ships in F32 across 8 shards — roughly
34 GB, about 18 minutes of download.

**2. Out-of-memory via silent disk offload.** Loading with `dtype="auto"` honored the F32
checkpoint (9B params × 4 bytes ≈ 36 GB). Accelerate's `device_map="auto"` responded by
offloading layers to disk, producing "meta" tensors with no backing data. Failure surfaced
as `NotImplementedError: Cannot copy out of meta tensor`.
*Resolved* by loading in half precision and removing `device_map="auto"` — the model then
loaded successfully in ~38 seconds.

**3. Metal kernel assertion on Apple GPU.** With the model loaded, inference aborted inside
Apple's MPS backend: `Destination NDArray and Accumulator NDArray cannot have different
datatype in MPSNDArrayMatrixMultiplication`. Retrying in float16 produced the identical
assertion, indicating the dtype flag was not the operative variable.

**4. Root cause identified on CPU.** Running on CPU converted the opaque Metal assertion
into a precise error: `mat1 and mat2 must have the same dtype, but got Float and BFloat16`,
raised inside the model's image-pooling connector. The vision tower emits Float32
activations while the loaded weights are half precision, and the model's custom modeling
code performs no cast to reconcile them.

**Conclusion:** MolmoPoint's released code does not support half precision outside CUDA.
Ai2's own usage example wraps inference in `torch.autocast("cuda", ...)`, which masks the
mismatch automatically. There is no autocast equivalent on Apple MPS. Running at full F32
precision would require ~36 GB of memory. The model is effectively CUDA-only in practice.

---

## Attempt 2: Moondream 2 — adopted

Moondream 2 was selected as the replacement. Two factors made it defensible rather than a
fallback:

- **CAP itself benchmarks three different pointers** — Gemini Robotics-ER 1.5, Moondream,
  and Molmo — so the pointing model is an interchangeable component of the method, not a
  fixed dependency.
- **Pointing is a native capability**, not a prompt convention: the model returns (x, y)
  coordinates for every instance of a described object as a first-class output.

| | MolmoPoint-8B | Moondream 2 |
|---|---|---|
| Parameters | ~9B | ~2B |
| Download | ~34 GB / 18 min | 3.85 GB / 2 min |
| Memory (half precision) | ~18 GB | ~4 GB |
| Ran on available hardware | No | Yes |

One integration issue arose: `RuntimeError: Passed CPU tensor to MPS op`. Moondream detects
Apple GPU availability internally and migrates some tensors there, conflicting with
CPU-loaded weights. Resolved by running the model explicitly on MPS. A CPU-only path was
also added, which suppresses the internal GPU detection, as a fallback.

**Result:** first successful grounding returned a point at (252, 344) in a 720×960 frame.

---

## Follow-on issue: a single point is only valid for one frame

Initial integration logged the model's prediction as a fixed marker. This was incorrect.
The phone is rigidly mounted to the gripper, so during a trial **the camera moves while the
object remains stationary** — the object's pixel position shifts continuously even though
it has not moved in the world. A frame-0 coordinate is therefore correct only at frame 0.

CAP addresses this by deprojecting the pixel into a 3D contact anchor and updating it each
frame from camera odometry.

**Our approach: seed SAM2 with the grounding model's output.** The predicted point is
supplied to SAM2's video predictor as a positive click prompt on frame 0; SAM2 propagates
the object mask across the video, and the mask centroid gives the object's position on
every frame.

This has three advantages:
- It reuses SAM2, already integrated for gripper-finger segmentation.
- The grounding model runs **once per trial**, not once per frame — keeping the expensive
  component to a single forward pass.
- It requires no depth data or camera intrinsics, which we do not currently export.

Multi-object support falls out of the same design: each predicted point is assigned its own
SAM2 object ID and all targets are tracked in a single propagation pass, so tracking three
objects costs approximately the same as tracking one.

---

## Also completed

**Hindsight contact labeling** (CAP §3.1.4), implemented independently of any model. The
contact frame is detected as the moment the gripper aperture ceases to decrease — the
fingers have closed and halted against the object — and the contact anchor is the midpoint
between the two finger-mask centroids at that instant. Runs in seconds on existing outputs
with no model, no GPU, and no per-trial configuration. On the validation trial it detected
contact events at frames 857, 1116, and 1274.

This is worth distinguishing from the grounding work: **hindsight labeling is ground truth**
(where the gripper actually made contact), whereas the grounding model produces a
**prediction**. Both are now displayed in the same view, which allows the model's accuracy
to be assessed directly against observed behavior.

---

## Current pipeline

```
pick_points.py              mark gripper fingers on frame 0            (one-time, per trial)
extract_gripper_sam2.py     SAM2 finger masks + aperture signal        (~30 min CPU)
extract_contact_anchor.py   hindsight contact anchors                  (seconds)
ground_targets_moondream.py text prompt → object point(s) on frame 0   (seconds)
extract_target_track.py     SAM2 tracking of grounded targets          (~30 min CPU)
build_gripper_rrd.py        assemble the synchronized .rrd             (seconds)
```

The Rerun recording now carries RGB video, depth video, 6-DoF pose and trajectory,
gripper-finger segmentation, gripper aperture and open/closed state, hindsight contact
anchors, per-frame grasp center, and per-frame tracked target objects — all on one
scrubbable timeline.

---

## Takeaways

1. **A purpose-built 2B model outperformed a general-purpose 9B model** for this task —
   not on accuracy, but on the dimension that mattered: it ran at all on available hardware.
   Nine gigabytes of extra parameters bought nothing here.

2. **Model size is not the binding constraint; supported precision is.** The blocker was not
   raw memory but the absence of a working half-precision path outside CUDA. The
   distinction only became visible after moving to CPU, where the error message was
   specific instead of a GPU-level assertion.

3. **Run expensive models sparsely and track cheaply.** Pointing once and propagating with
   SAM2 keeps the VLM to one forward pass per trial. This pattern should generalize to any
   future model in this slot.

4. **The pointing model is a swappable component.** Because CAP treats it that way and our
   scripts communicate through plain JSON, replacing Moondream later — with a CUDA machine
   and MolmoPoint, or with an API model — requires no changes downstream.

---

## Known gaps and next steps

- **Metric depth and camera intrinsics are not currently exported.** Both are required to
  convert 2D points into CAP's true 3D contact anchors, which would also make anchors
  visible in the 3D panel rather than only the video panel. AnySense exposes intrinsics via
  its streaming API and saves metric depth separately from the depth visualization video.
- **Contact anchors are currently drawn statically**, which has the same camera-motion
  problem that was corrected for target tracking. Each anchor should be shown at its own
  contact frame or tracked forward.
- **CPU runtime is the practical bottleneck** — roughly 30 minutes per SAM2 pass on a
  ~1,275-frame trial. Moving the SAM2 and grounding stages to a CUDA machine would reduce
  this to a few minutes and would also make MolmoPoint viable if we wanted to revisit it.
- **Batch processing across trials** is not yet automated; each trial is currently run
  manually.
