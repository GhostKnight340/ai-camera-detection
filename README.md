# AI Camera Detection

Two browser prototypes that analyse a live camera feed entirely on-device.
No video ever leaves the browser.

Live: https://ghostknight340.github.io/ai-camera-detection/

## Pages

| File | What it detects | Dependencies |
|---|---|---|
| `index.html` | People + their actions, from 33 pose landmarks each | ~6 MB model from a CDN, cached after first load |
| `motion.html` | Movement anywhere in frame | none |

## Actions

`index.html` runs MediaPipe PoseLandmarker and derives actions from joint
geometry. Each label carries a continuous confidence computed from measured
angles and distances, not a trained classifier:

| Action | Derived from |
|---|---|
| Standing / Walking | torso upright, knees extended, hip-centre velocity |
| Sitting / Crouching | knee flexion, plus hip height relative to knees |
| Bending over | torso tilt away from vertical |
| Lying down | torso near-horizontal with legs extended |
| Arms raised | wrist height above shoulder, in torso lengths |
| Reaching out | elbow extension combined with wrist offset from the midline |
| Hand to head | wrist-to-nose distance, scaled by torso length |

Labels are held for 400 ms before committing so they do not flicker, and
each tracked person keeps a stable id by nearest-centroid matching.

## What this cannot do

Semantic events like "item placed in backpack" need a classifier trained on
a labelled dataset of that specific event. Pose geometry alone cannot infer
intent or identify objects. Anything of that kind requires collecting and
annotating footage, then training a temporal action model.

Also out of scope here: re-identifying a person across cameras, and any
form of identity recognition.

## Running locally

The camera API requires `https://` or `localhost`:

    python -m http.server 8000

Then open http://localhost:8000/

## Limits

- Actions need hips, knees and shoulders visible; a head-and-shoulders
  framing will not classify posture reliably.
- `motion.html` detects movement only — anything that moves triggers it.
- Prototypes: single camera, no persistence, not tested over long runs.

## MoViNet (optional second detector)

`index.html` can also run **MoViNet-A0**, a real trained Kinetics-600 video
classifier, alongside the geometry heuristics. It loads on demand — the
16 MB model is not downloaded unless you press the button.

- weights: `litert-community/MoViNet-A0-Stream-LiteRT` (Hugging Face, CORS-enabled)
- runtime: `@tensorflow/tfjs-tflite` (WASM), labels in `kinetics600.txt`
- measured inference: ~22 ms per frame, driven at ~10 Hz

### Why its temporal state is held at zero

`probe.html` dumps the model signature. It takes **47 inputs** — the image
`[1,3,172,172]` in NCHW, plus 46 state tensors — and returns **28 outputs**:
a `[1,600]` head plus only 27 state tensors.

The 16 cumulative-pool states (`[1,C,1,1]`) pair 1:1 with their outputs by
shape and order. The spatial buffers do not:

| shape | inputs | outputs |
|---|---|---|
| `[1,80,22,22]`  | 6  | 3 |
| `[1,184,11,11]` | 16 | 6 |
| `[1,112,11,11]` | 2  | 1 |
| `[1,384,6,6]`   | 4  | 1 |

16 against 6 is not an integer ratio, so blocks carry different buffer depths
and the correct wiring cannot be recovered from shapes alone. Feeding state
back wrongly does not raise an error — it yields confident nonsense — so state
is held at zero, which is exactly the condition the model sees on a stream's
first frame. Temporal stability comes from averaging predictions over a
window instead.

To get true streaming behaviour you need the original model definition to
recover the buffer layout, then feed each output back to its matching input.

### What to expect

Kinetics-600 is everyday activity — "playing guitar", "eating cake". Without
temporal state the classifier leans on scene appearance, so it is weakest at
exactly the distinctions the pose heuristics handle well (standing vs walking).
Treat it as a comparison baseline, not a replacement.
