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
