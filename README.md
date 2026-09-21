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

## Trained gesture classifier

Press **Gestures** to load MediaPipe's `GestureRecognizer` — Google's
production-trained model, 8 MB, real confidences, full frame rate. It reports
seven categories: thumb up, thumb down, open palm, closed fist, victory,
pointing up, I-love-you. Hand skeletons are drawn on the feed.

It is dependable for one specific reason: **it is trained on exactly the
categories it reports.**

## Why no Kinetics model is used

MoViNet-A0 was integrated and then removed. Two independent problems:

**1. The label set is wrong for posture.** Kinetics-400 and Kinetics-600
contain no plain `walking`, `standing` or `sitting` class. The nearest entries
are `walking the dog`, `walking through snow`, `jaywalking`, `moon walking`,
`standing on hands`, `squat`. A Kinetics classifier asked "is this person
standing or walking" must answer from 600 unrelated activities, so it is
confidently wrong by construction. Swapping in a stronger Kinetics model
(TimeSformer, ViViT, VideoMAE) does not change this.

**2. The streaming build's state wiring is not recoverable.** It takes 47
inputs — image `[1,3,172,172]` NCHW plus 46 state tensors — and returns 28
outputs: a `[1,600]` head plus only 27 state tensors. The 16 cumulative-pool
states pair 1:1 by shape and order. The spatial buffers do not:

| shape | inputs | outputs |
|---|---|---|
| `[1,80,22,22]`  | 6  | 3 |
| `[1,184,11,11]` | 16 | 6 |
| `[1,112,11,11]` | 2  | 1 |
| `[1,384,6,6]`   | 4  | 1 |

16 against 6 is not an integer ratio, so blocks carry different buffer depths
and the wiring cannot be derived from shapes. Wrong feedback does not raise an
error, it yields confident nonsense.

Measured, for the record: loads in ~3 s, inference ~22 ms via
`@tensorflow/tfjs-tflite`, weights from `litert-community/MoViNet-A0-Stream-LiteRT`.
Note also that `tfhub.dev` is fully sunset — every URL now redirects to a
Kaggle search page, and Kaggle model downloads need auth and send no CORS.

## The right model for posture, and why it is not here

The correct taxonomy is **NTU RGB+D 60/120**: walking, sitting down, standing
up, falling down, staggering, picking up, throwing. The credible pretrained
weights are ST-GCN++ and PoseC3D in [MMAction2](https://github.com/open-mmlab/mmaction2),
as PyTorch checkpoints. They consume pose keypoints — which this page already
produces — so the integration is natural, but they must be exported to ONNX
first and run through `onnxruntime-web`. Searching Hugging Face for
ready-made ONNX skeleton-action models returns only zero-download hobby
uploads; nothing credible is browser-ready today.

Until that export exists, posture stays on the geometry heuristics, which are
built on MediaPipe Pose — itself a proven production model.

## Behavioural cues (loss prevention)

Press **Behaviour cues**. Six pose-derived signals are scored per person and
combined into a 0-100 figure, with the highest-scoring person shown in the rail
and flagged amber on the video above 45.

| Cue | Weight | Measured as |
|---|---|---|
| Hand at waist | 0.32 | wrist within 0.3–0.85 torso lengths of the hip midpoint, below the shoulder line, sustained |
| Hand out of sight | 0.22 | wrist visibility collapsing while both shoulders stay clearly visible |
| Repeated reaching | 0.15 | completed reach-and-retract cycles over 20 s |
| Lingering | 0.12 | seconds accumulated below the motion threshold |
| Looking around | 0.11 | range of nose offset from shoulder centre over 4 s |
| Turned away | 0.08 | shoulder width shrinking relative to torso length |

Weights sum to 1.0. All distances are in torso lengths, so the score does not
change with distance from the camera. Sustained contact accumulates and decays
at 0.8x, so a single frame cannot move the score.

### What this is not

It does not see objects. It cannot observe an item leaving a shelf, entering a
pocket or crossing a threshold, and it has no concept of payment. It infers
nothing about intent.

Every cue also describes ordinary shopping. Putting a phone away scores "hand
at waist". Crouching at a low shelf scores "lingering" and "hand out of sight".
Looking for a staff member scores "looking around". Expect a high false
positive rate — that is inherent to the method, not a tuning problem.

Treat the number as a prompt to look at a camera, never as evidence. Acting on
it directly would mean accusing people of theft on the basis of body posture,
which is both wrong and legally hazardous. Detaining someone on that basis
creates liability regardless of what the software displayed.

### Before deploying it anywhere

Filming identifiable people for behavioural analysis is regulated. In Morocco
that means Law 09-08 and a CNDP declaration; in the EU, GDPR, where scoring
individuals by behaviour invites Article 22 and DPIA obligations. Retail loss
prevention also has a long documented record of biased application, and a
system that directs staff attention will reproduce whatever bias is in how it
is used. Get legal advice specific to your jurisdiction before any live use.

A genuinely reliable concealment detector needs object tracking — seeing the
item itself — which means a trained detector for the merchandise and a model
of item-to-person transfer. That is a different and much larger project.
