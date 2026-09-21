# AI Camera Detection

Two browser prototypes for detecting people in a camera frame, running
entirely on-device. No video ever leaves the browser.

Live: https://ghostknight340.github.io/ai-camera-detection/

## Pages

| File | Detection | Dependencies |
|---|---|---|
| `index.html` | MediaPipe EfficientDet-Lite0, `person` class | ~7 MB model from a CDN, cached after first load |
| `motion.html` | Background subtraction (classical CV) | none |

## Shared logic

Both pages share the part that actually matters:

- **Detection zone**, draggable and resizable, in normalised coordinates
- **2-second debounce** — state must hold before it flips, so it does not
  flicker on brief movement
- **Counters** — time in current state, cumulative clear time, event count
- `motion.html` adds a 3-minute occupancy timeline

Only the detector underneath would change in a real deployment; the zone
and state logic stays as-is.

## Running locally

The camera API requires `https://` or `localhost`:

    python -m http.server 8000

Then open http://localhost:8000/

## Known limits

- `motion.html` detects **movement**, not people. Anything that moves
  through the zone triggers it.
- `index.html` detects people properly, but only presence — it does not
  identify or track anyone across frames.
- Prototypes only: single camera, no persistence, not tested for long runs.
