# Design and validation notes

## Ownership

The UI thread captures frames and submits copied arrays with increasing sequences.
The inference worker owns YOLO, ByteTrack state and the CLIP cache. Its condition
variable protects one pending request and one completed result, so memory does not
grow with capture duration. A pending request is replaced by a newer one.

Published result packets are not mutated after publication. The UI applies each
packet once and owns all trajectory/counter updates. Rendering uses a copy of the
packet's frame, so a slow detector does not put old boxes on an unrelated new frame.
During camera startup and after resets, raw frames are displayed until a valid
result is ready.

Controls increment a settings generation. Results from an old generation are
discarded by the UI; the next processed generation resets the backend tracker and
classification cache. ROI changes are committed only on mouse release, avoiding
inference over a half-drawn or empty crop.

For file processing, headless mode calls the same inference method synchronously.
It avoids frame drops and permits deterministic synthetic regression checks.

## Counting and cache scope

Trails, direction and crossed IDs are retained for active tracked objects only.
An absent ID starts fresh when seen again. This bounds metadata by active detections;
the heatmap is fixed to the frame size. The counter is cumulative by label until reset.
A changed line/settings generation clears all trails and counts to avoid false
crossings based on old coordinates.

CLIP cache keys are nonnegative tracker IDs. Untracked detections have distinct
negative per-frame IDs and retain YOLO labels/confidences. CLIP is optional, batched,
cached, refreshed every 60 inference calls, and reset when disabled. A failed batch
logs a warning and clears its cache before using YOLO labels.

## Failures and shutdown

Exceptions from inference are surfaced to the main loop rather than silently killing
the worker. Shutdown drops pending work, wakes the condition, and joins for five
seconds. A native backend call cannot be safely interrupted in Python; a stuck call
produces an error after that timeout. The daemon worker does not block process exit.
Camera/writer/window cleanup remains in nested finally blocks.

Video writers are checked with isOpened. CLI output files are checked before work
starts and existing files cause an error. Recording uses nanosecond filenames.
The output path check is intended for a single local process, not concurrent hostile
writers.

## Checks

Unit regressions use fake backends and synchronization events, not camera timing.
The mailbox test holds the first inference busy, replaces the pending frame twice,
then verifies only the first and newest frames were processed. Other checks exercise
actual rendering and MP4 encode/decode, cache invalidation, ROI release over the panel,
stationary direction, gaps, reset, and resource cleanup.

The CPU smoke job uses upstream real weights on synthetic constant pixels. It checks
model import/loading, ByteTrack compatibility/reset, CLIP prompt/image preprocessing
and finite batch scores. An additional deterministic encoder checks that learned
logit scale is applied. There is no real-scene accuracy dataset or measured FPS result.

References: [Ultralytics tracker API](https://docs.ultralytics.com/reference/trackers/track/)
and [CLIP model forward](https://github.com/openai/CLIP/blob/main/clip/model.py).
