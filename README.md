# NMTorch Visual Monitor

NMTorch is a local, consent-based visual monitor. It detects supported objects and people, assigns temporary anonymous person IDs, estimates body poses and simple movement/posture labels, and shows occupancy and dwell-time estimates. It does not monitor networks or radios.

## Run with Docker Compose

Requirements: Docker Desktop with Docker Compose.

```powershell
docker compose up --build -d
docker compose ps
```

Open [http://localhost:8765](http://localhost:8765). The web port is bound to loopback and is not published to your LAN. To use another local port, set `NMTORCH_PORT` before starting Compose.

View logs with `docker compose logs -f`; stop the app with `docker compose down`.

## Visual features

- The camera starts only after the user checks the consent statement and separately grants browser camera permission. Audio is never requested.
- Frames are downscaled in the browser and sent to the same-origin local container for CPU inference. Frames are processed in memory and discarded; no image, video, or event history is written to disk.
- OpenCV MobileNet-SSD provides object and person boxes for its limited set of 20 Pascal VOC categories. LiteRT runs the MoveNet Lightning pose model on up to five detected people per frame.
- Person tracking IDs are anonymous, browser-memory labels. They reset when monitoring stops and do not recognize or re-identify a person.
- The overlay emphasizes a pose-estimated head region and draws a temporary head crop alongside the full-body skeleton. Arm and leg descriptions report landmark movement only; no facial-expression or emotion analysis is performed.
- The monitor estimates posture and walking/running from pose landmarks and short-term box movement. These are heuristics, not a trained general action-recognition model. They may be wrong, especially with occlusion, camera movement, or low frame rates.
- Occupancy, peak occupancy, approximate dwell time, and track appearance/disappearance describe only the current camera view. There are no calibrated zones or reliable congestion measurements.
- There is no face identity or emotion recognition, sign-language interpretation, fall detection, safety alarm, or medical function.
- Timestamped summaries are generated every five seconds, kept in memory only, limited to 12 items, and cleared when monitoring stops.

## Privacy and resource limits

Compose binds the web port to `127.0.0.1`, drops Linux capabilities, runs as a non-root user, enables a read-only root filesystem, and limits the container to 2 CPUs and 1 GiB of memory. No host devices, privileged mode, or host mounts are required. Python dependencies and pinned model assets are installed or fetched only during the Docker image build, never on the host. The running application makes no outbound telemetry calls; camera requests are same-origin only.

Ask everyone present for informed consent before enabling the camera. Stop monitoring when it is no longer needed. Object labels, pose estimates, and movement labels are not guaranteed to be accurate.

Third-party model and runtime attribution is in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Tests

Run the standard-library tests in a dedicated container target without installing packages on the host:

```powershell
docker build --target test -t nmtorch:test .
docker run --rm nmtorch:test
```
