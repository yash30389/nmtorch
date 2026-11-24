# NMTorch

NMTorch is a lightweight network observability dashboard with an optional, consent-based camera preview. It runs in an isolated Docker container and reports only network interfaces and cached ARP neighbors visible to that container.

## Run with Docker Compose

Requirements: Docker Desktop with Docker Compose.

```powershell
docker compose up --build -d
docker compose ps
```

Open [http://localhost:8765](http://localhost:8765). To use another local port, set `NMTORCH_PORT` before starting Compose. The dashboard is bound to loopback and is not published to your LAN.

View logs with `docker compose logs -f`; stop the app with `docker compose down`.

## Features and scope

- Shows container-visible network interfaces, link state, IPv4 address, MAC address, and receive/transmit counters.
- Displays entries already present in the container's neighbor cache. It does not probe addresses, scan the LAN, or connect to discovered devices.
- Refreshes every four seconds; traffic rates are calculated in the browser from consecutive counter snapshots.
- Wi-Fi access-point discovery and Bluetooth discovery are explicitly unavailable. Docker Desktop does not pass host radio hardware or its scan results into this container.
- The optional visual monitor starts only after you affirm that this is your space and people present are informed and consenting. The browser separately requests camera permission; audio is never requested.
- Camera frames are sampled at a low rate and sent only to this loopback-bound container for object labels, then discarded. The browser temporarily holds the prior detected boxes and one small frame in memory to estimate motion and clears them when stopped. Timestamped summaries are produced every five seconds and kept only in the browser's memory; no images, video, or event history are persisted.
- MobileNet-SSD is filtered by confidence and per-class non-maximum suppression to reduce weak and duplicate boxes. Object labels can still be wrong. The timeline describes changes in detected box position only; it does not classify actions or intent, and camera movement can look like object movement.
- The bundled MobileNet-SSD model recognizes only 20 common Pascal VOC categories; unsupported objects are not labeled. Visual analysis runs in the container and requires CPU/memory, unlike the passive network dashboard.
- Does not jam, interfere with, pair with, or control radios or other devices.
- Makes no outbound telemetry calls and stores no application data.

This dashboard observes the container's network namespace, not all devices on the host's LAN. Docker Desktop networking may expose only the container's virtual interface and a small set of cached neighbors. Host-level Wi-Fi/Bluetooth visibility would require a separately designed, permission-based integration outside this container; it is not simulated here.

## Security, privacy, and resource limits

Compose binds the web port to `127.0.0.1`, drops Linux capabilities, runs as a non-root user, enables a read-only root filesystem, and sets CPU, memory, process, and temporary-filesystem limits. No host devices, privileged mode, or host mounts are required. Python computer-vision dependencies and the pinned model are installed/downloaded only while building the Docker image, never on the host. Browser frames are posted to the same-origin local service; network frame requests reject cross-origin origins.

The model may mislabel objects and movement. It does not identify people or classify their actions or emotions. Ask everyone present for informed consent before enabling the camera, and stop the camera when monitoring is no longer needed.

Third-party model attribution and license information is in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Tests

Run the standard-library tests in a dedicated container target without installing packages on the host:

```powershell
docker build --target test -t nmtorch:test .
docker run --rm nmtorch:test
```
