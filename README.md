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
- Camera frames are sampled at a low rate and sent only to this loopback-bound container for face-presence estimates and common-object labels, then discarded. The browser temporarily holds one small prior frame in memory for coarse scene-motion detection and clears it when stopped. No images, video, timestamps, or event history are persisted.
- Face detection does not identify people. Object labels can be wrong. Motion is a change between consecutive frames, not a person's action or intent. Only the current result is displayed.
- The bundled MobileNet-SSD model recognizes a limited set of common Pascal VOC categories; it does not identify arbitrary objects. Visual analysis runs in the container and requires CPU/memory, unlike the passive network dashboard.
- Does not jam, interfere with, pair with, or control radios or other devices.
- Makes no outbound telemetry calls and stores no application data.

This dashboard observes the container's network namespace, not all devices on the host's LAN. Docker Desktop networking may expose only the container's virtual interface and a small set of cached neighbors. Host-level Wi-Fi/Bluetooth visibility would require a separately designed, permission-based integration outside this container; it is not simulated here.

## Security, privacy, and resource limits

Compose binds the web port to `127.0.0.1`, drops Linux capabilities, runs as a non-root user, enables a read-only root filesystem, and sets CPU, memory, process, and temporary-filesystem limits. No host devices, privileged mode, or host mounts are required. Python computer-vision dependencies and the pinned model are installed/downloaded only while building the Docker image, never on the host. Browser frames are posted to the same-origin local service; network frame requests reject cross-origin origins.

The model may mislabel faces, people, or objects. Do not use it to identify, track, make decisions about, or infer intent from people. Ask everyone present for informed consent before enabling the camera, and stop the camera when monitoring is no longer needed.

Third-party model attribution and license information is in [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).

## Tests

Run the standard-library tests in a dedicated container target without installing packages on the host:

```powershell
docker build --target test -t nmtorch:test .
docker run --rm nmtorch:test
```
