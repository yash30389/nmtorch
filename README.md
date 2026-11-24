# NMTorch

NMTorch is a small, read-only network observability dashboard. It runs in an isolated Docker container and reports only network interfaces and cached ARP neighbors visible to that container.

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
- Does not jam, interfere with, pair with, or control radios or other devices.
- Makes no outbound telemetry calls and stores no application data.

This dashboard observes the container's network namespace, not all devices on the host's LAN. Docker Desktop networking may expose only the container's virtual interface and a small set of cached neighbors. Host-level Wi-Fi/Bluetooth visibility would require a separately designed, permission-based integration outside this container; it is not simulated here.

## Security and resource limits

Compose binds the web port to `127.0.0.1`, drops Linux capabilities, runs as a non-root user, enables a read-only root filesystem, and sets CPU, memory, process, and temporary-filesystem limits. No host devices, privileged mode, host mounts, or extra packages are required at runtime.

## Tests

Run the standard-library tests in a dedicated container target without installing packages on the host:

```powershell
docker build --target test -t nmtorch:test .
docker run --rm nmtorch:test
```
