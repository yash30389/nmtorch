"use strict";

const elements = {
  refresh: document.querySelector("#refresh-button"),
  lastUpdated: document.querySelector("#last-updated"),
  interfaceCount: document.querySelector("#interface-count"),
  interfaceSummary: document.querySelector("#interface-summary"),
  interfaceList: document.querySelector("#interface-list"),
  neighborCount: document.querySelector("#neighbor-count"),
  neighborList: document.querySelector("#neighbor-list"),
  toast: document.querySelector("#toast"),
  cameraConsent: document.querySelector("#camera-consent"),
  cameraStatus: document.querySelector("#camera-status"),
  cameraVideo: document.querySelector("#camera-video"),
  cameraOverlay: document.querySelector("#camera-overlay"),
  cameraPlaceholder: document.querySelector("#camera-placeholder"),
  cameraLive: document.querySelector("#camera-live"),
  cameraMessage: document.querySelector("#camera-message"),
  startCamera: document.querySelector("#start-camera"),
  stopCamera: document.querySelector("#stop-camera"),
  sceneResults: document.querySelector("#scene-results"),
  sceneTimeline: document.querySelector("#scene-timeline"),
};

let refreshTimer;
let toastTimer;
let timelineTimer;
let previousCounters = new Map();
let cameraStream;
let cameraActive = false;
let cameraRun = 0;
let previousObjectDetections = [];
let sceneWindowSamples = 0;
let sceneWindowObjects = new Map();
const captureCanvas = document.createElement("canvas");
captureCanvas.width = 320;
captureCanvas.height = 240;
const captureContext = captureCanvas.getContext("2d", { alpha: false });
const overlayContext = elements.cameraOverlay.getContext("2d");
const motionCanvas = document.createElement("canvas");
motionCanvas.width = 160;
motionCanvas.height = 120;
const motionContext = motionCanvas.getContext("2d", { willReadFrequently: true });
let previousMotionFrame;
const SCENE_SUMMARY_INTERVAL_MS = 5000;
const MAX_TIMELINE_ITEMS = 12;
const OBJECT_MOVEMENT_THRESHOLD = 0.035;
const MAX_OBJECT_MATCH_DISTANCE = 0.25;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unitIndex]}`;
}

function createTextElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function renderInterfaces(interfaces) {
  elements.interfaceCount.textContent = String(interfaces.length);
  const active = interfaces.filter((item) => item.state === "up").length;
  elements.interfaceSummary.textContent = `${active} link${active === 1 ? "" : "s"} up`;
  elements.interfaceList.replaceChildren();

  if (interfaces.length === 0) {
    elements.interfaceList.append(createTextElement("div", "empty-state", "No network interfaces were exposed to the container."));
    return;
  }

  for (const item of interfaces) {
    const card = createTextElement("article", "interface-card", "");
    const name = createTextElement("div", "interface-name", "");
    name.append(createTextElement("span", "interface-symbol", item.type === "wireless" ? "⌁" : "↗"));
    const nameText = createTextElement("div", "", "");
    nameText.append(createTextElement("strong", "", item.name));
    nameText.append(createTextElement("small", "", item.type));
    name.append(nameText);
    card.append(name);

    const fields = [
      ["IPv4 ADDRESS", item.ipv4 || "Not assigned"],
      ["RECEIVED", `${formatBytes(item.received_bytes)} · ${rate(item, "received_bytes")}`],
      ["TRANSMITTED", `${formatBytes(item.transmitted_bytes)} · ${rate(item, "transmitted_bytes")}`],
    ];
    for (const [label, value] of fields) {
      const field = createTextElement("div", "interface-field", "");
      field.append(createTextElement("span", "", label));
      field.append(createTextElement("strong", "", value));
      card.append(field);
    }

    const state = createTextElement("span", `status-chip${item.state === "up" ? "" : " down"}`, item.state);
    card.append(state);
    elements.interfaceList.append(card);
  }
}

function rate(item, counter) {
  const key = `${item.name}:${counter}`;
  const value = item[counter];
  const previous = previousCounters.get(key);
  previousCounters.set(key, { value, at: performance.now() });
  if (!previous || value < previous.value) return "—/s";
  const elapsed = Math.max((performance.now() - previous.at) / 1000, 0.001);
  return `${formatBytes(Math.round((value - previous.value) / elapsed))}/s`;
}

function renderNeighbors(neighbors) {
  elements.neighborCount.textContent = String(neighbors.length);
  elements.neighborList.replaceChildren();
  if (neighbors.length === 0) {
    elements.neighborList.append(createTextElement("div", "empty-state", "No cached neighbors are currently visible to this container."));
    return;
  }

  const table = createTextElement("table", "neighbor-table", "");
  const header = document.createElement("thead");
  const headerRow = document.createElement("tr");
  for (const title of ["IP ADDRESS", "LINK ADDRESS", "STATE", "INTERFACE"]) {
    headerRow.append(createTextElement("th", "", title));
  }
  header.append(headerRow);
  table.append(header);

  const body = document.createElement("tbody");
  for (const neighbor of neighbors) {
    const row = document.createElement("tr");
    row.append(createTextElement("td", "mono", neighbor.address));
    row.append(createTextElement("td", "mono", neighbor.mac));
    row.append(createTextElement("td", "", neighbor.state));
    row.append(createTextElement("td", "", neighbor.interface));
    body.append(row);
  }
  table.append(body);
  elements.neighborList.append(table);
}

function renderScene(result) {
  const rows = [
    ["Visible objects", result.objects.map((item) => `${item.label} × ${item.count}`).join(", ") || "None recognized"],
    ["Object movement", summarizeCurrentMovement(result.movementStates)],
    ["Scene motion", result.motion_detected ? `Detected (${result.motion_percent}%)` : "Not detected"],
  ];
  elements.sceneResults.replaceChildren();
  for (const [label, value] of rows) {
    const row = createTextElement("div", "scene-row", "");
    row.append(createTextElement("span", "", label));
    row.append(createTextElement("strong", "", value));
    elements.sceneResults.append(row);
  }
  drawDetections(result.detections);
}

function compareObjectPositions(detections) {
  const previous = previousObjectDetections;
  const matchedPrevious = new Set();
  const states = detections.map((detection) => {
    const centerX = detection.x + detection.width / 2;
    const centerY = detection.y + detection.height / 2;
    let bestIndex = -1;
    let bestDistance = Infinity;

    previous.forEach((candidate, index) => {
      if (matchedPrevious.has(index) || candidate.label !== detection.label) return;
      const previousX = candidate.x + candidate.width / 2;
      const previousY = candidate.y + candidate.height / 2;
      const distance = Math.hypot(centerX - previousX, centerY - previousY);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });

    if (bestIndex < 0 || bestDistance > MAX_OBJECT_MATCH_DISTANCE) {
      return { ...detection, movement: "new" };
    }
    matchedPrevious.add(bestIndex);
    return {
      ...detection,
      movement: bestDistance >= OBJECT_MOVEMENT_THRESHOLD ? "changed" : "stable",
    };
  });

  previousObjectDetections = detections.map((detection) => ({ ...detection }));
  return states;
}

function summarizeCurrentMovement(movementStates) {
  if (movementStates.length === 0) return "No supported objects";
  const descriptions = new Map();
  for (const detection of movementStates) {
    if (!descriptions.has(detection.label)) {
      descriptions.set(detection.label, detection.movement);
    } else if (detection.movement === "changed") {
      descriptions.set(detection.label, "changed");
    }
  }
  return [...descriptions.entries()]
    .map(([label, movement]) => `${label}: ${movement === "changed" ? "position changed" : movement === "stable" ? "position stable" : "just appeared"}`)
    .join("; ");
}

function recordSceneSample(movementStates) {
  sceneWindowSamples += 1;
  const labels = new Map();
  for (const detection of movementStates) {
    const stats = labels.get(detection.label) || { count: 0, changed: false, appeared: false };
    stats.count += 1;
    if (detection.movement === "changed") stats.changed = true;
    if (detection.movement === "new") stats.appeared = true;
    labels.set(detection.label, stats);
  }

  for (const [label, current] of labels) {
    const stats = sceneWindowObjects.get(label) || {
      observedSamples: 0,
      changedSamples: 0,
      appearedSamples: 0,
      maxCount: 0,
    };
    stats.observedSamples += 1;
    stats.changedSamples += Number(current.changed);
    stats.appearedSamples += Number(current.appeared);
    stats.maxCount = Math.max(stats.maxCount, current.count);
    sceneWindowObjects.set(label, stats);
  }
}

function appendSceneSummary() {
  if (!cameraActive) return;
  const timestamp = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  let description = "No supported object labels recognized.";
  if (sceneWindowObjects.size > 0) {
    description = [...sceneWindowObjects.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([label, stats]) => {
        const observation = stats.changedSamples > 0
          ? `position changed in ${stats.changedSamples} of ${stats.observedSamples} samples`
          : stats.appearedSamples > 0
            ? "appeared in view; no repeated movement confirmed"
            : "no clear position change";
        return `${label} × ${stats.maxCount}: ${observation}`;
      })
      .join(" · ");
  } else if (sceneWindowSamples === 0) {
    description = "No camera frame was analyzed in this interval.";
  }

  const item = createTextElement("article", "timeline-item", "");
  item.append(createTextElement("time", "timeline-time", timestamp));
  item.append(createTextElement("span", "timeline-description", description));
  if (elements.sceneTimeline.querySelector(".empty-state")) {
    elements.sceneTimeline.replaceChildren();
  }
  elements.sceneTimeline.prepend(item);
  while (elements.sceneTimeline.children.length > MAX_TIMELINE_ITEMS) {
    elements.sceneTimeline.lastElementChild.remove();
  }
  sceneWindowSamples = 0;
  sceneWindowObjects = new Map();
}

function measureMotion() {
  motionContext.drawImage(captureCanvas, 0, 0, motionCanvas.width, motionCanvas.height);
  const current = motionContext.getImageData(0, 0, motionCanvas.width, motionCanvas.height).data;
  let changedPixels = 0;
  if (previousMotionFrame) {
    for (let index = 0; index < current.length; index += 4) {
      const difference = Math.abs(current[index] - previousMotionFrame[index])
        + Math.abs(current[index + 1] - previousMotionFrame[index + 1])
        + Math.abs(current[index + 2] - previousMotionFrame[index + 2]);
      if (difference > 90) changedPixels += 1;
    }
  }
  previousMotionFrame = new Uint8ClampedArray(current);
  const motionPercent = changedPixels / (motionCanvas.width * motionCanvas.height) * 100;
  return { motion_detected: motionPercent >= 3.5, motion_percent: Math.round(motionPercent * 10) / 10 };
}

function drawDetections(detections) {
  const canvas = elements.cameraOverlay;
  overlayContext.clearRect(0, 0, canvas.width, canvas.height);
  overlayContext.lineWidth = 2;
  overlayContext.font = "12px system-ui, sans-serif";
  for (const detection of detections) {
    const x = detection.x * canvas.width;
    const y = detection.y * canvas.height;
    const width = detection.width * canvas.width;
    const height = detection.height * canvas.height;
    overlayContext.strokeStyle = "#64d5a0";
    overlayContext.strokeRect(x, y, width, height);
    overlayContext.fillStyle = "#101718e8";
    const labelWidth = overlayContext.measureText(detection.label).width + 10;
    overlayContext.fillRect(x, Math.max(0, y - 19), labelWidth, 19);
    overlayContext.fillStyle = "#e6ece9";
    overlayContext.fillText(detection.label, x + 5, Math.max(13, y - 5));
  }
}

function stopCamera(message) {
  cameraRun += 1;
  cameraActive = false;
  window.clearInterval(timelineTimer);
  timelineTimer = undefined;
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = undefined;
  }
  elements.cameraVideo.pause();
  elements.cameraVideo.srcObject = null;
  elements.cameraOverlay.getContext("2d").clearRect(0, 0, 320, 240);
  previousMotionFrame = undefined;
  previousObjectDetections = [];
  sceneWindowSamples = 0;
  sceneWindowObjects = new Map();
  elements.sceneTimeline.replaceChildren(
    createTextElement("div", "empty-state", "Timestamped movement summaries appear every five seconds."),
  );
  elements.cameraPlaceholder.hidden = false;
  elements.cameraLive.classList.remove("visible");
  elements.sceneResults.replaceChildren(
    createTextElement("div", "empty-state", "Camera is off. No video or event history is retained."),
  );
  elements.cameraStatus.textContent = "CAMERA OFF";
  elements.startCamera.hidden = false;
  elements.stopCamera.hidden = true;
  elements.cameraConsent.disabled = false;
  if (message) elements.cameraMessage.textContent = message;
}

async function captureFrames(run) {
  while (cameraActive && run === cameraRun) {
    try {
      if (elements.cameraVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        captureContext.drawImage(elements.cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        const frame = await new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.65));
        if (!frame) throw new Error("Could not encode a camera frame.");
        const response = await fetch("/api/camera/analyze", {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: frame,
          cache: "no-store",
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || `Analysis failed (${response.status}).`);
        if (run !== cameraRun) return;
        const movementStates = compareObjectPositions(result.detections);
        recordSceneSample(movementStates);
        renderScene({ ...result, movementStates, ...measureMotion() });
        elements.cameraMessage.textContent = "Live object estimates. Five-second summaries stay in memory and are cleared when stopped.";
      }
      await new Promise((resolve) => window.setTimeout(resolve, 750));
    } catch (error) {
      if (run !== cameraRun) return;
      const message = error instanceof Error ? error.message : "Camera analysis failed.";
      stopCamera(`Camera stopped: ${message}`);
      return;
    }
  }
}

async function startCamera() {
  if (!elements.cameraConsent.checked) {
    elements.cameraMessage.textContent = "Confirm the consent statement before starting the camera.";
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    elements.cameraMessage.textContent = "This browser does not support camera access on this page.";
    return;
  }

  elements.startCamera.disabled = true;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
        frameRate: { ideal: 3, max: 5 },
      },
    });
    elements.cameraVideo.srcObject = cameraStream;
    await elements.cameraVideo.play();
    cameraActive = true;
    const run = ++cameraRun;
    elements.cameraPlaceholder.hidden = true;
    elements.cameraLive.classList.add("visible");
    elements.cameraStatus.textContent = "CAMERA ON";
    elements.startCamera.hidden = true;
    elements.stopCamera.hidden = false;
    elements.cameraConsent.disabled = true;
    elements.cameraMessage.textContent = "Camera is active. No recording is made; one small prior frame is held in browser memory for motion comparison.";
    elements.sceneTimeline.replaceChildren(
      createTextElement("div", "empty-state", "Waiting for the first five-second summary."),
    );
    timelineTimer = window.setInterval(appendSceneSummary, SCENE_SUMMARY_INTERVAL_MS);
    captureFrames(run);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Camera permission was not granted.";
    stopCamera(`Could not start camera: ${message}`);
  } finally {
    elements.startCamera.disabled = false;
  }
}

function showError(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("visible"), 5000);
}

async function refresh() {
  elements.refresh.disabled = true;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`Status request failed (${response.status}).`);
    const status = await response.json();
    renderInterfaces(status.interfaces);
    renderNeighbors(status.neighbors);
    elements.lastUpdated.textContent = new Date().toLocaleTimeString();
    elements.toast.classList.remove("visible");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Could not read network status.";
    elements.lastUpdated.textContent = "Update failed";
    showError(message);
  } finally {
    elements.refresh.disabled = false;
    window.clearTimeout(refreshTimer);
    refreshTimer = window.setTimeout(refresh, 4000);
  }
}

elements.refresh.addEventListener("click", refresh);
elements.startCamera.addEventListener("click", startCamera);
elements.stopCamera.addEventListener("click", () => stopCamera("Camera stopped. No video or event history was saved."));
elements.cameraConsent.addEventListener("change", () => {
  if (!elements.cameraConsent.checked && cameraActive) stopCamera("Consent withdrawn. Camera stopped.");
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && cameraActive) stopCamera("Camera stopped because this tab is no longer visible.");
});
window.addEventListener("pagehide", () => stopCamera(""));
refresh();
