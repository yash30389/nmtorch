"use strict";

const elements = {
  cameraConsent: document.querySelector("#camera-consent"),
  cameraStatus: document.querySelector("#camera-status"),
  cameraVideo: document.querySelector("#camera-video"),
  cameraOverlay: document.querySelector("#camera-overlay"),
  cameraPlaceholder: document.querySelector("#camera-placeholder"),
  cameraLive: document.querySelector("#camera-live"),
  faceFocusPanel: document.querySelector("#face-focus-panel"),
  faceFocus: document.querySelector("#face-focus"),
  cameraMessage: document.querySelector("#camera-message"),
  startCamera: document.querySelector("#start-camera"),
  stopCamera: document.querySelector("#stop-camera"),
  peopleCount: document.querySelector("#people-count"),
  occupancyPeak: document.querySelector("#occupancy-peak"),
  averageDwell: document.querySelector("#average-dwell"),
  objectCount: document.querySelector("#object-count"),
  sceneResults: document.querySelector("#scene-results"),
  sceneTimeline: document.querySelector("#scene-timeline"),
};

let timelineTimer;
let cameraStream;
let cameraActive = false;
let cameraRun = 0;
let trackFrame = 0;
let nextPersonId = 1;
let maxOccupancy = 0;
let peopleTracks = new Map();
let previousObjectDetections = [];
let sceneWindowSamples = 0;
let sceneWindowPeak = 0;
let sceneWindowEntries = 0;
let sceneWindowExits = 0;
let sceneWindowMovementSamples = 0;
let sceneWindowPeople = new Map();
let sceneWindowObjects = new Map();
const captureCanvas = document.createElement("canvas");
captureCanvas.width = 320;
captureCanvas.height = 240;
const captureContext = captureCanvas.getContext("2d", { alpha: false });
const overlayContext = elements.cameraOverlay.getContext("2d");
const faceFocusContext = elements.faceFocus.getContext("2d", { alpha: false });
const motionCanvas = document.createElement("canvas");
motionCanvas.width = 160;
motionCanvas.height = 120;
const motionContext = motionCanvas.getContext("2d", { willReadFrequently: true });
let previousMotionFrame;

const SCENE_SUMMARY_INTERVAL_MS = 5000;
const MAX_TIMELINE_ITEMS = 12;
const MAX_TRACK_MISSED_FRAMES = 4;
const PERSON_MATCH_DISTANCE = 0.24;
const MOVEMENT_THRESHOLD = 0.03;
const POSE_SCORE_THRESHOLD = 0.3;
const SKELETON_LINKS = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
];

function createTextElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function pointMap(pose) {
  return new Map((pose || []).filter((point) => point.score >= POSE_SCORE_THRESHOLD)
    .map((point) => [point.name, point]));
}

function jointAngle(first, middle, last) {
  const firstVector = [first.x - middle.x, first.y - middle.y];
  const lastVector = [last.x - middle.x, last.y - middle.y];
  const divisor = Math.hypot(...firstVector) * Math.hypot(...lastVector);
  if (divisor === 0) return null;
  const cosine = Math.max(-1, Math.min(1,
    (firstVector[0] * lastVector[0] + firstVector[1] * lastVector[1]) / divisor));
  return Math.acos(cosine) * 180 / Math.PI;
}

function estimateActivity(detection, track, now) {
  const points = pointMap(detection.pose);
  const shoulderNames = ["left_shoulder", "right_shoulder"];
  const hipNames = ["left_hip", "right_hip"];
  const shoulderPoints = shoulderNames.map((name) => points.get(name)).filter(Boolean);
  const hipPoints = hipNames.map((name) => points.get(name)).filter(Boolean);
  if (shoulderPoints.length === 2 && hipPoints.length === 2) {
    const shoulders = {
      x: (shoulderPoints[0].x + shoulderPoints[1].x) / 2,
      y: (shoulderPoints[0].y + shoulderPoints[1].y) / 2,
    };
    const hips = {
      x: (hipPoints[0].x + hipPoints[1].x) / 2,
      y: (hipPoints[0].y + hipPoints[1].y) / 2,
    };
    if (Math.abs(shoulders.x - hips.x) * (4 / 3) > Math.abs(shoulders.y - hips.y) * 1.4) {
      return "horizontal posture";
    }

  }

  if (track.previousCenter && track.previousAt) {
    const elapsed = Math.max((now - track.previousAt) / 1000, 0.001);
    const centerX = detection.x + detection.width / 2;
    const centerY = detection.y + detection.height / 2;
    const distance = Math.hypot(centerX - track.previousCenter.x, centerY - track.previousCenter.y);
    const bodySize = Math.max(detection.width, detection.height, 0.05);
    const normalizedSpeed = distance / bodySize / elapsed;
    if (normalizedSpeed >= 0.6) return "possible running";
    if (normalizedSpeed >= 0.12) return "possible walking";
  }

  const kneeAngles = [
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
  ].map(([hip, knee, ankle]) => {
    if (!points.has(hip) || !points.has(knee) || !points.has(ankle)) return null;
    return jointAngle(points.get(hip), points.get(knee), points.get(ankle));
  }).filter(Number.isFinite);
  if (kneeAngles.length > 0) {
    const averageAngle = kneeAngles.reduce((total, angle) => total + angle, 0) / kneeAngles.length;
    if (averageAngle < 120) return "sitting posture";
    if (averageAngle >= 145) return "standing posture";
    return "posture uncertain";
  }
  if (shoulderPoints.length === 2 && hipPoints.length === 2) return "posture uncertain";
  return "pose unavailable";
}

function estimateBodyPartMovement(detection, track) {
  const current = pointMap(detection.pose);
  const previous = pointMap(track.pose);
  if (!track.box || previous.size === 0) return [];

  const parts = [
    ["left arm", ["left_elbow", "left_wrist"]],
    ["right arm", ["right_elbow", "right_wrist"]],
    ["left leg", ["left_knee", "left_ankle"]],
    ["right leg", ["right_knee", "right_ankle"]],
  ];
  return parts.filter(([, jointNames]) => {
    let largestChange = 0;
    for (const name of jointNames) {
      const currentPoint = current.get(name);
      const previousPoint = previous.get(name);
      if (!currentPoint || !previousPoint) continue;
      const currentX = (currentPoint.x - detection.x) / Math.max(detection.width, 0.01);
      const currentY = (currentPoint.y - detection.y) / Math.max(detection.height, 0.01);
      const previousX = (previousPoint.x - track.box.x) / Math.max(track.box.width, 0.01);
      const previousY = (previousPoint.y - track.box.y) / Math.max(track.box.height, 0.01);
      largestChange = Math.max(largestChange, Math.hypot(currentX - previousX, currentY - previousY));
    }
    return largestChange >= 0.12;
  }).map(([part]) => `${part} moving`);
}

function centerOf(detection) {
  return {
    x: detection.x + detection.width / 2,
    y: detection.y + detection.height / 2,
  };
}

function intersectionOverUnion(first, second) {
  const left = Math.max(first.x, second.x);
  const top = Math.max(first.y, second.y);
  const right = Math.min(first.x + first.width, second.x + second.width);
  const bottom = Math.min(first.y + first.height, second.y + second.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = first.width * first.height + second.width * second.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function assignPersonTracks(detections) {
  trackFrame += 1;
  const now = performance.now();
  const availableTracks = [...peopleTracks.values()].filter(
    (track) => trackFrame - track.lastFrame <= MAX_TRACK_MISSED_FRAMES,
  );
  const matchedIds = new Set();
  const entries = [];
  let movementCount = 0;

  const people = detections.filter((detection) => detection.label === "person");
  for (const detection of people) {
    let bestTrack;
    let bestScore = Infinity;
    const center = centerOf(detection);
    for (const track of availableTracks) {
      if (matchedIds.has(track.id)) continue;
      const distance = Math.hypot(center.x - track.center.x, center.y - track.center.y);
      const overlap = intersectionOverUnion(detection, track.box);
      if (distance > PERSON_MATCH_DISTANCE && overlap < 0.05) continue;
      const score = distance + (1 - overlap) * 0.08;
      if (score < bestScore) {
        bestScore = score;
        bestTrack = track;
      }
    }

    const isNew = !bestTrack;
    const bodyParts = isNew ? [] : estimateBodyPartMovement(detection, bestTrack);
    if (isNew) {
      bestTrack = {
        id: `P${nextPersonId++}`,
        startedAt: now,
        lastFrame: trackFrame,
        lastAt: now,
        center,
        box: { ...detection },
      };
      peopleTracks.set(bestTrack.id, bestTrack);
      entries.push(bestTrack.id);
    } else {
      const distance = Math.hypot(center.x - bestTrack.center.x, center.y - bestTrack.center.y);
      if (distance >= MOVEMENT_THRESHOLD) movementCount += 1;
      bestTrack.previousCenter = bestTrack.center;
      bestTrack.previousAt = bestTrack.lastAt;
      bestTrack.lastAt = now;
      bestTrack.lastFrame = trackFrame;
      bestTrack.center = center;
    }
    matchedIds.add(bestTrack.id);
    detection.trackId = bestTrack.id;
    detection.trackIsNew = isNew;
    detection.bodyParts = bodyParts;
    detection.activity = estimateActivity(detection, bestTrack, now);
    detection.dwellSeconds = Math.max(0, Math.floor((now - bestTrack.startedAt) / 1000));
    bestTrack.activity = detection.activity;
    bestTrack.pose = detection.pose || [];
    bestTrack.box = {
      x: detection.x,
      y: detection.y,
      width: detection.width,
      height: detection.height,
    };
  }

  let exits = 0;
  for (const [id, track] of peopleTracks) {
    if (trackFrame - track.lastFrame > MAX_TRACK_MISSED_FRAMES) {
      peopleTracks.delete(id);
      exits += 1;
    }
  }
  return { people, entries, exits, movementCount };
}

function compareObjectPositions(detections) {
  const currentObjects = detections.filter((detection) => detection.label !== "person");
  const previous = previousObjectDetections;
  const matchedPrevious = new Set();
  const states = currentObjects.map((detection) => {
    const center = centerOf(detection);
    let bestIndex = -1;
    let bestDistance = Infinity;
    previous.forEach((candidate, index) => {
      if (matchedPrevious.has(index) || candidate.label !== detection.label) return;
      const candidateCenter = centerOf(candidate);
      const distance = Math.hypot(center.x - candidateCenter.x, center.y - candidateCenter.y);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    });
    if (bestIndex < 0 || bestDistance > 0.25) return { ...detection, movement: "new" };
    matchedPrevious.add(bestIndex);
    return { ...detection, movement: bestDistance >= MOVEMENT_THRESHOLD ? "changed" : "stable" };
  });
  previousObjectDetections = currentObjects.map((detection) => ({ ...detection }));
  return states;
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function renderScene(result, tracks, objectStates, motion) {
  const people = tracks.people;
  const otherObjects = result.objects.filter((item) => item.label !== "person");
  elements.peopleCount.textContent = String(people.length);
  elements.occupancyPeak.textContent = String(maxOccupancy);
  const activeTracks = [...peopleTracks.values()].filter(
    (track) => trackFrame - track.lastFrame <= MAX_TRACK_MISSED_FRAMES,
  );
  const averageDwell = activeTracks.length
    ? Math.round(activeTracks.reduce((total, track) => total + (performance.now() - track.startedAt) / 1000, 0) / activeTracks.length)
    : 0;
  elements.averageDwell.textContent = activeTracks.length ? formatDuration(averageDwell) : "—";
  elements.objectCount.textContent = String(otherObjects.reduce((total, item) => total + item.count, 0));

  const rows = [
    ["People", people.length ? people.map((item) => {
      const bodyParts = item.bodyParts.length
        ? item.bodyParts.join(", ")
        : item.trackIsNew ? "collecting pose baseline" : "no clear limb change";
      return `${item.trackId} · ${item.activity} · ${bodyParts} · ${formatDuration(item.dwellSeconds)}`;
    }).join("; ") : "None detected"],
    ["Other objects", otherObjects.map((item) => `${item.label} × ${item.count}`).join(", ") || "None recognized"],
    ["Object movement", objectStates.map((item) => `${item.label}: ${item.movement === "changed" ? "position changed" : item.movement === "stable" ? "stable" : "new"}`).join("; ") || "—"],
    ["Scene motion", motion.motion_detected ? `Detected (${motion.motion_percent}%)` : "Not detected"],
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

function getHeadRegion(detection, canvas) {
  const points = pointMap(detection.pose);
  const facialPoints = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
    .map((name) => points.get(name))
    .filter(Boolean);
  const personX = detection.x * canvas.width;
  const personY = detection.y * canvas.height;
  const personWidth = detection.width * canvas.width;
  const personHeight = detection.height * canvas.height;
  if (facialPoints.length < 2) {
    return {
      x: personX + personWidth * 0.2,
      y: personY,
      width: personWidth * 0.6,
      height: Math.max(12, personHeight * 0.18),
      landmarkBased: false,
    };
  }

  const pixels = facialPoints.map((point) => ({
    x: point.x * canvas.width,
    y: point.y * canvas.height,
  }));
  const eyes = [points.get("left_eye"), points.get("right_eye")].filter(Boolean);
  const nose = points.get("nose");
  const minX = Math.min(...pixels.map((point) => point.x));
  const maxX = Math.max(...pixels.map((point) => point.x));
  const eyeDistance = eyes.length === 2
    ? Math.abs(eyes[0].x - eyes[1].x) * canvas.width
    : 0;
  const width = Math.min(
    canvas.width,
    Math.max(maxX - minX, eyeDistance * 2.2, personWidth * 0.13, 16),
  );
  const eyeY = eyes.length
    ? eyes.reduce((total, point) => total + point.y * canvas.height, 0) / eyes.length
    : Math.min(...pixels.map((point) => point.y));
  const noseY = nose ? nose.y * canvas.height : eyeY + width * 0.35;
  const height = Math.min(canvas.height, width * 1.3);
  const x = Math.max(0, Math.min(canvas.width - width, (minX + maxX - width) / 2));
  const y = Math.max(0, Math.min(canvas.height - height, eyeY - height * 0.35));
  const boundedHeight = Math.min(canvas.height - y, Math.max(height, noseY - y + width * 0.2));
  return {
    x,
    y,
    width,
    height: boundedHeight,
    landmarkBased: true,
  };
}

function drawHeadFocus(detection, region, canvas) {
  overlayContext.strokeStyle = "#ffd166";
  overlayContext.lineWidth = 2;
  overlayContext.strokeRect(region.x, region.y, region.width, region.height);
  const title = `${detection.trackId} · HEAD`;
  const labelWidth = Math.min(canvas.width - region.x, overlayContext.measureText(title).width + 10);
  overlayContext.fillStyle = "#101718e8";
  overlayContext.fillRect(region.x, Math.max(0, region.y - 19), labelWidth, 19);
  overlayContext.fillStyle = "#fff1bd";
  overlayContext.fillText(title, region.x + 5, Math.max(13, region.y - 5), Math.max(1, labelWidth - 8));

  faceFocusContext.clearRect(0, 0, elements.faceFocus.width, elements.faceFocus.height);
  faceFocusContext.save();
  faceFocusContext.translate(elements.faceFocus.width, 0);
  faceFocusContext.scale(-1, 1);
  faceFocusContext.drawImage(
    captureCanvas,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    elements.faceFocus.width,
    elements.faceFocus.height,
  );
  faceFocusContext.restore();
  elements.faceFocusPanel.hidden = false;
}

function drawDetections(detections) {
  const canvas = elements.cameraOverlay;
  overlayContext.clearRect(0, 0, canvas.width, canvas.height);
  overlayContext.lineWidth = 2;
  overlayContext.font = "12px system-ui, sans-serif";
  elements.faceFocusPanel.hidden = true;
  faceFocusContext.clearRect(0, 0, elements.faceFocus.width, elements.faceFocus.height);
  let headFocus;
  for (const detection of detections) {
    const x = detection.x * canvas.width;
    const y = detection.y * canvas.height;
    const width = detection.width * canvas.width;
    const height = detection.height * canvas.height;
    const person = detection.label === "person";
    if (!person) {
      overlayContext.strokeStyle = "#70b8f5";
      overlayContext.strokeRect(x, y, width, height);
      overlayContext.fillStyle = "#101718e8";
      const labelWidth = Math.min(canvas.width - x, overlayContext.measureText(detection.label).width + 10);
      overlayContext.fillRect(x, Math.max(0, y - 19), labelWidth, 19);
      overlayContext.fillStyle = "#e6ece9";
      overlayContext.fillText(detection.label, x + 5, Math.max(13, y - 5), Math.max(1, labelWidth - 8));
      continue;
    }

    const region = getHeadRegion(detection, canvas);
    if (!headFocus || height > headFocus.height) headFocus = { detection, region, height };
    const points = pointMap(detection.pose);
    overlayContext.strokeStyle = "#64d5a0";
    for (const [firstName, lastName] of SKELETON_LINKS) {
      const first = points.get(firstName);
      const last = points.get(lastName);
      if (!first || !last) continue;
      overlayContext.beginPath();
      overlayContext.moveTo(first.x * canvas.width, first.y * canvas.height);
      overlayContext.lineTo(last.x * canvas.width, last.y * canvas.height);
      overlayContext.stroke();
    }
    for (const point of points.values()) {
      overlayContext.beginPath();
      overlayContext.arc(point.x * canvas.width, point.y * canvas.height, 2.5, 0, Math.PI * 2);
      overlayContext.fillStyle = "#f3ca79";
      overlayContext.fill();
    }
  }
  if (headFocus) drawHeadFocus(headFocus.detection, headFocus.region, canvas);
}

function recordSceneSample(result, tracks, motion) {
  sceneWindowSamples += 1;
  sceneWindowPeak = Math.max(sceneWindowPeak, tracks.people.length);
  sceneWindowEntries += tracks.entries.length;
  sceneWindowExits += tracks.exits;
  sceneWindowMovementSamples += tracks.movementCount;
  for (const person of tracks.people) {
    const stats = sceneWindowPeople.get(person.trackId) || {
      samples: 0,
      dwellSeconds: person.dwellSeconds,
      activities: new Map(),
      bodyParts: new Map(),
    };
    stats.samples += 1;
    stats.dwellSeconds = Math.max(stats.dwellSeconds, person.dwellSeconds);
    stats.activities.set(person.activity, (stats.activities.get(person.activity) || 0) + 1);
    const bodyParts = person.bodyParts.length
      ? person.bodyParts
      : [person.trackIsNew ? "collecting pose baseline" : "no clear limb change"];
    for (const part of bodyParts) {
      stats.bodyParts.set(part, (stats.bodyParts.get(part) || 0) + 1);
    }
    sceneWindowPeople.set(person.trackId, stats);
  }
  for (const object of result.objects) {
    if (object.label === "person") continue;
    sceneWindowObjects.set(object.label, Math.max(sceneWindowObjects.get(object.label) || 0, object.count));
  }
  if (motion.motion_detected) sceneWindowMovementSamples += 1;
}

function appendSceneSummary() {
  if (!cameraActive) return;
  const timestamp = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  let description = "No camera frame was analyzed in this interval.";
  if (sceneWindowSamples > 0) {
    const parts = [`peak occupancy ${sceneWindowPeak}`];
    if (sceneWindowEntries) parts.push(`${sceneWindowEntries} track${sceneWindowEntries === 1 ? "" : "s"} appeared`);
    if (sceneWindowExits) parts.push(`${sceneWindowExits} track${sceneWindowExits === 1 ? "" : "s"} left view`);
    if (sceneWindowPeople.size) {
      const activitySummary = [...sceneWindowPeople.entries()]
        .sort(([left], [right]) => left.localeCompare(right, undefined, { numeric: true }))
        .map(([id, stats]) => {
          const [activity, count] = [...stats.activities.entries()]
            .sort((left, right) => right[1] - left[1])[0];
          const partChanges = [...stats.bodyParts.entries()]
            .filter(([part]) => part !== "collecting pose baseline" && part !== "no clear limb change")
            .sort((left, right) => right[1] - left[1])
            .slice(0, 2)
            .map(([part]) => part);
          const bodyDescription = partChanges.join(", ") || "no clear limb change";
          return `${id}: ${activity}; ${bodyDescription} (${count}/${stats.samples} activity samples), observed ${formatDuration(stats.dwellSeconds)}`;
        });
      parts.push(activitySummary.join(", "));
    }
    if (sceneWindowObjects.size) {
      parts.push([...sceneWindowObjects.entries()]
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([label, count]) => `${label} × ${count}`).join(", "));
    }
    if (sceneWindowMovementSamples) parts.push("movement observed");
    description = parts.join(" · ");
  }

  const item = createTextElement("article", "timeline-item", "");
  item.append(createTextElement("time", "timeline-time", timestamp));
  item.append(createTextElement("span", "timeline-description", description));
  if (elements.sceneTimeline.querySelector(".empty-state")) elements.sceneTimeline.replaceChildren();
  elements.sceneTimeline.prepend(item);
  while (elements.sceneTimeline.children.length > MAX_TIMELINE_ITEMS) {
    elements.sceneTimeline.lastElementChild.remove();
  }
  sceneWindowSamples = 0;
  sceneWindowPeak = 0;
  sceneWindowEntries = 0;
  sceneWindowExits = 0;
  sceneWindowMovementSamples = 0;
  sceneWindowPeople = new Map();
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

function resetMonitorState() {
  elements.faceFocusPanel.hidden = true;
  faceFocusContext.clearRect(0, 0, elements.faceFocus.width, elements.faceFocus.height);
  trackFrame = 0;
  nextPersonId = 1;
  maxOccupancy = 0;
  peopleTracks = new Map();
  previousObjectDetections = [];
  previousMotionFrame = undefined;
  sceneWindowSamples = 0;
  sceneWindowPeak = 0;
  sceneWindowEntries = 0;
  sceneWindowExits = 0;
  sceneWindowMovementSamples = 0;
  sceneWindowPeople = new Map();
  sceneWindowObjects = new Map();
  elements.sceneTimeline.replaceChildren(
    createTextElement("div", "empty-state", "Updates appear here while monitoring is active."),
  );
  elements.peopleCount.textContent = "—";
  elements.occupancyPeak.textContent = "—";
  elements.averageDwell.textContent = "—";
  elements.objectCount.textContent = "—";
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
  overlayContext.clearRect(0, 0, 320, 240);
  resetMonitorState();
  elements.cameraPlaceholder.hidden = false;
  elements.cameraLive.classList.remove("visible");
  elements.sceneResults.replaceChildren(
    createTextElement("div", "empty-state", "Camera is off. No video or event history is retained."),
  );
  elements.cameraStatus.textContent = "CAMERA OFF";
  elements.startCamera.hidden = false;
  elements.stopCamera.hidden = true;
  elements.cameraConsent.disabled = false;
  elements.cameraMessage.textContent = message || "No camera is active.";
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
        const tracks = assignPersonTracks(result.detections);
        maxOccupancy = Math.max(maxOccupancy, tracks.people.length);
        const objectStates = compareObjectPositions(result.detections);
        const motion = measureMotion();
        recordSceneSample(result, tracks, motion);
        renderScene(result, tracks, objectStates, motion);
        elements.cameraMessage.textContent = "Local estimates are active. Stop monitoring to clear temporary tracks and updates.";
      }
      await new Promise((resolve) => window.setTimeout(resolve, 750));
    } catch (error) {
      if (run !== cameraRun) return;
      const message = error instanceof Error ? error.message : "Camera analysis failed.";
      stopCamera(`Monitoring stopped: ${message}`);
      return;
    }
  }
}

async function startCamera() {
  if (!elements.cameraConsent.checked) {
    elements.cameraMessage.textContent = "Confirm the consent statement before starting monitoring.";
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
    resetMonitorState();
    elements.cameraPlaceholder.hidden = true;
    elements.cameraLive.classList.add("visible");
    elements.cameraStatus.textContent = "CAMERA ON";
    elements.startCamera.hidden = true;
    elements.stopCamera.hidden = false;
    elements.cameraConsent.disabled = true;
    elements.cameraMessage.textContent = "Camera is active. Frames are processed locally and discarded.";
    timelineTimer = window.setInterval(appendSceneSummary, SCENE_SUMMARY_INTERVAL_MS);
    captureFrames(run);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Camera permission was not granted.";
    stopCamera(`Could not start camera: ${message}`);
  } finally {
    elements.startCamera.disabled = false;
  }
}

elements.startCamera.addEventListener("click", startCamera);
elements.stopCamera.addEventListener("click", () => stopCamera("Monitoring stopped. Temporary tracks and updates were cleared."));
elements.cameraConsent.addEventListener("change", () => {
  if (!elements.cameraConsent.checked && cameraActive) stopCamera("Consent withdrawn. Monitoring stopped.");
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && cameraActive) stopCamera("Monitoring stopped because this tab is no longer visible.");
});
window.addEventListener("pagehide", () => stopCamera(""));
