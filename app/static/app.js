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
};

let refreshTimer;
let toastTimer;
let previousCounters = new Map();

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
  return `${formatBytes((value - previous.value) / elapsed)}/s`;
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
refresh();
