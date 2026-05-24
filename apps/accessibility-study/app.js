const STORAGE_KEY = "sidewalk_accessibility_study_v2_pittsburgh";

const state = loadState();
let trials = { imageTrials: [], routeTrials: [] };
let trialStartedAt = Date.now();
let zoomUsed = false;

const app = document.querySelector("#app");
const progressText = document.querySelector("#progressText");
const progressBar = document.querySelector("#progressBar");

function loadState() {
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved) return JSON.parse(saved);
  const id = crypto.randomUUID ? crypto.randomUUID() : `p-${Date.now()}-${Math.random()}`;
  return {
    participant_id: id,
    session_id: `s-${Date.now()}`,
    step: "consent",
    currentImageIndex: 0,
    currentRouteIndex: 0,
    profile: {},
    imageResponses: [],
    routeResponses: [],
    usability: null,
    created_at: new Date().toISOString(),
  };
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

async function init() {
  const response = await fetch(`data/trials.json?v=${Date.now()}`, { cache: "no-store" });
  trials = await response.json();
  render();
}

function render() {
  saveState();
  updateProgress();
  if (state.step === "consent") return renderTemplate("consentTemplate", bindConsent);
  if (state.step === "profile") return renderTemplate("profileTemplate", bindProfile);
  if (state.step === "tutorial") return renderTemplate("tutorialTemplate", bindTutorial);
  if (state.step === "image") return renderImageTrial();
  if (state.step === "route") return renderRouteTrial();
  if (state.step === "usability") return renderTemplate("usabilityTemplate", bindUsability);
  return renderDone();
}

function renderTemplate(id, binder) {
  app.innerHTML = "";
  app.append(document.querySelector(`#${id}`).content.cloneNode(true));
  binder();
  app.focus();
}

function bindConsent() {
  app.querySelector("[data-action='start']").addEventListener("click", () => {
    state.step = "profile";
    render();
  });
}

function bindProfile() {
  const nameInput = app.querySelector("#participantName");
  const aid = app.querySelector("#aidSelect");
  const experience = app.querySelector("#experienceSelect");
  nameInput.value = state.profile.participant_name || "";
  aid.value = state.profile.primary_aid || "";
  experience.value = state.profile.experience_level || "";
  bindChips("concerns", state.profile.concerns || []);

  app.querySelector("[data-action='save-profile']").addEventListener("click", () => {
    if (!aid.value || !experience.value) {
      alert("Please select your mobility aid and how often sidewalk conditions affect you.");
      return;
    }
    state.profile = {
      participant_name: nameInput.value.trim() || null,
      primary_aid: aid.value,
      experience_level: experience.value,
      concerns: selectedChips("concerns"),
      device_width: window.innerWidth,
      user_agent: navigator.userAgent,
      saved_at: new Date().toISOString(),
    };
    state.step = "tutorial";
    render();
  });
}

function bindTutorial() {
  app.querySelector("[data-action='begin-trials']").addEventListener("click", () => {
    state.step = "image";
    trialStartedAt = Date.now();
    render();
  });
}

function renderImageTrial() {
  if (state.currentImageIndex >= trials.imageTrials.length) {
    state.step = "route";
    trialStartedAt = Date.now();
    return render();
  }
  app.innerHTML = "";
  app.append(document.querySelector("#imageTrialTemplate").content.cloneNode(true));
  const trial = trials.imageTrials[state.currentImageIndex];
  zoomUsed = false;
  app.querySelector("#trialImage").src = trial.src;
  app.querySelector("#trialImage").alt = trial.alt;
  app.querySelector("#trialCaption").textContent = trial.caption;
  bindChips("imageReasons", []);
  bindZoom(app.querySelector("#trialImage"));
  app.querySelectorAll("[data-response]").forEach((button) => {
    button.addEventListener("click", () => recordImageResponse(trial, button.dataset.response));
  });
  trialStartedAt = Date.now();
  app.focus();
}

function recordImageResponse(trial, response) {
  const record = {
    trial_id: `${state.session_id}-image-${state.currentImageIndex}`,
    participant_id: state.participant_id,
    session_id: state.session_id,
    trial_type: "image",
    image_id: trial.id,
    image_src: trial.src,
    mobility_aid: state.profile.primary_aid,
    response,
    reason_chips: selectedChips("imageReasons"),
    response_time_ms: Date.now() - trialStartedAt,
    zoom_used: zoomUsed,
    skipped: response === "skip",
    created_at: new Date().toISOString(),
  };
  state.imageResponses.push(record);
  state.currentImageIndex += 1;
  render();
}

function renderRouteTrial() {
  if (state.currentRouteIndex >= trials.routeTrials.length) {
    state.step = "usability";
    return render();
  }
  app.innerHTML = "";
  app.append(document.querySelector("#routeTrialTemplate").content.cloneNode(true));
  const trial = trials.routeTrials[state.currentRouteIndex];
  app.querySelector("#routePrompt").textContent = trial.prompt;
  renderRouteOption(app.querySelector("#routeAOption"), trial.routeA);
  renderRouteOption(app.querySelector("#routeBOption"), trial.routeB);
  bindChips("routeReasons", []);
  app.querySelectorAll(".route-grid img").forEach((image) => bindZoom(image));
  app.querySelectorAll("[data-route-response]").forEach((button) => {
    button.addEventListener("click", () => recordRouteResponse(trial, button.dataset.routeResponse));
  });
  trialStartedAt = Date.now();
  app.focus();
}

function renderRouteOption(container, route) {
  container.innerHTML = "";
  const caption = document.createElement("figcaption");
  caption.textContent = route.label;

  if (Array.isArray(route.images)) {
    const evidence = document.createElement("div");
    evidence.className = "route-evidence";
    route.images.forEach((item, index) => {
      const wrap = document.createElement("div");
      const image = document.createElement("img");
      const label = document.createElement("span");
      image.src = item.src;
      image.alt = item.alt || `${route.label} street-view scene ${index + 1}`;
      label.textContent = `Scene ${index + 1}`;
      wrap.append(image, label);
      evidence.append(wrap);
    });
    container.append(evidence, caption);
    return;
  }

  const image = document.createElement("img");
  image.src = route.src;
  image.alt = route.alt || route.label;
  container.append(image, caption);
}

function recordRouteResponse(trial, selected) {
  const record = {
    trial_id: `${state.session_id}-route-${state.currentRouteIndex}`,
    participant_id: state.participant_id,
    session_id: state.session_id,
    trial_type: "route",
    route_pair_id: trial.id,
    route_a_id: trial.routeA.id,
    route_b_id: trial.routeB.id,
    route_a_type: trial.routeA.type,
    route_b_type: trial.routeB.type,
    route_a_images: routeImageList(trial.routeA).join("|"),
    route_b_images: routeImageList(trial.routeB).join("|"),
    selected_route: selected,
    reason_chips: selectedChips("routeReasons"),
    response_time_ms: Date.now() - trialStartedAt,
    skipped: selected === "skip",
    created_at: new Date().toISOString(),
  };
  state.routeResponses.push(record);
  state.currentRouteIndex += 1;
  render();
}

function routeImageList(route) {
  if (Array.isArray(route.images)) return route.images.map((item) => item.src);
  return route.src ? [route.src] : [];
}

function bindUsability() {
  app.querySelector("[data-action='finish']").addEventListener("click", () => {
    state.usability = {
      participant_id: state.participant_id,
      session_id: state.session_id,
      ease_score: Number(app.querySelector("#easeScore").value),
      information_score: Number(app.querySelector("#infoScore").value),
      reuse_score: Number(app.querySelector("#reuseScore").value),
      free_text: app.querySelector("#freeText").value.trim(),
      created_at: new Date().toISOString(),
    };
    state.step = "done";
    render();
  });
}

function renderDone() {
  renderTemplate("doneTemplate", () => {
    app.querySelector("#doneSummary").textContent =
      `Saved ${state.imageResponses.length} image responses and ${state.routeResponses.length} route responses.`;
    app.querySelector("[data-action='download-json']").addEventListener("click", downloadJson);
    app.querySelector("[data-action='download-csv']").addEventListener("click", downloadCsv);
    app.querySelector("[data-action='reset']").addEventListener("click", resetSession);
    submitToServer();
  });
}

async function submitToServer() {
  const statusEl = app.querySelector("#submitStatus");
  if (statusEl) statusEl.textContent = "Submitting…";
  try {
    const res = await fetch("/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(exportPayload()),
    });
    if (res.ok) {
      if (statusEl) statusEl.textContent = "Responses saved to research server.";
    } else {
      if (statusEl) statusEl.textContent = "Server save failed — please download your data below.";
    }
  } catch {
    if (statusEl) statusEl.textContent = "Offline — please download your data below.";
  }
}

function bindChips(group, selectedValues) {
  app.querySelectorAll(`[data-chip-group='${group}'] button`).forEach((button) => {
    const selected = selectedValues.includes(button.dataset.value);
    button.setAttribute("aria-pressed", String(selected));
    button.addEventListener("click", () => {
      const isPressed = button.getAttribute("aria-pressed") === "true";
      button.setAttribute("aria-pressed", String(!isPressed));
    });
  });
}

function selectedChips(group) {
  return [...app.querySelectorAll(`[data-chip-group='${group}'] button[aria-pressed='true']`)].map(
    (button) => button.dataset.value
  );
}

function bindZoom(image) {
  image.addEventListener("click", () => openModal(image.src, image.alt));
  const zoomButton = app.querySelector("#zoomButton");
  if (zoomButton) zoomButton.addEventListener("click", () => openModal(image.src, image.alt));
}

function openModal(src, alt) {
  zoomUsed = true;
  const modal = document.querySelector("#modal");
  const modalImage = document.querySelector("#modalImage");
  modalImage.src = src;
  modalImage.alt = alt;
  modal.hidden = false;
}

document.querySelector("#closeModal").addEventListener("click", () => {
  document.querySelector("#modal").hidden = true;
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document.querySelector("#modal").hidden = true;
    return;
  }
  // Keyboard shortcuts for trial responses (only when modal is closed)
  if (!document.querySelector("#modal").hidden) return;
  const key = event.key;
  if (state.step === "image") {
    const map = { "1": "yes", "2": "unsure", "3": "no", "s": "skip", "S": "skip" };
    if (map[key]) {
      const btn = app.querySelector(`[data-response='${map[key]}']`);
      if (btn) btn.click();
    }
    if (key === "z" || key === "Z") {
      const img = app.querySelector("#trialImage");
      if (img) openModal(img.src, img.alt);
    }
  }
  if (state.step === "route") {
    const map = { "1": "route_a", "2": "route_b", "3": "no_preference", "4": "neither", "s": "skip", "S": "skip" };
    if (map[key]) {
      const btn = app.querySelector(`[data-route-response='${map[key]}']`);
      if (btn) btn.click();
    }
  }
});

function updateProgress() {
  const totalTasks = trials.imageTrials.length + trials.routeTrials.length;
  const completed = Math.min(state.imageResponses.length + state.routeResponses.length, totalTasks);
  if (state.step === "consent" || state.step === "profile" || state.step === "tutorial") {
    progressText.textContent = "Setup";
    progressBar.style.width = "0%";
    return;
  }
  if (state.step === "usability") {
    progressText.textContent = "Final questions";
    progressBar.style.width = "96%";
    return;
  }
  if (state.step === "done") {
    progressText.textContent = "Complete";
    progressBar.style.width = "100%";
    return;
  }
  progressText.textContent = `${completed} / ${totalTasks}`;
  progressBar.style.width = `${Math.round((completed / Math.max(1, totalTasks)) * 100)}%`;
}

function exportPayload() {
  return {
    participant: {
      participant_id: state.participant_id,
      session_id: state.session_id,
      created_at: state.created_at,
      ...state.profile,
    },
    image_trials: state.imageResponses,
    route_trials: state.routeResponses,
    usability: state.usability,
    exported_at: new Date().toISOString(),
  };
}

function downloadJson() {
  downloadBlob(
    JSON.stringify(exportPayload(), null, 2),
    `accessibility-study-${state.session_id}.json`,
    "application/json"
  );
}

function downloadCsv() {
  const rows = [
    ...state.imageResponses.map((row) => ({ ...row, reason_chips: row.reason_chips.join("|") })),
    ...state.routeResponses.map((row) => ({ ...row, reason_chips: row.reason_chips.join("|") })),
  ];
  const keys = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const csv = [keys.join(","), ...rows.map((row) => keys.map((key) => csvCell(row[key])).join(","))].join("\n");
  downloadBlob(csv, `accessibility-study-${state.session_id}.csv`, "text/csv");
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value).replaceAll('"', '""');
  return /[",\n]/.test(text) ? `"${text}"` : text;
}

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function resetSession() {
  if (!confirm("Reset this local session? Export first if you need the data.")) return;
  localStorage.removeItem(STORAGE_KEY);
  location.reload();
}

init().catch((error) => {
  app.innerHTML = `<section class="panel"><h2>Could not load study</h2><p>${error.message}</p></section>`;
});
