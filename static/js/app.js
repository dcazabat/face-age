const okStatusClass = "status status-ok";
const errorStatusClass = "status status-error";

const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const overlayEl = document.getElementById("overlay");
const cameraToggleBtn = document.getElementById("camera-toggle-btn");
const captureBtn = document.getElementById("capture-btn");
const refreshIntervalInput = document.getElementById("refresh-interval");
const refreshIntervalValue = document.getElementById("refresh-interval-value");

const uploadCanvas = document.createElement("canvas");
const uploadCtx = uploadCanvas.getContext("2d");
const overlayCtx = overlayEl.getContext("2d");
const analyzeUrl = document.body.dataset.faceAgeAnalyzeUrl || "/analyze";
const clientLogUrl = document.body.dataset.faceAgeClientLogUrl || "/client-log";
const refreshIntervalStorageKey = document.body.dataset.faceAgeRefreshStorageKey || "face-age-refresh-interval-seconds";

let stream = null;
let analyzing = false;
let analyzeTimer = null;
let lastAnalyzeError = null;

function loadRefreshInterval() {
  try {
    const savedValue = Number(window.localStorage.getItem(refreshIntervalStorageKey));
    if (Number.isFinite(savedValue)) {
      const boundedValue = Math.min(10, Math.max(1, savedValue));
      refreshIntervalInput.value = String(boundedValue);
    }
  } catch (_) {
    // If storage is unavailable, fall back to the default value.
  }
}

function saveRefreshInterval() {
  try {
    window.localStorage.setItem(refreshIntervalStorageKey, refreshIntervalInput.value);
  } catch (_) {
    // Ignore storage failures and keep the current in-memory value.
  }
}

function getAnalyzeIntervalMs() {
  const seconds = Number(refreshIntervalInput.value || 3);
  return Math.min(10, Math.max(1, seconds)) * 1000;
}

function updateRefreshLabel() {
  refreshIntervalValue.textContent = `${refreshIntervalInput.value} s`;
}

function updateCameraToggleLabel() {
  const isCameraRunning = Boolean(stream);
  cameraToggleBtn.textContent = isCameraRunning ? "Apagar cámara" : "Encender cámara";
  cameraToggleBtn.className = isCameraRunning
    ? "camera-toggle-btn is-on rounded-lg border border-amber-400/40 bg-amber-500/15 px-4 py-2 text-sm font-semibold text-amber-300 transition hover:bg-amber-500/25"
    : "camera-toggle-btn rounded-lg border border-emerald-400/40 bg-emerald-500/15 px-4 py-2 text-sm font-semibold text-emerald-300 transition hover:bg-emerald-500/25";
}

function restartAnalysisLoop() {
  if (!stream) return;
  stopAnalysisLoop();
  analyzeTimer = setInterval(analyzeOnce, getAnalyzeIntervalMs());
}

async function sendClientLog(event, message, details = {}) {
  if (!clientLogUrl) return;
  try {
    const payload = JSON.stringify({ event, message, details });
    if (navigator.sendBeacon) {
      const beaconSent = navigator.sendBeacon(
        clientLogUrl,
        new Blob([payload], { type: "application/json" }),
      );
      if (beaconSent) return;
    }
    await fetch(clientLogUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      keepalive: true,
    });
  } catch (_) {
    // Logging must never break the camera flow.
  }
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? errorStatusClass : okStatusClass;
}

function stopAnalysisLoop() {
  if (analyzeTimer) {
    clearInterval(analyzeTimer);
    analyzeTimer = null;
  }
}

function stopCameraTracks() {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    track.stop();
  }
  stream = null;
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlayEl.width, overlayEl.height);
}

function getTimestampName() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const y = now.getFullYear();
  const m = pad(now.getMonth() + 1);
  const d = pad(now.getDate());
  const hh = pad(now.getHours());
  const mm = pad(now.getMinutes());
  const ss = pad(now.getSeconds());
  return `capture_${y}${m}${d}_${hh}${mm}${ss}.jpg`;
}

function getCurrentFrameDataUrl() {
  const width = videoEl.videoWidth;
  const height = videoEl.videoHeight;
  if (!width || !height) return null;
  uploadCanvas.width = width;
  uploadCanvas.height = height;
  uploadCtx.drawImage(videoEl, 0, 0, width, height);
  return uploadCanvas.toDataURL("image/jpeg", 0.82);
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    const raw = await response.text();
    throw new Error(`Respuesta no JSON (${response.status}): ${raw.slice(0, 120)}`);
  }
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || "No se pudo completar la acción");
  }
  return data;
}

function drawPredictions(predictions, frameWidth, frameHeight) {
  const w = videoEl.clientWidth;
  const h = videoEl.clientHeight;
  if (!w || !h) return;

  overlayEl.width = w;
  overlayEl.height = h;
  overlayCtx.clearRect(0, 0, w, h);

  const scaleX = w / frameWidth;
  const scaleY = h / frameHeight;

  overlayCtx.lineWidth = 2;
  overlayCtx.font = "14px sans-serif";

  for (const p of predictions) {
    const x = p.x * scaleX;
    const y = p.y * scaleY;
    const bw = p.w * scaleX;
    const bh = p.h * scaleY;

    overlayCtx.strokeStyle = "#22c55e";
    overlayCtx.strokeRect(x, y, bw, bh);

    const label = `Edad ~${p.age}+ | ${p.gender} | ${p.emotion} (${p.confidence.toFixed(1)}%)`;
    overlayCtx.fillStyle = "#00e5ff";
    overlayCtx.fillText(label, x, Math.max(16, y - 6));
  }
}

async function analyzeOnce() {
  if (!stream || analyzing) return;
  const image = getCurrentFrameDataUrl();
  if (!image) return;
  analyzing = true;
  try {
    const data = await postJson(analyzeUrl, { image });
    drawPredictions(data.predictions, data.frame_width, data.frame_height);
    lastAnalyzeError = null;
  } catch (error) {
    setStatus(error.message, true);
    if (error.message !== lastAnalyzeError) {
      lastAnalyzeError = error.message;
      sendClientLog("analysis_error", error.message, {
        source: "analyzeOnce",
        page: window.location.href,
      });
    }
  } finally {
    analyzing = false;
  }
}

async function startCamera() {
  if (stream) return;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        aspectRatio: { ideal: 16 / 9 },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    updateCameraToggleLabel();
    sendClientLog("camera_started", "La cámara se encendió correctamente.", {
      page: window.location.href,
    });
    restartAnalysisLoop();
  } catch (error) {
    setStatus("No se pudo abrir la cámara del navegador.", true);
    sendClientLog("camera_start_error", error.message, {
      page: window.location.href,
    });
  }
}

function stopCamera() {
  stopAnalysisLoop();
  stopCameraTracks();
  videoEl.srcObject = null;
  clearOverlay();
  updateCameraToggleLabel();
}

async function captureFrame() {
  if (!stream) {
    setStatus("Primero encendé la cámara.", true);
    return;
  }
  try {
    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;
    if (!width || !height) {
      throw new Error("No hay frame disponible para capturar.");
    }

    const shouldDownload = window.confirm("¿Querés descargar la captura?");
    if (!shouldDownload) {
      setStatus("Captura descartada.");
      return;
    }

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = width;
    captureCanvas.height = height;
    const captureCtx = captureCanvas.getContext("2d");
    captureCtx.drawImage(videoEl, 0, 0, width, height);
    captureCtx.drawImage(overlayEl, 0, 0, width, height);

    const downloadUrl = captureCanvas.toDataURL("image/jpeg", 0.92);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = getTimestampName();
    document.body.appendChild(link);
    link.click();
    link.remove();

    setStatus("Captura descargada en tu dispositivo.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

cameraToggleBtn.addEventListener("click", () => {
  if (stream) {
    stopCamera();
  } else {
    startCamera();
  }
});
captureBtn.addEventListener("click", captureFrame);
loadRefreshInterval();
saveRefreshInterval();
updateRefreshLabel();
refreshIntervalInput.addEventListener("input", () => {
  saveRefreshInterval();
  updateRefreshLabel();
  if (stream) {
    restartAnalysisLoop();
  }
});
window.addEventListener("beforeunload", stopCamera);

updateCameraToggleLabel();
