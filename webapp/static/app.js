const state = {
  jobId: null,
  pollTimer: null,
  rects: [],
  drawing: false,
  startX: 0,
  startY: 0,
  currentRect: null,
};

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const autoMaskBtn = document.getElementById("autoMaskBtn");
const clearRectsBtn = document.getElementById("clearRectsBtn");
const uploadManualMaskBtn = document.getElementById("uploadManualMaskBtn");
const runAutoBtn = document.getElementById("runAutoBtn");
const runManualBtn = document.getElementById("runManualBtn");

const jobIdEl = document.getElementById("jobId");
const jobStatusEl = document.getElementById("jobStatus");
const jobMessageEl = document.getElementById("jobMessage");
const jobErrorEl = document.getElementById("jobError");

const inputImage = document.getElementById("inputImage");
const autoMaskImage = document.getElementById("autoMaskImage");
const outputImage = document.getElementById("outputImage");
const downloadLink = document.getElementById("downloadLink");

const canvas = document.getElementById("markCanvas");
const ctx = canvas.getContext("2d");

function setButtonsEnabled(enabled) {
  autoMaskBtn.disabled = !enabled;
  clearRectsBtn.disabled = !enabled;
  uploadManualMaskBtn.disabled = !enabled;
  runAutoBtn.disabled = !enabled;
  runManualBtn.disabled = !enabled;
}

function setStatus(job) {
  jobIdEl.textContent = job.id || "-";
  jobStatusEl.textContent = job.status || "-";
  jobMessageEl.textContent = job.message || "-";
  jobErrorEl.textContent = job.error || "-";
}

function syncCanvasSize() {
  const rect = inputImage.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width));
  canvas.height = Math.max(1, Math.floor(rect.height));
  drawRects();
}

function drawRect(rect, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
}

function drawRects() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const r of state.rects) {
    drawRect(r, "#e12f2f");
  }
  if (state.currentRect) {
    drawRect(state.currentRect, "#2254da");
  }
}

function mousePos(evt) {
  const rect = canvas.getBoundingClientRect();
  return { x: evt.clientX - rect.left, y: evt.clientY - rect.top };
}

canvas.addEventListener("mousedown", (evt) => {
  if (!state.jobId) {
    return;
  }
  state.drawing = true;
  const p = mousePos(evt);
  state.startX = p.x;
  state.startY = p.y;
  state.currentRect = { x: p.x, y: p.y, w: 0, h: 0 };
  drawRects();
});

canvas.addEventListener("mousemove", (evt) => {
  if (!state.drawing) {
    return;
  }
  const p = mousePos(evt);
  const x = Math.min(state.startX, p.x);
  const y = Math.min(state.startY, p.y);
  const w = Math.abs(p.x - state.startX);
  const h = Math.abs(p.y - state.startY);
  state.currentRect = { x, y, w, h };
  drawRects();
});

canvas.addEventListener("mouseup", () => {
  if (!state.drawing) {
    return;
  }
  state.drawing = false;
  if (state.currentRect && state.currentRect.w > 5 && state.currentRect.h > 5) {
    state.rects.push(state.currentRect);
  }
  state.currentRect = null;
  drawRects();
});

canvas.addEventListener("mouseleave", () => {
  if (state.drawing) {
    state.drawing = false;
    state.currentRect = null;
    drawRects();
  }
});

window.addEventListener("resize", () => {
  if (inputImage.src) {
    syncCanvasSize();
  }
});

async function apiJson(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return body;
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(async () => {
    if (!state.jobId) return;
    const job = await apiJson(`/api/jobs/${state.jobId}`).catch(() => null);
    if (!job) return;
    setStatus(job);
    if (job.auto_mask_url) {
      autoMaskImage.src = `${job.auto_mask_url}?t=${Date.now()}`;
    }
    if (job.output_url) {
      outputImage.src = `${job.output_url}?t=${Date.now()}`;
      downloadLink.href = job.output_url;
      downloadLink.classList.remove("hidden");
    }
    if (job.status === "completed" || job.status === "failed") {
      stopPolling();
    }
  }, 1200);
}

function buildManualMaskBlob() {
  if (!inputImage.naturalWidth || !inputImage.naturalHeight) {
    throw new Error("input image not ready");
  }
  if (!state.rects.length) {
    throw new Error("no manual rectangle drawn");
  }
  const off = document.createElement("canvas");
  off.width = inputImage.naturalWidth;
  off.height = inputImage.naturalHeight;
  const offCtx = off.getContext("2d");
  offCtx.fillStyle = "black";
  offCtx.fillRect(0, 0, off.width, off.height);
  offCtx.fillStyle = "white";

  const sx = inputImage.naturalWidth / canvas.width;
  const sy = inputImage.naturalHeight / canvas.height;
  for (const r of state.rects) {
    offCtx.fillRect(
      Math.round(r.x * sx),
      Math.round(r.y * sy),
      Math.round(r.w * sx),
      Math.round(r.h * sy),
    );
  }
  return new Promise((resolve) => off.toBlob(resolve, "image/png"));
}

uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files || !fileInput.files.length) {
    alert("Select an image first.");
    return;
  }
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  const job = await apiJson("/api/jobs", { method: "POST", body: fd });
  state.jobId = job.id;
  state.rects = [];
  setStatus(job);
  setButtonsEnabled(true);
  inputImage.src = `${job.input_url}?t=${Date.now()}`;
  autoMaskImage.removeAttribute("src");
  outputImage.removeAttribute("src");
  downloadLink.classList.add("hidden");
  await new Promise((resolve) => {
    inputImage.onload = () => resolve();
  });
  syncCanvasSize();
});

autoMaskBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  const job = await apiJson(`/api/jobs/${state.jobId}/auto-mask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  setStatus(job);
  if (job.auto_mask_url) {
    autoMaskImage.src = `${job.auto_mask_url}?t=${Date.now()}`;
  }
});

clearRectsBtn.addEventListener("click", () => {
  state.rects = [];
  drawRects();
});

uploadManualMaskBtn.addEventListener("click", async () => {
  if (!state.jobId) return;
  const blob = await buildManualMaskBlob();
  const fd = new FormData();
  fd.append("file", blob, "manual_mask.png");
  const job = await apiJson(`/api/jobs/${state.jobId}/manual-mask`, { method: "POST", body: fd });
  setStatus(job);
});

async function runProcess(maskSource) {
  if (!state.jobId) return;
  if (maskSource === "manual") {
    const blob = await buildManualMaskBlob();
    const fd = new FormData();
    fd.append("file", blob, "manual_mask.png");
    await apiJson(`/api/jobs/${state.jobId}/manual-mask`, { method: "POST", body: fd });
  }
  const job = await apiJson(`/api/jobs/${state.jobId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mask_source: maskSource,
      steps: 20,
      guidance_scale: 7.0,
      seed: 42,
    }),
  });
  setStatus(job);
  startPolling();
}

runAutoBtn.addEventListener("click", async () => {
  await runProcess("auto").catch((err) => alert(err.message));
});

runManualBtn.addEventListener("click", async () => {
  await runProcess("manual").catch((err) => alert(err.message));
});

setButtonsEnabled(false);

