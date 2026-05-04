/*
   NeuroScan v2 — app.js
   Features: Patient form, Analyze, Save History, Search,
             Export CSV, Modal detail view, Delete
 */

const API_URL = ""; // IMPORTANT: use same domain (Render deployment)

// ── Class config
const CLASS_CONFIG = {
glioma: { color: "#dc2626", bg: "#fef2f2", bar: "#ef4444",
desc: "Glioma is a tumor that originates in the glial cells of the brain." },
meningioma: { color: "#ea580c", bg: "#fff7ed", bar: "#f97316",
desc: "Meningioma arises from the meninges." },
notumor: { color: "#16a34a", bg: "#f0fdf4", bar: "#22c55e",
desc: "No tumor detected in this scan." },
pituitary: { color: "#2563eb", bg: "#eff6ff", bar: "#3b82f6",
desc: "Pituitary tumor affects the pituitary gland." },
};
// ── DOM refs 
const uploadZone    = document.getElementById("uploadZone");
const fileInput     = document.getElementById("fileInput");
const uploadInner   = document.getElementById("uploadInner");
const previewImg    = document.getElementById("previewImg");
const fileInfo      = document.getElementById("fileInfo");
const fileName      = document.getElementById("fileName");
const clearBtn      = document.getElementById("clearBtn");
const analyzeBtn    = document.getElementById("analyzeBtn");

const stateIdle     = document.getElementById("stateIdle");
const stateLoading  = document.getElementById("stateLoading");
const stateOutput   = document.getElementById("stateOutput");
const stateError    = document.getElementById("stateError");

const loadingStep   = document.getElementById("loadingStep");
const diagName      = document.getElementById("diagName");
const diagDesc      = document.getElementById("diagDesc");
const diagnosisBox  = document.getElementById("diagnosisBox");
const confCircle    = document.getElementById("confCircle");
const probBars      = document.getElementById("probBars");
const resultInfoGrid= document.getElementById("resultInfoGrid");
const errorMsg      = document.getElementById("errorMsg");

const saveBtn       = document.getElementById("saveBtn");
const newBtn        = document.getElementById("newBtn");
const retryBtn      = document.getElementById("retryBtn");

const statusDot     = document.getElementById("statusDot");
const statusText    = document.getElementById("statusText");

const historyList   = document.getElementById("historyList");
const historyEmpty  = document.getElementById("historyEmpty");
const historyCount  = document.getElementById("historyCount");
const searchInput   = document.getElementById("searchInput");
const exportBtn     = document.getElementById("exportBtn");
const clearHistBtn  = document.getElementById("clearHistoryBtn");

const modalOverlay  = document.getElementById("modalOverlay");
const modalBody     = document.getElementById("modalBody");
const modalClose    = document.getElementById("modalClose");
const modalOkBtn    = document.getElementById("modalOkBtn");
const modalDeleteBtn= document.getElementById("modalDeleteBtn");

// ── State 
let currentFile     = null;
let lastResult      = null;   // last API result
let currentPage     = "scan"; // "scan" | "history"
let openRecordId    = null;

// Set today's date as default for scan date
document.getElementById("scanDate").valueAsDate = new Date();

// HEALTH CHECK
async function checkHealth() {
  try {
    const res = await fetch(`${API_URL}/health`);
    if (res.ok) {
      statusDot.className = "status-dot online";
      statusText.textContent = "Server Online";
    } else throw new Error();
  } catch {
    statusDot.className = "status-dot error";
    statusText.textContent = "Server Offline";
  }
}
checkHealth();
setInterval(checkHealth, 15000);


// TAB SWITCHING

function switchTab(tab) {
  currentPage = tab;
  document.getElementById("pageScan").classList.toggle("hidden", tab !== "scan");
  document.getElementById("pageHistory").classList.toggle("hidden", tab !== "history");
  document.getElementById("tabScan").classList.toggle("active", tab === "scan");
  document.getElementById("tabHistory").classList.toggle("active", tab === "history");
  if (tab === "history") renderHistory();
}


// FILE UPLOAD
uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault(); uploadZone.classList.remove("dragover");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handleFile(f);
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    uploadInner.classList.add("hidden");
  };
  reader.readAsDataURL(file);
  fileName.textContent = file.name;
  fileInfo.classList.remove("hidden");
  analyzeBtn.disabled = false;
  showState("idle");
  lastResult = null;
  saveBtn.disabled = false;
}

clearBtn.addEventListener("click", resetUpload);

function resetUpload() {
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewImg.classList.add("hidden");
  uploadInner.classList.remove("hidden");
  fileInfo.classList.add("hidden");
  analyzeBtn.disabled = true;
  showState("idle");
  lastResult = null;
}

// FORM HELPERS
function getPatientData() {
  return {
    name      : document.getElementById("patientName").value.trim(),
    age       : document.getElementById("patientAge").value.trim(),
    gender    : document.getElementById("patientGender").value,
    patientId : document.getElementById("patientId").value.trim(),
    scanDate  : document.getElementById("scanDate").value,
    refDoctor : document.getElementById("refDoctor").value.trim(),
    symptoms  : document.getElementById("symptoms").value.trim(),
    scanType  : document.getElementById("scanType").value,
    priority  : document.getElementById("priority").value,
  };
}

function validatePatient() {
  const d = getPatientData();
  if (!d.name)   { alert("Please enter patient name.");   return false; }
  if (!d.age)    { alert("Please enter patient age.");    return false; }
  if (!d.gender) { alert("Please select patient gender."); return false; }
  return true;
}

// ANALYZE
analyzeBtn.addEventListener("click", runAnalysis);
retryBtn.addEventListener("click",  runAnalysis);

async function runAnalysis() {
  if (!currentFile) return;
  if (!validatePatient()) return;

  showState("loading");
  analyzeBtn.disabled = true;
  saveBtn.disabled = true;

  const steps = [
    "Preprocessing image...",
    "Running CNN forward pass...",
    "Computing softmax probabilities...",
    "Generating diagnosis...",
  ];
  let si = 0;
  const stepTimer = setInterval(() => {
    si = (si + 1) % steps.length;
    loadingStep.textContent = steps[si];
  }, 800);

  try {
    const fd = new FormData();
    fd.append("file", currentFile);

    const res = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
    clearInterval(stepTimer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${res.status}`);
    }

    lastResult = await res.json();
    renderResult(lastResult);

  } catch (err) {
    clearInterval(stepTimer);
    errorMsg.textContent = err.message || "Could not connect to server.";
    showState("error");
    analyzeBtn.disabled = false;
  }
}

// RENDER RESULT
function renderResult(data) {
  const cls    = data.predicted_class;
  const conf   = (data.confidence * 100).toFixed(1);
  const probs  = data.probabilities;
  const cfg    = CLASS_CONFIG[cls] || {};
  const patient = getPatientData();

  // Diagnosis box
  diagName.textContent = cls.charAt(0).toUpperCase() + cls.slice(1);
  diagDesc.textContent = cfg.desc || "";
  diagnosisBox.style.borderColor  = cfg.color || "#e2e8f0";
  diagnosisBox.style.background   = cfg.bg    || "#f8fafc";
  confCircle.textContent          = `${conf}%`;
  confCircle.style.background     = cfg.color || "#2563eb";

  // Probability bars
  probBars.innerHTML = "";
  Object.entries(probs).sort((a,b) => b[1]-a[1]).forEach(([label, prob]) => {
    const pct  = (prob * 100).toFixed(1);
    const c    = CLASS_CONFIG[label]?.bar || "#94a3b8";
    const bold = label === cls ? "font-weight:700;color:var(--text-1)" : "";
    const row  = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-label" style="${bold}">${label}</span>
      <div class="prob-track">
        <div class="prob-fill" data-w="${pct}" style="background:${c}; opacity:${label===cls?1:0.5}"></div>
      </div>
      <span class="prob-pct">${pct}%</span>`;
    probBars.appendChild(row);
  });

  // Patient info chips
  resultInfoGrid.innerHTML = "";
  const chips = [
    ["Patient",  patient.name   || "—"],
    ["Age",      patient.age    || "—"],
    ["Gender",   patient.gender || "—"],
    ["Priority", patient.priority],
    ["Scan Type",patient.scanType || "—"],
    ["Date",     patient.scanDate || "—"],
  ];
  chips.forEach(([label, val]) => {
    const d = document.createElement("div");
    d.className = "info-chip";
    d.innerHTML = `<div class="info-chip-label">${label}</div><div class="info-chip-value">${val}</div>`;
    resultInfoGrid.appendChild(d);
  });

  showState("output");
  saveBtn.disabled = false;

  // Animate bars
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".prob-fill").forEach(el => { el.style.width = el.dataset.w + "%"; });
  }));
}

// STATE MANAGER
function showState(s) {
  stateIdle.classList.add("hidden");
  stateLoading.classList.add("hidden");
  stateOutput.classList.add("hidden");
  stateError.classList.add("hidden");
  ({ idle: stateIdle, loading: stateLoading, output: stateOutput, error: stateError }[s] || stateIdle)
    .classList.remove("hidden");
}

newBtn.addEventListener("click", () => {
  resetUpload();
  analyzeBtn.disabled = true;
});

// HISTORY — STORAGE (localStorage)
function loadHistory() {
  return JSON.parse(localStorage.getItem("neuro_history") || "[]");
}
function saveHistory(arr) {
  localStorage.setItem("neuro_history", JSON.stringify(arr));
}

// ── Save Record 
saveBtn.addEventListener("click", () => {
  if (!lastResult) return;
  const patient = getPatientData();
  if (!validatePatient()) return;

  const record = {
    id         : Date.now().toString(),
    savedAt    : new Date().toISOString(),
    patient,
    result     : lastResult,
  };

  const arr = loadHistory();
  arr.unshift(record);
  saveHistory(arr);

  saveBtn.disabled = true;
  saveBtn.textContent = "✅ Saved!";
  setTimeout(() => { saveBtn.textContent = "💾 Save to History"; }, 2000);
});

// RENDER HISTORY LIST
function renderHistory(filter = "") {
  const all     = loadHistory();
  const query   = filter.toLowerCase();
  const records = query
    ? all.filter(r =>
        r.patient.name.toLowerCase().includes(query) ||
        (r.patient.patientId || "").toLowerCase().includes(query) ||
        r.result.predicted_class.toLowerCase().includes(query)
      )
    : all;

  historyCount.textContent = `${all.length} record${all.length !== 1 ? "s" : ""} saved`;

  if (records.length === 0) {
    historyEmpty.classList.remove("hidden");
    historyList.innerHTML = "";
    return;
  }
  historyEmpty.classList.add("hidden");
  historyList.innerHTML = "";

  records.forEach(rec => {
    const cls   = rec.result.predicted_class;
    const conf  = (rec.result.confidence * 100).toFixed(1);
    const cfg   = CLASS_CONFIG[cls] || {};
    const date  = new Date(rec.savedAt).toLocaleDateString("en-IN", { day:"2-digit", month:"short", year:"numeric" });
    const initials = (rec.patient.name || "?").split(" ").map(w => w[0]).join("").toUpperCase().slice(0,2);

    const card = document.createElement("div");
    card.className = "record-card";
    card.innerHTML = `
      <div class="record-left">
        <div class="record-avatar" style="background:${cfg.color || '#64748b'}">${initials}</div>
        <div class="record-info">
          <div class="record-name">${rec.patient.name || "Unknown"}</div>
          <div class="record-meta">
            ${rec.patient.age ? `Age ${rec.patient.age}` : ""}
            ${rec.patient.gender ? ` · ${rec.patient.gender}` : ""}
            ${rec.patient.patientId ? ` · ID: ${rec.patient.patientId}` : ""}
          </div>
        </div>
      </div>
      <div class="record-right">
        <span class="record-badge badge-${cls}">${cls}</span>
        <span class="record-conf">${conf}%</span>
        <span class="record-priority priority-${rec.patient.priority}">${rec.patient.priority}</span>
        <span class="record-date">${date}</span>
      </div>`;
    card.addEventListener("click", () => openModal(rec));
    historyList.appendChild(card);
  });
}

// SEARCH
searchInput.addEventListener("input", () => renderHistory(searchInput.value));

// EXPORT CSV
exportBtn.addEventListener("click", () => {
  const arr = loadHistory();
  if (!arr.length) { alert("No records to export."); return; }

  const headers = [
    "Record ID", "Saved At", "Patient Name", "Age", "Gender",
    "Patient ID", "Scan Date", "Referring Doctor", "Symptoms",
    "Scan Type", "Priority", "Predicted Class", "Confidence (%)",
    "Glioma (%)", "Meningioma (%)", "No Tumor (%)", "Pituitary (%)"
  ];

  const rows = arr.map(r => {
    const p = r.patient;
    const res = r.result;
    const probs = res.probabilities;
    return [
      r.id,
      new Date(r.savedAt).toLocaleString("en-IN"),
      p.name, p.age, p.gender, p.patientId || "",
      p.scanDate || "", p.refDoctor || "", (p.symptoms || "").replace(/,/g, ";"),
      p.scanType || "", p.priority,
      res.predicted_class,
      (res.confidence * 100).toFixed(2),
      ((probs.glioma     || 0) * 100).toFixed(2),
      ((probs.meningioma || 0) * 100).toFixed(2),
      ((probs.notumor    || 0) * 100).toFixed(2),
      ((probs.pituitary  || 0) * 100).toFixed(2),
    ].map(v => `"${v}"`).join(",");
  });

  const csv  = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url;
  a.download = `neuroscan_history_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

// CLEAR ALL HISTORY
clearHistBtn.addEventListener("click", () => {
  if (!confirm("Delete all history records? This cannot be undone.")) return;
  saveHistory([]);
  renderHistory();
});

// MODAL
function openModal(rec) {
  openRecordId = rec.id;
  const cls    = rec.result.predicted_class;
  const conf   = (rec.result.confidence * 100).toFixed(1);
  const cfg    = CLASS_CONFIG[cls] || {};
  const p      = rec.patient;
  const probs  = rec.result.probabilities;

  modalBody.innerHTML = `
    <!-- Diagnosis -->
    <div>
      <div class="modal-section-title">Diagnosis</div>
      <div class="modal-diag-box" style="border-color:${cfg.color};background:${cfg.bg}">
        <div>
          <div style="font-size:0.65rem;color:var(--text-3);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px">Predicted Class</div>
          <div class="modal-diag-name" style="color:${cfg.color}">${cls}</div>
          <div style="font-size:0.75rem;color:var(--text-2);margin-top:4px;max-width:340px;line-height:1.5">${cfg.desc||""}</div>
        </div>
        <div class="modal-diag-conf" style="color:${cfg.color}">${conf}%</div>
      </div>
    </div>

    <!-- Patient Info -->
    <div>
      <div class="modal-section-title">Patient Information</div>
      <div class="modal-grid">
        <div class="modal-field"><div class="modal-field-label">Full Name</div><div class="modal-field-value">${p.name||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Age</div><div class="modal-field-value">${p.age||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Gender</div><div class="modal-field-value">${p.gender||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Patient ID</div><div class="modal-field-value">${p.patientId||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Scan Date</div><div class="modal-field-value">${p.scanDate||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Referring Doctor</div><div class="modal-field-value">${p.refDoctor||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Scan Type</div><div class="modal-field-value">${p.scanType||"—"}</div></div>
        <div class="modal-field"><div class="modal-field-label">Priority</div><div class="modal-field-value">${p.priority||"—"}</div></div>
        ${p.symptoms ? `<div class="modal-field full"><div class="modal-field-label">Symptoms / Notes</div><div class="modal-field-value">${p.symptoms}</div></div>` : ""}
      </div>
    </div>

    <!-- Probabilities -->
    <div>
      <div class="modal-section-title">Class Probabilities</div>
      <div style="display:flex;flex-direction:column;gap:8px">
        ${Object.entries(probs).sort((a,b)=>b[1]-a[1]).map(([label, prob]) => `
          <div class="prob-row">
            <span class="prob-label" style="${label===cls?"font-weight:700;color:var(--text-1)":""}">${label}</span>
            <div class="prob-track" style="height:8px">
              <div class="prob-fill" style="width:${(prob*100).toFixed(1)}%;background:${CLASS_CONFIG[label]?.bar||'#94a3b8'};opacity:${label===cls?1:0.5};height:100%;border-radius:99px"></div>
            </div>
            <span class="prob-pct">${(prob*100).toFixed(1)}%</span>
          </div>`).join("")}
      </div>
    </div>

    <!-- Saved timestamp -->
    <div style="font-size:0.7rem;color:var(--text-3);font-family:var(--mono);text-align:right">
      Saved: ${new Date(rec.savedAt).toLocaleString("en-IN")}
    </div>
  `;

  modalOverlay.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  modalOverlay.classList.add("hidden");
  document.body.style.overflow = "";
  openRecordId = null;
}

modalClose.addEventListener("click", closeModal);
modalOkBtn.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", e => { if (e.target === modalOverlay) closeModal(); });

modalDeleteBtn.addEventListener("click", () => {
  if (!openRecordId) return;
  if (!confirm("Delete this record?")) return;
  const arr = loadHistory().filter(r => r.id !== openRecordId);
  saveHistory(arr);
  closeModal();
  renderHistory(searchInput.value);
});

// INIT
showState("idle");