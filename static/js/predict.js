const { readJson, refreshSummary, escapeHtml } = window.CatIdentity;

const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const resultState = document.getElementById('resultState');
const resultBox = document.getElementById('resultBox');
const finalLabel = document.getElementById('finalLabel');
const scorePill = document.getElementById('scorePill');
const bestKnown = document.getElementById('bestKnown');
const secondScore = document.getElementById('secondScore');
const unknownScore = document.getElementById('unknownScore');
const notCatScore = document.getElementById('notCatScore');
const blurScore = document.getElementById('blurScore');
const brightness = document.getElementById('brightness');
const topMatches = document.getElementById('topMatches');
const qualityNotes = document.getElementById('qualityNotes');
const video = document.getElementById('video');
const cameraCanvas = document.getElementById('cameraCanvas');
const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');

const state = {
  currentFile: null,
  capturedDataUrl: null,
  stream: null,
};

const labelMap = {
  unknown: 'ไม่รู้จัก',
  not_cat: 'ไม่ใช่แมว',
  low_quality: 'ภาพไม่ชัด',
};

function formatLabel(label) {
  return labelMap[label] || label || '-';
}

function setPreview(dataUrl, label) {
  preview.src = dataUrl;
  fileName.textContent = label;
  predictBtn.disabled = false;
}

function setResultState(message) {
  resultState.textContent = message;
  resultState.classList.remove('hidden');
  resultBox.classList.add('hidden');
}

function renderResult(data) {
  resultState.classList.add('hidden');
  resultBox.classList.remove('hidden');
  finalLabel.textContent = formatLabel(data.final_label);
  scorePill.textContent = `score ${data.best_known_score ?? '-'}`;
  bestKnown.textContent = data.best_known_name || '-';
  secondScore.textContent = data.second_known_score ?? '-';
  unknownScore.textContent = data.best_unknown_score ?? '-';
  notCatScore.textContent = data.best_not_cat_score ?? '-';
  blurScore.textContent = data.blur_score ?? '-';
  brightness.textContent = data.brightness ?? '-';
  qualityNotes.textContent = data.quality_pass ? 'คุณภาพผ่าน' : (data.quality_reasons || []).join(', ') || 'คุณภาพไม่ผ่าน';

  const matches = data.top_matches || [];
  topMatches.innerHTML = matches.length
    ? matches.map((row) => `<li><span>${escapeHtml(row.label)}</span><strong>${row.score}</strong></li>`).join('')
    : '<li><span>No match</span><strong>-</strong></li>';
}

async function predict() {
  predictBtn.disabled = true;
  setResultState('กำลังทำนาย...');

  let response;
  if (state.currentFile) {
    const formData = new FormData();
    formData.append('file', state.currentFile);
    response = await fetch('/api/predict', { method: 'POST', body: formData });
  } else if (state.capturedDataUrl) {
    response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: state.capturedDataUrl }),
    });
  } else {
    setResultState('เลือกรูปก่อน');
    predictBtn.disabled = false;
    return;
  }

  const data = await readJson(response);
  predictBtn.disabled = false;

  if (!response.ok) {
    setResultState(data.message || 'ทำนายไม่สำเร็จ');
    return;
  }

  renderResult(data);
}

async function startCamera() {
  if (state.stream) {
    return;
  }
  state.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = state.stream;
}

function captureFrame() {
  if (!state.stream) {
    return;
  }

  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  cameraCanvas.width = width;
  cameraCanvas.height = height;
  const ctx = cameraCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, width, height);
  state.capturedDataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);
  state.currentFile = null;
  setPreview(state.capturedDataUrl, 'camera_capture.jpg');
}

fileInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  state.currentFile = file;
  state.capturedDataUrl = null;
  const reader = new FileReader();
  reader.onload = () => setPreview(reader.result, file.name);
  reader.readAsDataURL(file);
});

predictBtn.addEventListener('click', predict);
startCameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', captureFrame);

document.querySelectorAll('.tab-pill').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tab-pill').forEach((tab) => tab.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(`tab-${button.dataset.tab}`).classList.add('active');
  });
});

window.addEventListener('beforeunload', () => {
  if (!state.stream) {
    return;
  }
  state.stream.getTracks().forEach((track) => track.stop());
});

setResultState('พร้อม');
refreshSummary();
