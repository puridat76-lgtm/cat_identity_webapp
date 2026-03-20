const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const fileName = document.getElementById('fileName');
const predictBtn = document.getElementById('predictBtn');
const rebuildBtn = document.getElementById('rebuildBtn');
const refreshBtn = document.getElementById('refreshBtn');
const statusBadge = document.getElementById('statusBadge');
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

let currentFile = null;
let capturedDataUrl = null;
let stream = null;

async function refreshStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  if (!data.gallery_ready) {
    statusBadge.textContent = 'Gallery empty';
  } else {
    statusBadge.textContent = `${data.known_labels.length} labels / ${data.gallery_images} images`;
  }
}

function setPreviewFromDataUrl(dataUrl, label) {
  preview.src = dataUrl;
  fileName.textContent = label;
  predictBtn.disabled = false;
}

fileInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) return;
  currentFile = file;
  capturedDataUrl = null;
  const reader = new FileReader();
  reader.onload = () => setPreviewFromDataUrl(reader.result, file.name);
  reader.readAsDataURL(file);
});

predictBtn.addEventListener('click', async () => {
  predictBtn.disabled = true;
  resultState.classList.remove('hidden');
  resultBox.classList.add('hidden');
  resultState.textContent = 'Predicting...';

  let response;
  if (currentFile) {
    const formData = new FormData();
    formData.append('file', currentFile);
    response = await fetch('/api/predict', { method: 'POST', body: formData });
  } else if (capturedDataUrl) {
    response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: capturedDataUrl }),
    });
  } else {
    resultState.textContent = 'Please choose an image first.';
    predictBtn.disabled = false;
    return;
  }

  const data = await response.json();
  predictBtn.disabled = false;

  if (!response.ok) {
    resultState.textContent = data.message || 'Prediction failed.';
    return;
  }

  renderResult(data);
});

function renderResult(data) {
  resultState.classList.add('hidden');
  resultBox.classList.remove('hidden');
  finalLabel.textContent = data.final_label;
  scorePill.textContent = `score: ${data.best_known_score}`;
  bestKnown.textContent = `${data.best_known_name || '-'} (${data.best_known_score})`;
  secondScore.textContent = data.second_known_score;
  unknownScore.textContent = data.best_unknown_score ?? '-';
  notCatScore.textContent = data.best_not_cat_score ?? '-';
  blurScore.textContent = data.blur_score;
  brightness.textContent = data.brightness;
  qualityNotes.textContent = data.quality_pass ? 'Quality passed' : (data.quality_reasons.join(', ') || 'Failed');

  topMatches.innerHTML = '';
  (data.top_matches || []).forEach((row) => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${row.label}</span><strong>${row.score}</strong>`;
    topMatches.appendChild(li);
  });
  if (!data.top_matches || data.top_matches.length === 0) {
    const li = document.createElement('li');
    li.innerHTML = '<span>No known matches</span><strong>-</strong>';
    topMatches.appendChild(li);
  }
}

rebuildBtn.addEventListener('click', async () => {
  rebuildBtn.disabled = true;
  const response = await fetch('/api/rebuild', { method: 'POST' });
  const data = await response.json();
  rebuildBtn.disabled = false;
  await refreshStatus();
  alert(`Gallery rebuilt\nLabels: ${data.known_labels.length}\nGallery images: ${data.gallery_images}`);
});

refreshBtn.addEventListener('click', refreshStatus);

startCameraBtn.addEventListener('click', async () => {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
});

captureBtn.addEventListener('click', () => {
  if (!stream) return;
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  cameraCanvas.width = width;
  cameraCanvas.height = height;
  const ctx = cameraCanvas.getContext('2d');
  ctx.drawImage(video, 0, 0, width, height);
  capturedDataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);
  currentFile = null;
  setPreviewFromDataUrl(capturedDataUrl, 'camera_capture.jpg');
});

document.querySelectorAll('.tab').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
    button.classList.add('active');
    document.getElementById(`tab-${button.dataset.tab}`).classList.add('active');
  });
});

refreshStatus();
