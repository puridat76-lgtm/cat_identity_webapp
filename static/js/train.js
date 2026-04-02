const {
  readJson,
  refreshSummary,
  setMessage,
  updateGlobalStatus,
} = window.CatIdentity;

const trainActionBtn = document.getElementById('trainActionBtn');
const trainRefreshBtn = document.getElementById('trainRefreshBtn');
const trainStatusText = document.getElementById('trainStatusText');
const trainStageText = document.getElementById('trainStageText');
const trainProcessedText = document.getElementById('trainProcessedText');
const trainValidText = document.getElementById('trainValidText');
const trainElapsedText = document.getElementById('trainElapsedText');
const trainCurrentText = document.getElementById('trainCurrentText');
const trainProgressFill = document.getElementById('trainProgressFill');
const trainPercentText = document.getElementById('trainPercentText');
const trainLinePath = document.getElementById('trainLinePath');
const trainAreaPath = document.getElementById('trainAreaPath');
const trainLineDot = document.getElementById('trainLineDot');
const trainChartMaxLabel = document.getElementById('trainChartMaxLabel');
const trainSplitSummary = document.getElementById('trainSplitSummary');
const trainGalleryBar = document.getElementById('trainGalleryBar');
const trainNotCatBar = document.getElementById('trainNotCatBar');
const trainUnknownBar = document.getElementById('trainUnknownBar');
const trainGalleryText = document.getElementById('trainGalleryText');
const trainNotCatText = document.getElementById('trainNotCatText');
const trainUnknownText = document.getElementById('trainUnknownText');
const trainTimeline = document.getElementById('trainTimeline');
const trainMessage = document.getElementById('trainMessage');

const state = {
  poller: null,
  job: null,
};

function formatStage(stage) {
  const stageMap = {
    idle: 'รอเริ่ม',
    prepare: 'กำลังเตรียมรายการรูป',
    sync: 'กำลังซิงก์ไฟล์ที่เปลี่ยน',
    extract: 'กำลังสร้างเวกเตอร์',
    remove: 'กำลังล้างรายการที่ถูกลบ',
    save: 'กำลังบันทึก index',
    completed: 'เสร็จแล้ว',
    failed: 'เกิดข้อผิดพลาด',
  };
  return stageMap[stage] || stage || '-';
}

function setButtons(isRunning) {
  [trainActionBtn].forEach((button) => {
    button.disabled = isRunning;
    button.textContent = isRunning ? 'กำลัง Train...' : 'Train';
  });
}

function clearPoller() {
  if (!state.poller) {
    return;
  }
  window.clearTimeout(state.poller);
  state.poller = null;
}

function renderChart(job) {
  const history = job?.history || [];
  if (!history.length) {
    trainLinePath.setAttribute('points', '');
    trainAreaPath.setAttribute('d', '');
    trainLineDot.style.opacity = '0';
    trainChartMaxLabel.textContent = '0 images';
    return;
  }

  const width = 600;
  const height = 240;
  const paddingX = 22;
  const paddingY = 20;
  const maxX = Math.max(history.at(-1).elapsed_seconds || 0.1, 0.1);
  const maxY = Math.max(job.total_images || 1, 1);
  const plotWidth = width - paddingX * 2;
  const plotHeight = height - paddingY * 2;

  const points = history.map((row) => {
    const x = paddingX + ((row.elapsed_seconds || 0) / maxX) * plotWidth;
    const y = height - paddingY - ((row.processed_images || 0) / maxY) * plotHeight;
    return { x, y };
  });

  const polyline = points.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');
  const first = points[0];
  const last = points.at(-1);
  const area = [
    `M ${first.x.toFixed(2)} ${height - paddingY}`,
    ...points.map((point) => `L ${point.x.toFixed(2)} ${point.y.toFixed(2)}`),
    `L ${last.x.toFixed(2)} ${height - paddingY}`,
    'Z',
  ].join(' ');

  trainLinePath.setAttribute('points', polyline);
  trainAreaPath.setAttribute('d', area);
  trainLineDot.setAttribute('cx', `${last.x.toFixed(2)}`);
  trainLineDot.setAttribute('cy', `${last.y.toFixed(2)}`);
  trainLineDot.style.opacity = '1';
  trainChartMaxLabel.textContent = `${job.processed_images || 0} / ${job.total_images || 0} images`;
}

function renderTimeline(job) {
  const rows = (job?.history || []).slice(-6).reverse();
  if (!rows.length) {
    trainTimeline.innerHTML = '<div class="timeline-card">ยังไม่มีงาน Train</div>';
    return;
  }

  trainTimeline.innerHTML = rows
    .map(
      (row) => `
        <article class="timeline-card">
          <strong>${row.processed_images}/${job.total_images}</strong>
          <span>${row.elapsed_seconds}s</span>
          <span>valid ${row.valid_images}</span>
        </article>
      `
    )
    .join('');
}

function renderJob(job) {
  const current = job || {
    status: 'idle',
    stage: 'idle',
    progress: 0,
    total_images: 0,
    processed_images: 0,
    valid_images: 0,
    elapsed_seconds: 0,
    current_label: null,
    current_image: null,
    split_processed: { gallery: 0, not_cat: 0, unknown_cat: 0 },
    split_totals: { gallery: 0, not_cat: 0, unknown_cat: 0 },
    history: [],
  };

  state.job = current;
  const isRunning = current.status === 'running';
  const percent = Math.round((current.progress || 0) * 100);
  const splitProcessed = current.split_processed || { gallery: 0, not_cat: 0, unknown_cat: 0 };
  const splitTotals = current.split_totals || { gallery: 0, not_cat: 0, unknown_cat: 0 };
  const maxSplit = Math.max(splitTotals.gallery || 0, splitTotals.not_cat || 0, splitTotals.unknown_cat || 0, 1);

  setButtons(isRunning);
  trainStatusText.textContent = current.status || 'idle';
  trainStageText.textContent = current.error || formatStage(current.stage);
  trainProcessedText.textContent = `${current.processed_images || 0} / ${current.total_images || 0}`;
  trainValidText.textContent = `valid ${current.valid_images || 0}`;
  trainElapsedText.textContent = `${(current.elapsed_seconds || 0).toFixed(1)}s`;
  trainCurrentText.textContent = current.current_label ? `${current.current_label} · ${current.current_image}` : '-';
  trainPercentText.textContent = `${percent}%`;
  trainProgressFill.style.width = `${percent}%`;
  trainSplitSummary.textContent = `${splitProcessed.gallery || 0} / ${splitProcessed.not_cat || 0} / ${splitProcessed.unknown_cat || 0}`;
  trainGalleryText.textContent = `${splitProcessed.gallery || 0}`;
  trainNotCatText.textContent = `${splitProcessed.not_cat || 0}`;
  trainUnknownText.textContent = `${splitProcessed.unknown_cat || 0}`;
  trainGalleryBar.style.width = `${((splitProcessed.gallery || 0) / maxSplit) * 100}%`;
  trainNotCatBar.style.width = `${((splitProcessed.not_cat || 0) / maxSplit) * 100}%`;
  trainUnknownBar.style.width = `${((splitProcessed.unknown_cat || 0) / maxSplit) * 100}%`;

  renderChart(current);
  renderTimeline(current);
}

async function refreshTrainStatus() {
  const previousStatus = state.job?.status;
  const response = await fetch('/api/train/status');
  const data = await readJson(response);
  if (!response.ok) {
    return;
  }

  renderJob(data.job);
  updateGlobalStatus(data.summary);

  if (previousStatus === 'running' && data.job.status === 'completed') {
    setMessage(trainMessage, 'Train แล้ว', 'success');
  } else if (previousStatus === 'running' && data.job.status === 'failed') {
    setMessage(trainMessage, data.job.error || 'Train ไม่สำเร็จ', 'error');
  }

  if (data.job.status === 'running') {
    clearPoller();
    state.poller = window.setTimeout(refreshTrainStatus, 700);
  } else {
    clearPoller();
  }
}

async function startTrain() {
  setButtons(true);
  setMessage(trainMessage, '');

  const response = await fetch('/api/train', { method: 'POST' });
  const data = await readJson(response);
  if (!response.ok) {
    setButtons(false);
    setMessage(trainMessage, data.message || 'Train ไม่สำเร็จ', 'error');
    return;
  }

  renderJob(data.job);
  updateGlobalStatus(data.summary);
  clearPoller();
  state.poller = window.setTimeout(refreshTrainStatus, 300);
}

trainActionBtn.addEventListener('click', startTrain);
trainRefreshBtn.addEventListener('click', refreshTrainStatus);

renderJob(null);
Promise.all([refreshTrainStatus(), refreshSummary()]);
