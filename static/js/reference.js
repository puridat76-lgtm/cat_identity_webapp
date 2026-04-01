const {
  escapeHtml,
  readJson,
  refreshSummary,
  setMessage,
  updateGlobalStatus,
} = window.CatIdentity;

const referencePage = document.getElementById('referencePage');
const referenceKey = referencePage.dataset.referenceKey;
const referenceInput = document.getElementById('referenceInput');
const referenceCount = document.getElementById('referenceCount');
const referenceIndexState = document.getElementById('referenceIndexState');
const referenceVisibleCount = document.getElementById('referenceVisibleCount');
const referenceLoadMoreBtn = document.getElementById('referenceLoadMoreBtn');
const referenceGrid = document.getElementById('referenceGrid');
const referenceEmpty = document.getElementById('referenceEmpty');
const referenceMessage = document.getElementById('referenceMessage');

const state = {
  limit: 24,
  step: 128,
};

function renderReference(referenceSet, summary) {
  referenceCount.textContent = referenceSet.image_count ?? 0;
  referenceIndexState.textContent = summary?.index_status === 'needs_train' ? 'ต้อง Train ใหม่' : 'พร้อมใช้งาน';

  const images = referenceSet.images || [];
  const totalImages = referenceSet.image_count ?? images.length;
  const hiddenCount = referenceSet.hidden_count ?? 0;
  referenceVisibleCount.textContent = hiddenCount ? `${images.length} / ${totalImages} รูป` : `${totalImages} รูป`;
  referenceLoadMoreBtn.textContent = `+${Math.min(hiddenCount, state.step)} เพิ่มเติม`;
  referenceLoadMoreBtn.classList.toggle('hidden', hiddenCount === 0);
  referenceEmpty.classList.toggle('hidden', images.length !== 0);
  referenceGrid.innerHTML = images
    .map(
      (image) => `
        <article class="thumb-card large">
          <img src="${image.url}" alt="${escapeHtml(image.name)}" />
          <button type="button" class="thumb-delete" data-image-name="${escapeHtml(image.name)}">ลบ</button>
        </article>
      `
    )
    .join('');
}

async function loadReference(limit = state.limit) {
  state.limit = limit;
  const response = await fetch(`/api/reference-sets/${referenceKey}?limit=${state.limit}`);
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(referenceMessage, data.message || 'โหลดข้อมูลไม่สำเร็จ', 'error');
    return;
  }

  renderReference(data.reference_set, data.summary);
  updateGlobalStatus(data.summary);
}

async function uploadImages(files) {
  if (!files.length) {
    return;
  }

  const formData = new FormData();
  [...files].forEach((file) => formData.append('images', file));

  const response = await fetch(`/api/reference-sets/${referenceKey}/images`, { method: 'POST', body: formData });
  const data = await readJson(response);
  referenceInput.value = '';

  if (!response.ok) {
    setMessage(referenceMessage, data.message || 'เพิ่มรูปไม่สำเร็จ', 'error');
    return;
  }

  await loadReference(state.limit);
  setMessage(referenceMessage, 'เพิ่มรูปแล้ว', 'success');
}

async function deleteImage(imageName) {
  if (!window.confirm('ลบรูปนี้?')) {
    return;
  }

  const response = await fetch(`/api/reference-sets/${referenceKey}/images/${encodeURIComponent(imageName)}`, { method: 'DELETE' });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(referenceMessage, data.message || 'ลบรูปไม่สำเร็จ', 'error');
    return;
  }

  await loadReference(state.limit);
  setMessage(referenceMessage, 'ลบรูปแล้ว', 'success');
}

function loadMoreImages() {
  loadReference(state.limit + state.step);
}

referenceInput.addEventListener('change', () => {
  uploadImages(referenceInput.files);
});

referenceLoadMoreBtn.addEventListener('click', loadMoreImages);

referenceGrid.addEventListener('click', (event) => {
  const deleteButton = event.target.closest('[data-image-name]');
  if (deleteButton) {
    deleteImage(deleteButton.dataset.imageName);
  }
});

Promise.all([loadReference(), refreshSummary()]);
