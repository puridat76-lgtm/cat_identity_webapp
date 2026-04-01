const {
  escapeHtml,
  readJson,
  refreshSummary,
  setMessage,
  updateGlobalStatus,
} = window.CatIdentity;

const catList = document.getElementById('catList');
const catsEmpty = document.getElementById('catsEmpty');
const catSummaryText = document.getElementById('catSummaryText');
const searchInput = document.getElementById('searchInput');
const newCatBtn = document.getElementById('newCatBtn');
const catForm = document.getElementById('catForm');
const catIdInput = document.getElementById('catIdInput');
const formModeLabel = document.getElementById('formModeLabel');
const formTitle = document.getElementById('formTitle');
const nameInput = document.getElementById('nameInput');
const ownerInput = document.getElementById('ownerInput');
const locationInput = document.getElementById('locationInput');
const imagesInput = document.getElementById('imagesInput');
const imagesHint = document.getElementById('imagesHint');
const existingImageCount = document.getElementById('existingImageCount');
const imageGrid = document.getElementById('imageGrid');
const formMessage = document.getElementById('formMessage');
const deleteCatBtn = document.getElementById('deleteCatBtn');
const resetFormBtn = document.getElementById('resetFormBtn');

const state = {
  cats: [],
  selectedCatId: null,
  search: '',
};

function selectedCat() {
  return state.cats.find((cat) => cat.id === state.selectedCatId) || null;
}

function renderImageGrid(images) {
  existingImageCount.textContent = `${images.length}`;
  if (!images.length) {
    imageGrid.innerHTML = '<div class="empty-inline">ยังไม่มีรูป</div>';
    return;
  }

  imageGrid.innerHTML = images
    .map(
      (image) => `
        <article class="thumb-card">
          <img src="${image.url}" alt="${escapeHtml(image.name)}" />
          <button type="button" class="thumb-delete" data-image-name="${escapeHtml(image.name)}">ลบ</button>
        </article>
      `
    )
    .join('');
}

function resetForm() {
  state.selectedCatId = null;
  catIdInput.value = '';
  nameInput.value = '';
  ownerInput.value = '';
  locationInput.value = '';
  imagesInput.value = '';
  imagesHint.textContent = '0 ไฟล์';
  formModeLabel.textContent = 'เพิ่ม';
  formTitle.textContent = 'รายการใหม่';
  deleteCatBtn.classList.add('hidden');
  renderImageGrid([]);
  setMessage(formMessage, '');
  renderCats();
}

function fillForm(cat) {
  state.selectedCatId = cat.id;
  catIdInput.value = cat.id;
  nameInput.value = cat.name;
  ownerInput.value = cat.owner || '';
  locationInput.value = cat.location || '';
  imagesInput.value = '';
  imagesHint.textContent = '0 ไฟล์';
  formModeLabel.textContent = 'แก้ไข';
  formTitle.textContent = cat.name;
  deleteCatBtn.classList.remove('hidden');
  renderImageGrid(cat.images || []);
  setMessage(formMessage, '');
  renderCats();
}

function renderCats() {
  const filtered = state.cats.filter((cat) => {
    if (!state.search) {
      return true;
    }
    const text = [cat.name, cat.owner, cat.location].join(' ').toLowerCase();
    return text.includes(state.search);
  });

  catSummaryText.textContent = `${state.cats.length} ตัว`;
  catsEmpty.classList.toggle('hidden', filtered.length !== 0);
  catList.innerHTML = filtered
    .map((cat) => {
      const selected = cat.id === state.selectedCatId ? 'selected' : '';
      const meta = [cat.owner, cat.location].filter(Boolean).join(' • ') || 'ยังไม่ระบุ';
      const cover = cat.cover_image
        ? `<img src="${cat.cover_image}" alt="${escapeHtml(cat.name)}" />`
        : `<div class="thumb-fallback">${escapeHtml(cat.name.slice(0, 1).toUpperCase())}</div>`;

      return `
        <article class="list-item ${selected}">
          <button type="button" class="item-main" data-select-cat="${cat.id}">
            <div class="item-cover">${cover}</div>
            <div class="item-copy">
              <strong>${escapeHtml(cat.name)}</strong>
              <span>${escapeHtml(meta)}</span>
            </div>
          </button>
          <div class="item-side">
            <span class="count-pill">${cat.image_count} ภาพ</span>
            <div class="inline-actions">
              <button type="button" class="text-btn" data-select-cat="${cat.id}">แก้ไข</button>
              <button type="button" class="text-btn danger" data-delete-cat="${cat.id}">ลบ</button>
            </div>
          </div>
        </article>
      `;
    })
    .join('');
}

async function loadCats({ selectFirst = false } = {}) {
  const response = await fetch('/api/cats');
  const data = await readJson(response);
  if (!response.ok) {
    return;
  }

  state.cats = data.cats || [];
  updateGlobalStatus(data.summary);

  if (!state.cats.some((cat) => cat.id === state.selectedCatId)) {
    state.selectedCatId = selectFirst && state.cats.length ? state.cats[0].id : null;
  }

  renderCats();
  const current = selectedCat();
  if (current) {
    fillForm(current);
  } else {
    resetForm();
  }
}

async function saveCat(event) {
  event.preventDefault();
  setMessage(formMessage, '');

  const formData = new FormData();
  formData.append('name', nameInput.value.trim());
  formData.append('owner', ownerInput.value.trim());
  formData.append('location', locationInput.value.trim());
  [...imagesInput.files].forEach((file) => formData.append('images', file));

  const isEdit = Boolean(state.selectedCatId);
  const url = isEdit ? `/api/cats/${state.selectedCatId}` : '/api/cats';
  const method = isEdit ? 'PUT' : 'POST';

  const response = await fetch(url, { method, body: formData });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(formMessage, data.message || 'บันทึกไม่สำเร็จ', 'error');
    return;
  }

  state.selectedCatId = data.cat.id;
  await Promise.all([loadCats(), refreshSummary()]);
  const current = selectedCat();
  if (current) {
    fillForm(current);
  }
  setMessage(formMessage, 'บันทึกแล้ว', 'success');
}

async function deleteCat(catId) {
  const cat = state.cats.find((row) => row.id === catId);
  if (!window.confirm(`ลบ ${cat?.name || 'รายการนี้'} ?`)) {
    return;
  }

  const response = await fetch(`/api/cats/${catId}`, { method: 'DELETE' });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(formMessage, data.message || 'ลบไม่สำเร็จ', 'error');
    return;
  }

  if (state.selectedCatId === catId) {
    state.selectedCatId = null;
  }
  await Promise.all([loadCats({ selectFirst: true }), refreshSummary()]);
  setMessage(formMessage, 'ลบแล้ว', 'success');
}

async function deleteImage(imageName) {
  if (!state.selectedCatId || !window.confirm('ลบรูปนี้?')) {
    return;
  }

  const response = await fetch(`/api/cats/${state.selectedCatId}/images/${encodeURIComponent(imageName)}`, { method: 'DELETE' });
  const data = await readJson(response);
  if (!response.ok) {
    setMessage(formMessage, data.message || 'ลบรูปไม่สำเร็จ', 'error');
    return;
  }

  await Promise.all([loadCats(), refreshSummary()]);
  const current = selectedCat();
  if (current) {
    fillForm(current);
  }
  setMessage(formMessage, 'ลบรูปแล้ว', 'success');
}

imagesInput.addEventListener('change', () => {
  imagesHint.textContent = `${imagesInput.files.length} ไฟล์`;
});

searchInput.addEventListener('input', (event) => {
  state.search = event.target.value.trim().toLowerCase();
  renderCats();
});

newCatBtn.addEventListener('click', resetForm);
resetFormBtn.addEventListener('click', resetForm);
catForm.addEventListener('submit', saveCat);

catList.addEventListener('click', (event) => {
  const selectButton = event.target.closest('[data-select-cat]');
  if (selectButton) {
    const cat = state.cats.find((row) => row.id === Number(selectButton.dataset.selectCat));
    if (cat) {
      fillForm(cat);
    }
    return;
  }

  const deleteButton = event.target.closest('[data-delete-cat]');
  if (deleteButton) {
    deleteCat(Number(deleteButton.dataset.deleteCat));
  }
});

imageGrid.addEventListener('click', (event) => {
  const deleteButton = event.target.closest('[data-image-name]');
  if (deleteButton) {
    deleteImage(deleteButton.dataset.imageName);
  }
});

deleteCatBtn.addEventListener('click', () => {
  if (state.selectedCatId) {
    deleteCat(state.selectedCatId);
  }
});

resetForm();
loadCats({ selectFirst: true });
