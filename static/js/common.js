(function () {
  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch {
      return {};
    }
  }

  const indexStatusMap = {
    ready: 'พร้อมใช้งาน',
    needs_train: 'ต้อง Train ใหม่',
    empty: 'ยังไม่มี index',
  };

  function updateGlobalStatus(summary) {
    const badge = document.getElementById('globalStatusBadge');
    if (!badge || !summary) {
      return;
    }

    const statusText = indexStatusMap[summary.index_status] || '-';
    if (summary.index_status === 'needs_train') {
      badge.textContent = `${statusText} · ${summary.actual_total_images || 0}/${summary.indexed_total_images || 0}`;
      return;
    }

    badge.textContent = `${statusText} · ${summary.indexed_total_images || 0}`;
  }

  async function refreshSummary() {
    const response = await fetch('/api/status');
    const data = await readJson(response);
    if (response.ok) {
      updateGlobalStatus(data);
    }
    return { response, data };
  }

  function setMessage(element, message, tone = 'neutral') {
    if (!element) {
      return;
    }

    if (!message) {
      element.textContent = '';
      element.className = 'message hidden';
      return;
    }

    element.textContent = message;
    element.className = `message ${tone}`;
  }

  window.CatIdentity = {
    escapeHtml,
    readJson,
    refreshSummary,
    updateGlobalStatus,
    setMessage,
  };

  document.addEventListener('DOMContentLoaded', () => {
    refreshSummary();
  });
})();
