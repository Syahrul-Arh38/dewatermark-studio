/**
 * studio.js - Dewatermark AI Studio Interactive Engine
 * Handles Canvas drawing, before/after split slider, AI auto-scan, inpainting, and batch queue.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const canvasStage = document.getElementById('canvasStage');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const splitContainer = document.getElementById('splitContainer');
  const layerBefore = document.getElementById('layerBefore');
  const layerAfter = document.getElementById('layerAfter');
  const splitDivider = document.getElementById('splitDivider');

  const canvasBase = document.getElementById('canvasBase');
  const canvasResult = document.getElementById('canvasResult');
  const canvasMask = document.getElementById('canvasMask');

  const ctxBase = canvasBase.getContext('2d');
  const ctxResult = canvasResult.getContext('2d');
  const ctxMask = canvasMask.getContext('2d');

  const brushCursor = document.getElementById('brushCursor');
  const processingOverlay = document.getElementById('processingOverlay');
  const processingText = document.getElementById('processingText');

  // Tools & Controls
  const toolAuto = document.getElementById('toolAuto');
  const toolBrush = document.getElementById('toolBrush');
  const toolEraser = document.getElementById('toolEraser');
  const brushSizeInput = document.getElementById('brushSize');
  const brushSizeVal = document.getElementById('brushSizeVal');
  const sensitivityInput = document.getElementById('sensitivity');
  const sensitivityVal = document.getElementById('sensitivityVal');
  const dilationInput = document.getElementById('dilationPx');
  const dilationVal = document.getElementById('dilationVal');
  const checkStamp = document.getElementById('checkStamp');
  const checkFeather = document.getElementById('checkFeather');
  const selectEngine = document.getElementById('selectEngine');
  const selectWmMode = document.getElementById('selectWmMode');
  const wmModeHint = document.getElementById('wmModeHint');

  if (selectWmMode && wmModeHint) {
    selectWmMode.addEventListener('change', () => {
      const m = selectWmMode.value;
      if (m === 'sakerly') {
        wmModeHint.textContent = 'Mode Sakerly: Deteksi tanda tangan coral-red di sudut foto (<10ms).';
      } else if (m === 'rfans') {
        wmModeHint.textContent = 'Mode RFans: Memindai stempel bulat 4-sudut & teks transparan miring.';
      } else {
        wmModeHint.textContent = 'Mode Auto: Memindai semua jenis watermark secara komprehensif.';
      }
    });
  }

  const btnAutoScan = document.getElementById('btnAutoScan');
  const btnInpaint = document.getElementById('btnInpaint');
  const btnDownload = document.getElementById('btnDownload');
  const btnLoadDemo = document.getElementById('btnLoadDemo');
  const btnClearMask = document.getElementById('btnClearMask');
  const btnZoomFit = document.getElementById('btnZoomFit');
  const btnResetImage = document.getElementById('btnResetImage');

  // Stats
  const statDim = document.getElementById('statDim');
  const statMasked = document.getElementById('statMasked');
  const statStamps = document.getElementById('statStamps');
  const statTime = document.getElementById('statTime');

  // Colab GPU Acceleration Controls
  const checkRemoteGpu = document.getElementById('checkRemoteGpu');
  const colabUrlGroup = document.getElementById('colabUrlGroup');
  const colabUrlInput = document.getElementById('colabUrlInput');
  const btnTestColab = document.getElementById('btnTestColab');
  const gpuStatusBadge = document.getElementById('gpuStatusBadge');
  let apiBaseUrl = '';
  let isUsingColab = false;

  // View modes
  const modeSplit = document.getElementById('modeSplit');
  const modeClean = document.getElementById('modeClean');
  const modeOriginal = document.getElementById('modeOriginal');
  const modeMask = document.getElementById('modeMask');

  // Navigation & Batch Controls
  const navStudio = document.getElementById('navStudio');
  const navBatch = document.getElementById('navBatch');
  const studioMain = document.querySelector('.studio-main');
  const batchView = document.getElementById('batchView');
  const batchDropzone = document.getElementById('batchDropzone');
  const batchFileInput = document.getElementById('batchFileInput');
  const batchTotalCountBadge = document.getElementById('batchTotalCountBadge');
  const batchControlsCard = document.getElementById('batchControlsCard');
  const batchWmModeSelect = document.getElementById('batchWmModeSelect');
  const batchEngineSelect = document.getElementById('batchEngineSelect');
  const btnAddMoreBatch = document.getElementById('btnAddMoreBatch');
  const btnClearBatchQueue = document.getElementById('btnClearBatchQueue');
  const btnStartBatch = document.getElementById('btnStartBatch');
  const batchGlobalProgressCard = document.getElementById('batchGlobalProgressCard');
  const batchGlobalStatusText = document.getElementById('batchGlobalStatusText');
  const batchGlobalTimer = document.getElementById('batchGlobalTimer');
  const batchGlobalProgressFill = document.getElementById('batchGlobalProgressFill');
  const batchListContainer = document.getElementById('batchListContainer');
  const batchListBody = document.getElementById('batchListBody');
  const btnBatchDownloadAll = document.getElementById('btnBatchDownloadAll');

  // State
  let currentImage = null; // Image object
  let originalBase64 = null;
  let cleanedBase64 = null;
  let activeTool = 'auto'; // 'auto', 'brush', 'eraser'
  let isDrawing = false;
  let isDraggingSplit = false;
  let splitPercent = 50;
  let viewMode = 'split'; // 'split', 'clean', 'original', 'mask'
  let currentBatchId = null;

  // Initialize
  updateBrushSize();

  // ----------------------------------------------------
  // Brush & Slider Value Bindings
  // ----------------------------------------------------
  brushSizeInput.addEventListener('input', updateBrushSize);
  function updateBrushSize() {
    const sz = brushSizeInput.value;
    brushSizeVal.textContent = `${sz}px`;
    brushCursor.style.width = `${sz}px`;
    brushCursor.style.height = `${sz}px`;
  }

  sensitivityInput.addEventListener('input', () => {
    sensitivityVal.textContent = parseFloat(sensitivityInput.value).toFixed(2);
  });

  dilationInput.addEventListener('input', () => {
    dilationVal.textContent = `${dilationInput.value}px`;
  });

  // Tool Switching
  function setActiveTool(tool) {
    activeTool = tool;
    toolAuto.classList.toggle('active', tool === 'auto');
    toolBrush.classList.toggle('active', tool === 'brush');
    toolEraser.classList.toggle('active', tool === 'eraser');

    if (tool === 'brush' || tool === 'eraser') {
      canvasMask.style.pointerEvents = 'auto';
      brushCursor.style.display = 'block';
    } else {
      brushCursor.style.display = 'none';
    }
  }

  toolAuto.addEventListener('click', () => setActiveTool('auto'));
  toolBrush.addEventListener('click', () => setActiveTool('brush'));
  toolEraser.addEventListener('click', () => setActiveTool('eraser'));

  // ----------------------------------------------------
  // Viewport Sizing & Canvas Scale Adjustment
  // ----------------------------------------------------
  function fitCanvasToViewport(w, h) {
    const container = document.getElementById('viewportContainer');
    if (!container) return;
    const maxW = Math.max(280, container.clientWidth - 48);
    const maxH = Math.max(280, container.clientHeight - 96);

    const scale = Math.min(maxW / w, maxH / h, 1.0);
    const displayW = Math.round(w * scale);
    const displayH = Math.round(h * scale);

    splitContainer.style.width = `${displayW}px`;
    splitContainer.style.height = `${displayH}px`;
    canvasWrapper.style.width = `${displayW}px`;
    canvasWrapper.style.height = `${displayH}px`;

    canvasBase.style.width = `${displayW}px`;
    canvasBase.style.height = `${displayH}px`;
    canvasResult.style.width = `${displayW}px`;
    canvasResult.style.height = `${displayH}px`;
    canvasMask.style.width = `${displayW}px`;
    canvasMask.style.height = `${displayH}px`;
  }

  window.addEventListener('resize', () => {
    if (currentImage) {
      fitCanvasToViewport(currentImage.naturalWidth, currentImage.naturalHeight);
    }
  });

  btnZoomFit.addEventListener('click', () => {
    if (currentImage) {
      fitCanvasToViewport(currentImage.naturalWidth, currentImage.naturalHeight);
    }
  });

  // ----------------------------------------------------
  // Colab GPU Acceleration Logic
  // ----------------------------------------------------
  checkRemoteGpu.addEventListener('change', () => {
    colabUrlGroup.style.display = checkRemoteGpu.checked ? 'block' : 'none';
    updateApiBaseUrl();
  });

  colabUrlInput.addEventListener('input', updateApiBaseUrl);

  function updateApiBaseUrl() {
    if (checkRemoteGpu.checked && colabUrlInput.value.trim()) {
      apiBaseUrl = colabUrlInput.value.trim().replace(/\/+$/, '');
      isUsingColab = true;
      gpuStatusBadge.textContent = 'Colab GPU';
      gpuStatusBadge.className = 'gpu-badge badge-active';
    } else {
      apiBaseUrl = '';
      isUsingColab = false;
      gpuStatusBadge.textContent = 'Local CPU';
      gpuStatusBadge.className = 'gpu-badge';
    }
  }

  btnTestColab.addEventListener('click', async () => {
    const url = colabUrlInput.value.trim().replace(/\/+$/, '');
    if (!url) {
      alert("Masukkan URL tunnel Colab terlebih dahulu (misal: https://xxxx.trycloudflare.com)");
      return;
    }
    btnTestColab.textContent = '...';
    try {
      const res = await fetch(`${url}/api/sample`);
      const data = await res.json();
      if (data.status === 'ok') {
        alert("✅ Berhasil terhubung ke Google Colab GPU Server!\nSemua proses deteksi & inpainting sekarang menggunakan GPU Colab.");
        updateApiBaseUrl();
      } else {
        alert("⚠️ Respons diterima tetapi format tidak sesuai.");
      }
    } catch (err) {
      alert("❌ Gagal terhubung ke Colab URL.\nPastikan server di Google Colab masih berjalan dan URL Cloudflare tunnel masih aktif.\nError: " + err);
    } finally {
      btnTestColab.textContent = 'Test';
    }
  });

  // ----------------------------------------------------
  // Image Loading & Canvas Initialization
  // ----------------------------------------------------
  function loadImage(src, filename = "image.jpg") {
    showLoading("Loading image...");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      currentImage = img;
      originalBase64 = src;
      cleanedBase64 = null;

      // Setup canvas dimensions
      const w = img.naturalWidth;
      const h = img.naturalHeight;

      canvasBase.width = w;
      canvasBase.height = h;
      canvasResult.width = w;
      canvasResult.height = h;
      canvasMask.width = w;
      canvasMask.height = h;

      // Fit and scale properly inside viewport
      fitCanvasToViewport(w, h);

      // Draw original on base and result initially
      ctxBase.clearRect(0, 0, w, h);
      ctxBase.drawImage(img, 0, 0);

      ctxResult.clearRect(0, 0, w, h);
      ctxResult.drawImage(img, 0, 0);

      clearMask();

      // Update UI
      dropzone.style.display = 'none';
      canvasStage.style.display = 'flex';
      btnInpaint.disabled = false;
      btnDownload.disabled = true;

      statDim.textContent = `${w} × ${h}`;
      statMasked.textContent = '0 px';
      statStamps.textContent = '0';
      statTime.textContent = '—';

      setSplitPosition(50);
      setViewMode('split');
      hideLoading();

      // Trigger automatic scan on load
      autoScanWatermark();
    };
    img.src = src;
  }

  // File Upload Handlers
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (evt) => loadImage(evt.target.result, file.name);
      reader.readAsDataURL(file);
    }
  });

  // Drag and Drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      const reader = new FileReader();
      reader.onload = (evt) => loadImage(evt.target.result, file.name);
      reader.readAsDataURL(file);
    }
  });

  // Clipboard Paste Support (Ctrl+V)
  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = (evt) => loadImage(evt.target.result, "pasted_image.png");
        reader.readAsDataURL(file);
        break;
      }
    }
  });

  // Demo Sample Button
  btnLoadDemo.addEventListener('click', async () => {
    showLoading("Generating realistic watermark sample...");
    try {
      const res = await fetch(`${apiBaseUrl}/api/sample`);
      const data = await res.json();
      if (data.status === 'ok') {
        loadImage(data.image, data.filename);
      }
    } catch (err) {
      alert("Error loading demo sample: " + err);
      hideLoading();
    }
  });

  // ----------------------------------------------------
  // Interactive Brush / Eraser on Mask Canvas
  // ----------------------------------------------------
  function getCanvasCoords(e) {
    const rect = canvasMask.getBoundingClientRect();
    const scaleX = canvasMask.width / rect.width;
    const scaleY = canvasMask.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  canvasMask.addEventListener('mousedown', (e) => {
    if (activeTool === 'auto') return;
    isDrawing = true;
    drawStroke(e);
  });

  window.addEventListener('mousemove', (e) => {
    // Update floating brush cursor
    if (activeTool === 'brush' || activeTool === 'eraser') {
      brushCursor.style.left = `${e.clientX}px`;
      brushCursor.style.top = `${e.clientY}px`;
    }

    if (!isDrawing) return;
    drawStroke(e);
  });

  window.addEventListener('mouseup', () => {
    if (isDrawing) {
      isDrawing = false;
      updateMaskStats();
    }
  });

  function drawStroke(e) {
    const { x, y } = getCanvasCoords(e);
    const size = parseInt(brushSizeInput.value, 10);

    ctxMask.save();
    if (activeTool === 'eraser') {
      ctxMask.globalCompositeOperation = 'destination-out';
      ctxMask.beginPath();
      ctxMask.arc(x, y, size / 2, 0, Math.PI * 2);
      ctxMask.fill();
    } else {
      // Semi-transparent neon red brush
      ctxMask.globalCompositeOperation = 'source-over';
      ctxMask.fillStyle = 'rgba(239, 68, 68, 0.7)';
      ctxMask.beginPath();
      ctxMask.arc(x, y, size / 2, 0, Math.PI * 2);
      ctxMask.fill();
    }
    ctxMask.restore();
  }

  function clearMask() {
    ctxMask.clearRect(0, 0, canvasMask.width, canvasMask.height);
    updateMaskStats();
  }

  if (btnClearMask) {
    btnClearMask.addEventListener('click', clearMask);
  }

  function resetImage() {
    currentImage = null;
    originalBase64 = null;
    cleanedBase64 = null;
    if (fileInput) fileInput.value = '';

    // Clear canvases
    ctxBase.clearRect(0, 0, canvasBase.width, canvasBase.height);
    ctxResult.clearRect(0, 0, canvasResult.width, canvasResult.height);
    ctxMask.clearRect(0, 0, canvasMask.width, canvasMask.height);

    // Switch view back to dropzone
    canvasStage.style.display = 'none';
    dropzone.style.display = 'flex';

    // Disable action buttons
    btnInpaint.disabled = true;
    btnDownload.disabled = true;

    // Reset telemetry stats
    statDim.textContent = '—';
    statMasked.textContent = '0 px';
    statStamps.textContent = '0';
    statTime.textContent = '—';
  }

  if (btnResetImage) {
    btnResetImage.addEventListener('click', resetImage);
  }

  function updateMaskStats() {
    const imgData = ctxMask.getImageData(0, 0, canvasMask.width, canvasMask.height);
    let count = 0;
    for (let i = 3; i < imgData.data.length; i += 4) {
      if (imgData.data[i] > 20) count++;
    }
    statMasked.textContent = `${count.toLocaleString()} px`;
  }

  // ----------------------------------------------------
  // Split Curtain Slider (Before / After)
  // ----------------------------------------------------
  function setSplitPosition(percent) {
    splitPercent = Math.max(0, Math.min(100, percent));
    layerBefore.style.width = `${splitPercent}%`;
    splitDivider.style.left = `${splitPercent}%`;
  }

  splitDivider.addEventListener('mousedown', (e) => {
    isDraggingSplit = true;
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDraggingSplit || viewMode !== 'split') return;
    const rect = splitContainer.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = (x / rect.width) * 100;
    setSplitPosition(pct);
  });

  window.addEventListener('mouseup', () => {
    isDraggingSplit = false;
  });

  // View Mode Tabs
  function setViewMode(mode) {
    viewMode = mode;
    modeSplit.classList.toggle('active', mode === 'split');
    modeClean.classList.toggle('active', mode === 'clean');
    modeOriginal.classList.toggle('active', mode === 'original');
    modeMask.classList.toggle('active', mode === 'mask');

    if (mode === 'split') {
      layerBefore.style.display = 'block';
      layerAfter.style.display = 'block';
      splitDivider.style.display = 'block';
      canvasMask.style.display = 'block';
      setSplitPosition(splitPercent);
    } else if (mode === 'clean') {
      layerBefore.style.display = 'none';
      layerAfter.style.display = 'block';
      splitDivider.style.display = 'none';
      canvasMask.style.display = 'none';
    } else if (mode === 'original') {
      layerBefore.style.display = 'block';
      layerBefore.style.width = '100%';
      layerAfter.style.display = 'none';
      splitDivider.style.display = 'none';
      canvasMask.style.display = 'none';
    } else if (mode === 'mask') {
      layerBefore.style.display = 'block';
      layerBefore.style.width = '100%';
      layerAfter.style.display = 'none';
      splitDivider.style.display = 'none';
      canvasMask.style.display = 'block';
    }
  }

  modeSplit.addEventListener('click', () => setViewMode('split'));
  modeClean.addEventListener('click', () => setViewMode('clean'));
  modeOriginal.addEventListener('click', () => setViewMode('original'));
  modeMask.addEventListener('click', () => setViewMode('mask'));

  // ----------------------------------------------------
  // AI Detection & Inpainting Actions
  // ----------------------------------------------------
  async function autoScanWatermark() {
    const wmMode = selectWmMode ? selectWmMode.value : 'auto';
    const scanTitle = wmMode === 'sakerly' ? 'Detecting Sakerly Signature...' : (wmMode === 'rfans' ? 'Scanning RFans Stamps & Text...' : 'AI Scanning Watermarks & Stamps...');
    const scanSteps = wmMode === 'sakerly' 
      ? ["Color Gating", "Template Search", "Anchor Match", "Exact Mask"]
      : ["Corner Scan", "Dual CLAHE", "Transparent Sweep", "Mask Overlay"];

    showLoading(scanTitle, scanSteps);

    try {
      const payload = {
        image: originalBase64,
        wm_mode: wmMode,
        sensitivity: parseFloat(sensitivityInput.value),
        dilation_px: parseInt(dilationInput.value, 10),
        use_ocr: wmMode !== 'sakerly',
        use_heuristics: true
      };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 90000);

      const res = await fetch(`${apiBaseUrl}/api/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      const data = await res.json();

      if (data.status === 'ok') {
        // Draw returned binary mask onto canvasMask with translucent neon red
        const maskImg = new Image();
        maskImg.onload = () => {
          ctxMask.clearRect(0, 0, canvasMask.width, canvasMask.height);
          
          // Draw mask with custom color overlay
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = canvasMask.width;
          tempCanvas.height = canvasMask.height;
          const tempCtx = tempCanvas.getContext('2d');
          tempCtx.drawImage(maskImg, 0, 0);

          const imgData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
          const outData = ctxMask.createImageData(tempCanvas.width, tempCanvas.height);

          for (let i = 0; i < imgData.data.length; i += 4) {
            if (imgData.data[i] > 100) {
              outData.data[i] = 239;     // R
              outData.data[i + 1] = 68;  // G
              outData.data[i + 2] = 68;  // B
              outData.data[i + 3] = 180; // Alpha
            }
          }
          ctxMask.putImageData(outData, 0, 0);

          statMasked.textContent = `${data.pixels_masked.toLocaleString()} px`;
          statStamps.textContent = data.stamp_count;
          hideLoading();
        };
        maskImg.src = data.mask;
      } else {
        alert("Detection error: " + (data.error || "Unknown error"));
        hideLoading();
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        alert("Waktu proses melebihi batas waktu (Timeout). Pastikan Google Colab aktif dan link Cloudflare terhubung.");
      } else {
        alert("Network error during detection: " + err);
      }
      hideLoading();
    }
  }

  btnAutoScan.addEventListener('click', autoScanWatermark);

  // Inpaint Trigger
  btnInpaint.addEventListener('click', async () => {
    if (!currentImage) return;
    showLoading("Cleaning Watermark (Inpaint)...", [
      "Mask Buffer",
      isUsingColab ? "LaMa GPU Inpaint" : "Neural Inpaint",
      "Texture Blending",
      "Lossless Render"
    ]);

    try {
      // Export current mask canvas as PNG base64
      const maskDataUrl = canvasMask.toDataURL('image/png');

      const payload = {
        image: originalBase64,
        mask: maskDataUrl,
        method: selectEngine.value,
        feather: checkFeather.checked
      };

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const res = await fetch(`${apiBaseUrl}/api/inpaint`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      const data = await res.json();

      if (data.status === 'ok') {
        cleanedBase64 = data.cleaned;
        const cleanedImg = new Image();
        cleanedImg.onload = () => {
          ctxResult.clearRect(0, 0, canvasResult.width, canvasResult.height);
          ctxResult.drawImage(cleanedImg, 0, 0);

          statTime.textContent = `${data.elapsed_sec}s`;
          btnDownload.disabled = false;
          setViewMode('split');
          hideLoading();
        };
        cleanedImg.src = cleanedBase64;
      } else {
        alert("Inpainting error: " + (data.error || "Failed to inpaint"));
        hideLoading();
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        alert("Waktu inpainting melebihi batas (Timeout 60s). Silakan coba lagi.");
      } else {
        alert("Network error during inpainting: " + err);
      }
      hideLoading();
    }
  });

  // Download Cleaned Image
  btnDownload.addEventListener('click', () => {
    if (!cleanedBase64) return;
    const a = document.createElement('a');
    a.href = cleanedBase64;
    a.download = `dewatermark_cleaned_${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  // Navigation Switch (Studio vs Batch)
  navStudio.addEventListener('click', () => {
    navStudio.classList.add('active');
    navBatch.classList.remove('active');
    studioMain.style.display = 'grid';
    batchView.style.display = 'none';
  });

  navBatch.addEventListener('click', () => {
    navBatch.classList.add('active');
    navStudio.classList.remove('active');
    studioMain.style.display = 'none';
    batchView.style.display = 'flex';
  });

  // ----------------------------------------------------
  // Batch Queue & Row-by-Row Sequential Processing
  // ----------------------------------------------------
  let batchQueue = [];
  let isBatchRunning = false;

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function getStatusBadgeHtml(item) {
    switch (item.status) {
      case 'processing':
        return `<span class="batch-status-badge status-processing"><span class="pulse-dot"></span> Memproses...</span>`;
      case 'completed':
        return `<span class="batch-status-badge status-completed">✅ Selesai (${item.elapsedSec}s)</span>`;
      case 'error':
        return `<span class="batch-status-badge status-error">❌ Gagal</span>`;
      case 'pending':
      default:
        return `<span class="batch-status-badge status-pending">⏳ Siap</span>`;
    }
  }

  function getRowActionHtml(item) {
    if (item.status === 'completed') {
      const href = item.cleanedBase64 || item.downloadUrl || '#';
      return `<a href="${href}" download="cleaned_${item.filename}" class="btn-row-action btn-row-download" title="Unduh foto ini">⬇️ Unduh</a>`;
    }
    if (item.status === 'processing') {
      return `<span style="font-size:0.75rem; color:var(--accent-secondary); font-family:var(--font-mono);">⚡ Aktif</span>`;
    }
    return `<button class="btn-row-action btn-row-delete" onclick="window.removeBatchItem('${item.id}')" title="Hapus dari antrean">✕</button>`;
  }

  function renderBatchList() {
    if (batchTotalCountBadge) {
      batchTotalCountBadge.textContent = `${batchQueue.length} Foto`;
    }

    if (batchQueue.length === 0) {
      if (batchControlsCard) batchControlsCard.style.display = 'none';
      if (batchListContainer) batchListContainer.style.display = 'none';
      if (batchGlobalProgressCard) batchGlobalProgressCard.style.display = 'none';
      if (btnBatchDownloadAll) btnBatchDownloadAll.style.display = 'none';
      if (batchListBody) batchListBody.innerHTML = '';
      return;
    }

    if (batchControlsCard) batchControlsCard.style.display = 'flex';
    if (batchListContainer) batchListContainer.style.display = 'block';

    if (!batchListBody) return;
    batchListBody.innerHTML = '';

    batchQueue.forEach((item) => {
      const row = document.createElement('div');
      row.className = `batch-row ${item.status === 'processing' ? 'is-processing' : ''} ${item.status === 'completed' ? 'is-completed' : ''} ${item.status === 'error' ? 'is-error' : ''}`;
      row.id = `row-${item.id}`;

      row.innerHTML = `
        <div class="batch-col col-preview">
          <div class="batch-row-thumb-wrap">
            <img src="${item.thumb || item.previewUrl}" class="batch-row-thumb" alt="${item.filename}">
            ${item.status === 'completed' ? '<span class="batch-thumb-check">✓</span>' : ''}
          </div>
        </div>
        <div class="batch-col col-filename">
          <div class="batch-row-name" title="${item.filename}">${item.filename}</div>
          <div class="batch-row-meta">${formatBytes(item.size)}</div>
        </div>
        <div class="batch-col col-progress">
          <div class="batch-row-progress-wrap">
            <div class="batch-row-progress-track">
              <div class="batch-row-progress-bar" id="bar-${item.id}" style="width: ${item.progress}%;"></div>
            </div>
            <span class="batch-row-percent" id="pct-${item.id}">${item.progress}%</span>
          </div>
        </div>
        <div class="batch-col col-status" id="status-cell-${item.id}">
          ${getStatusBadgeHtml(item)}
        </div>
        <div class="batch-col col-actions" id="actions-cell-${item.id}">
          ${getRowActionHtml(item)}
        </div>
      `;

      batchListBody.appendChild(row);
    });
  }

  function addFilesToBatch(files) {
    if (!files || files.length === 0) return;
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file.type.startsWith('image/')) continue;
      const item = {
        id: 'item_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6),
        file: file,
        filename: file.name,
        size: file.size,
        previewUrl: URL.createObjectURL(file),
        status: 'pending',
        progress: 0,
        elapsedSec: null,
        thumb: null,
        downloadUrl: null
      };
      batchQueue.push(item);
    }
    renderBatchList();
  }

  // Window helper for row removal
  window.removeBatchItem = function(id) {
    if (isBatchRunning) return;
    const idx = batchQueue.findIndex(it => it.id === id);
    if (idx !== -1) {
      if (batchQueue[idx].previewUrl) URL.revokeObjectURL(batchQueue[idx].previewUrl);
      batchQueue.splice(idx, 1);
      renderBatchList();
    }
  };

  // Batch Dropzone Event Listeners
  if (batchDropzone) {
    batchDropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      batchDropzone.classList.add('dragover');
    });
    batchDropzone.addEventListener('dragleave', () => batchDropzone.classList.remove('dragover'));
    batchDropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      batchDropzone.classList.remove('dragover');
      if (e.dataTransfer.files) {
        addFilesToBatch(e.dataTransfer.files);
      }
    });
  }

  // Batch File Input (NO auto processing, just queue)
  if (batchFileInput) {
    batchFileInput.addEventListener('change', (e) => {
      if (e.target.files) {
        addFilesToBatch(e.target.files);
      }
      batchFileInput.value = '';
    });
  }

  // Clear Batch Queue Button
  if (btnClearBatchQueue) {
    btnClearBatchQueue.addEventListener('click', () => {
      if (isBatchRunning) return;
      batchQueue.forEach(it => {
        if (it.previewUrl) URL.revokeObjectURL(it.previewUrl);
      });
      batchQueue = [];
      currentBatchId = null;
      renderBatchList();
    });
  }

  // Manual Start Batch Button (proses satu persatu secara berurutan)
  if (btnStartBatch) {
    btnStartBatch.addEventListener('click', async () => {
      if (isBatchRunning) return;
      const pendingItems = batchQueue.filter(it => it.status !== 'completed');
      if (pendingItems.length === 0) {
        alert("Semua foto dalam antrean sudah selesai diproses. Tambahkan foto baru untuk melanjutkan.");
        return;
      }

      isBatchRunning = true;
      currentBatchId = 'batch_' + Date.now();

      // UI state during processing
      btnStartBatch.disabled = true;
      if (btnAddMoreBatch) btnAddMoreBatch.disabled = true;
      if (btnClearBatchQueue) btnClearBatchQueue.disabled = true;
      if (batchWmModeSelect) batchWmModeSelect.disabled = true;
      if (batchEngineSelect) batchEngineSelect.disabled = true;
      btnStartBatch.innerHTML = `<span class="pulse-dot"></span><span>Memproses Antrean...</span>`;

      if (batchGlobalProgressCard) batchGlobalProgressCard.style.display = 'block';

      const totalItems = pendingItems.length;
      let completedCount = 0;
      const globalStartT = performance.now();

      const globalTimerInterval = setInterval(() => {
        const elapsedSec = ((performance.now() - globalStartT) / 1000).toFixed(1);
        if (batchGlobalTimer) batchGlobalTimer.textContent = `${elapsedSec}s`;
      }, 100);

      // Process each file sequentially
      for (let i = 0; i < pendingItems.length; i++) {
        const item = pendingItems[i];
        item.status = 'processing';
        item.progress = 12;

        const rowEl = document.getElementById(`row-${item.id}`);
        const barEl = document.getElementById(`bar-${item.id}`);
        const pctEl = document.getElementById(`pct-${item.id}`);
        const statusCell = document.getElementById(`status-cell-${item.id}`);
        const actionsCell = document.getElementById(`actions-cell-${item.id}`);

        if (rowEl) {
          rowEl.className = 'batch-row is-processing';
          rowEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        if (statusCell) statusCell.innerHTML = getStatusBadgeHtml(item);
        if (actionsCell) actionsCell.innerHTML = getRowActionHtml(item);
        if (barEl) barEl.style.width = '12%';
        if (pctEl) pctEl.textContent = '12%';

        // Smooth progress animation ticker for this item
        let curItemProg = 12;
        const itemTicker = setInterval(() => {
          if (curItemProg < 88) {
            curItemProg += Math.max(1, (88 - curItemProg) * 0.12);
            const rProg = Math.round(curItemProg);
            if (barEl) barEl.style.width = `${rProg}%`;
            if (pctEl) pctEl.textContent = `${rProg}%`;
          }
        }, 120);

        // Prepare request
        const formData = new FormData();
        formData.append('files', item.file);
        formData.append('batch_id', currentBatchId);
        formData.append('method', batchEngineSelect ? batchEngineSelect.value : (selectEngine ? selectEngine.value : 'auto'));
        formData.append('wm_mode', batchWmModeSelect ? batchWmModeSelect.value : (selectWmMode ? selectWmMode.value : 'auto'));
        formData.append('sensitivity', sensitivityInput ? sensitivityInput.value : '0.5');
        formData.append('dilation_px', dilationInput ? dilationInput.value : '2');

        try {
          const res = await fetch(`${apiBaseUrl}/api/batch`, {
            method: 'POST',
            body: formData
          });
          const data = await res.json();
          clearInterval(itemTicker);

          if (data.status === 'ok' && data.items && data.items.length > 0) {
            const outItem = data.items[0];
            item.status = 'completed';
            item.progress = 100;
            item.elapsedSec = outItem.elapsed_sec;
            item.thumb = outItem.thumb || outItem.cleaned;
            item.cleanedBase64 = outItem.cleaned;
            item.downloadUrl = outItem.cleaned || (outItem.download_url ? `${apiBaseUrl}${outItem.download_url}` : null);

            if (rowEl) rowEl.className = 'batch-row is-completed';
            if (barEl) barEl.style.width = '100%';
            if (pctEl) pctEl.textContent = '100%';
            if (statusCell) statusCell.innerHTML = getStatusBadgeHtml(item);
            if (actionsCell) actionsCell.innerHTML = getRowActionHtml(item);

            // Update row thumbnail to cleaned image
            const thumbImg = rowEl ? rowEl.querySelector('.batch-row-thumb') : null;
            if (thumbImg && item.thumb) thumbImg.src = item.thumb;
            const thumbWrap = rowEl ? rowEl.querySelector('.batch-row-thumb-wrap') : null;
            if (thumbWrap && !thumbWrap.querySelector('.batch-thumb-check')) {
              const checkBadge = document.createElement('span');
              checkBadge.className = 'batch-thumb-check';
              checkBadge.textContent = '✓';
              thumbWrap.appendChild(checkBadge);
            }

            completedCount++;
          } else {
            throw new Error(data.error || "Gagal memproses gambar");
          }
        } catch (err) {
          clearInterval(itemTicker);
          item.status = 'error';
          item.progress = 100;
          if (rowEl) rowEl.className = 'batch-row is-error';
          if (barEl) barEl.style.width = '100%';
          if (pctEl) pctEl.textContent = '100%';
          if (statusCell) statusCell.innerHTML = getStatusBadgeHtml(item);
          if (actionsCell) actionsCell.innerHTML = getRowActionHtml(item);
          console.error("Batch item error:", err);
        }

        // Update global progress bar & text
        const overallPct = Math.round(((i + 1) / totalItems) * 100);
        if (batchGlobalProgressFill) batchGlobalProgressFill.style.width = `${overallPct}%`;
        if (batchGlobalStatusText) {
          batchGlobalStatusText.textContent = `Memproses: ${completedCount} dari ${totalItems} foto selesai (${overallPct}%)`;
        }
      }

      // Finish batch run
      clearInterval(globalTimerInterval);
      const totalElapsed = ((performance.now() - globalStartT) / 1000).toFixed(1);

      isBatchRunning = false;
      btnStartBatch.disabled = false;
      if (btnAddMoreBatch) btnAddMoreBatch.disabled = false;
      if (btnClearBatchQueue) btnClearBatchQueue.disabled = false;
      if (batchWmModeSelect) batchWmModeSelect.disabled = false;
      if (batchEngineSelect) batchEngineSelect.disabled = false;

      btnStartBatch.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        <span>Mulai Proses Lagi</span>
      `;

      if (completedCount > 0 && btnBatchDownloadAll) {
        btnBatchDownloadAll.style.display = 'inline-flex';
      }

      if (batchGlobalStatusText) {
        batchGlobalStatusText.textContent = `✅ Selesai! ${completedCount} foto berhasil dibersihkan (${totalElapsed}s)`;
      }
    });
  }

  // ZIP Download for Batch (Browser-Side JSZip Instant + Server Fallback)
  if (btnBatchDownloadAll) {
    btnBatchDownloadAll.addEventListener('click', async () => {
      const completedItems = batchQueue.filter(it => it.status === 'completed');
      if (completedItems.length === 0) return;

      const originalText = btnBatchDownloadAll.innerHTML;
      btnBatchDownloadAll.innerHTML = `<span>⏳ Mengunduh ZIP...</span>`;
      btnBatchDownloadAll.disabled = true;

      try {
        // Mode 1 (Super Cepat & 100% Anti-Gagal): Buat ZIP langsung di browser menggunakan JSZip
        if (window.JSZip) {
          const zip = new window.JSZip();
          for (const it of completedItems) {
            if (it.cleanedBase64) {
              const b64Data = it.cleanedBase64.replace(/^data:image\/[a-z]+;base64,/, '');
              zip.file(`cleaned_${it.filename}`, b64Data, { base64: true });
            }
          }
          const blob = await zip.generateAsync({ type: 'blob' });
          const blobUrl = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = blobUrl;
          a.download = `${currentBatchId || 'cleaned_photos'}.zip`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          window.URL.revokeObjectURL(blobUrl);
          return;
        }

        // Mode 2: Unduh ZIP tunggal gabungan dari server / Colab
        const downloadUrl = `${apiBaseUrl}/api/download-batch/${currentBatchId || 'batch'}`;
        const response = await fetch(downloadUrl);
        if (!response.ok) throw new Error("Gagal mengunduh ZIP dari server (" + response.status + ")");

        const blob = await response.blob();
        const blobUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `${currentBatchId || 'cleaned_photos'}.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(blobUrl);
      } catch (err) {
        console.warn("Direct blob download notice, fallback to standard link:", err);
        window.location.href = `${apiBaseUrl}/api/download-batch/${currentBatchId || 'batch'}`;
      } finally {
        btnBatchDownloadAll.innerHTML = originalText;
        btnBatchDownloadAll.disabled = false;
      }
    });
  }

  // ----------------------------------------------------
  // Dynamic Real-Time Progress & Heartbeat Controller
  // ----------------------------------------------------
  const progressTitle = document.getElementById('progressTitle');
  const progressDetail = document.getElementById('progressDetail');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressPercent = document.getElementById('progressPercent');
  const progressTimer = document.getElementById('progressTimer');
  const engineStatusText = document.getElementById('engineStatusText');
  const stepItems = [
    document.getElementById('stepItem0'),
    document.getElementById('stepItem1'),
    document.getElementById('stepItem2'),
    document.getElementById('stepItem3')
  ];

  let progressInterval = null;
  let progressStartTime = 0;
  let currentProgressVal = 0;

  function showLoading(title = "Processing...", customSteps = null) {
    if (progressInterval) clearInterval(progressInterval);
    processingOverlay.style.display = 'flex';
    if (progressTitle) progressTitle.textContent = title;
    if (progressBarFill) progressBarFill.style.width = '6%';
    if (progressPercent) progressPercent.textContent = '6%';
    currentProgressVal = 6;
    progressStartTime = performance.now();
    if (progressTimer) progressTimer.textContent = '0.0s';
    if (engineStatusText) {
      engineStatusText.textContent = isUsingColab ? "Colab GPU Active" : "Local Engine Active";
    }

    const defaultSteps = [
      "Mask Buffer",
      isUsingColab ? "LaMa GPU Inpaint" : "Neural Inpaint",
      "Texture Blending",
      "Render Output"
    ];
    const stepsToUse = customSteps || defaultSteps;

    stepItems.forEach((item, idx) => {
      if (item) {
        const labelEl = item.querySelector('.step-label');
        const iconEl = item.querySelector('.step-icon');
        if (labelEl && stepsToUse[idx]) labelEl.textContent = stepsToUse[idx];
        if (iconEl) iconEl.textContent = (idx + 1).toString();
        item.className = 'step-item' + (idx === 0 ? ' active' : '');
      }
    });

    updateStep(0, "Preparing neural pipeline & mask coordinates...");

    // Real-time ticking progress & timer loop (100ms)
    progressInterval = setInterval(() => {
      const elapsedSec = (performance.now() - progressStartTime) / 1000;
      if (progressTimer) progressTimer.textContent = `${elapsedSec.toFixed(1)}s`;

      // Smooth progress curve
      if (currentProgressVal < 92) {
        const remaining = 92 - currentProgressVal;
        currentProgressVal += Math.max(0.35, remaining * 0.08);
        const rounded = Math.round(currentProgressVal);
        if (progressBarFill) progressBarFill.style.width = `${rounded}%`;
        if (progressPercent) progressPercent.textContent = `${rounded}%`;

        if (currentProgressVal >= 25 && currentProgressVal < 65) {
          updateStep(1, isUsingColab ? "SimpleLaMa executing on Colab GPU..." : "Synthesizing surrounding texture...");
        } else if (currentProgressVal >= 65 && currentProgressVal < 88) {
          updateStep(2, "Feathering boundary & blending micro-textures...");
        } else if (currentProgressVal >= 88) {
          updateStep(3, "Receiving lossless high-res buffer...");
        }
      }

      if (elapsedSec > 12 && progressDetail) {
        progressDetail.textContent = "Processing high-res image... Please wait, GPU is working actively!";
      }
    }, 100);
  }

  function updateStep(activeIdx, detailText) {
    if (detailText && progressDetail) progressDetail.textContent = detailText;
    stepItems.forEach((item, idx) => {
      if (!item) return;
      const iconEl = item.querySelector('.step-icon');
      if (idx < activeIdx) {
        item.className = 'step-item completed';
        if (iconEl) iconEl.textContent = '✓';
      } else if (idx === activeIdx) {
        item.className = 'step-item active';
      } else {
        item.className = 'step-item';
      }
    });
  }

  function hideLoading(onComplete) {
    if (progressInterval) clearInterval(progressInterval);
    if (progressBarFill) progressBarFill.style.width = '100%';
    if (progressPercent) progressPercent.textContent = '100%';
    if (progressDetail) progressDetail.textContent = "Complete! Rendering canvas...";
    
    stepItems.forEach(item => {
      if (item) {
        item.className = 'step-item completed';
        const iconEl = item.querySelector('.step-icon');
        if (iconEl) iconEl.textContent = '✓';
      }
    });

    setTimeout(() => {
      processingOverlay.style.display = 'none';
      if (onComplete) onComplete();
    }, 320);
  }
});
