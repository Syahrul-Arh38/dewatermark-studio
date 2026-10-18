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

  const btnAutoScan = document.getElementById('btnAutoScan');
  const btnInpaint = document.getElementById('btnInpaint');
  const btnDownload = document.getElementById('btnDownload');
  const btnLoadDemo = document.getElementById('btnLoadDemo');
  const btnClearMask = document.getElementById('btnClearMask');
  const btnZoomFit = document.getElementById('btnZoomFit');

  // Stats
  const statDim = document.getElementById('statDim');
  const statMasked = document.getElementById('statMasked');
  const statStamps = document.getElementById('statStamps');
  const statTime = document.getElementById('statTime');

  // View modes
  const modeSplit = document.getElementById('modeSplit');
  const modeClean = document.getElementById('modeClean');
  const modeOriginal = document.getElementById('modeOriginal');
  const modeMask = document.getElementById('modeMask');

  // Navigation
  const navStudio = document.getElementById('navStudio');
  const navBatch = document.getElementById('navBatch');
  const studioMain = document.querySelector('.studio-main');
  const batchView = document.getElementById('batchView');
  const batchFileInput = document.getElementById('batchFileInput');
  const batchGrid = document.getElementById('batchGrid');
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
      const res = await fetch('/api/sample');
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

  btnClearMask.addEventListener('click', clearMask);

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
    if (!currentImage) return;
    showLoading("AI is scanning watermarks & circular stamps...");

    try {
      const payload = {
        image: originalBase64,
        sensitivity: parseFloat(sensitivityInput.value),
        dilation_px: parseInt(dilationInput.value, 10),
        use_ocr: true,
        use_heuristics: true
      };

      const res = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
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
      alert("Network error during detection: " + err);
      hideLoading();
    }
  }

  btnAutoScan.addEventListener('click', autoScanWatermark);

  // Inpaint Trigger
  btnInpaint.addEventListener('click', async () => {
    if (!currentImage) return;
    showLoading("Inpainting watermark with selected engine...");

    try {
      // Export current mask canvas as PNG base64
      const maskDataUrl = canvasMask.toDataURL('image/png');

      const payload = {
        image: originalBase64,
        mask: maskDataUrl,
        method: selectEngine.value,
        feather: checkFeather.checked
      };

      const res = await fetch('/api/inpaint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
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
      alert("Network error during inpaint: " + err);
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
  // Batch Queue Processing
  // ----------------------------------------------------
  batchFileInput.addEventListener('change', async (e) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const files = e.target.files;

    batchGrid.innerHTML = '';
    showLoading(`Processing batch of ${files.length} images...`);

    const formData = new FormData();
    for (let f of files) {
      formData.append('files', f);
    }
    formData.append('method', selectEngine.value);
    formData.append('sensitivity', sensitivityInput.value);
    formData.append('dilation_px', dilationInput.value);

    try {
      const res = await fetch('/api/batch', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.status === 'ok') {
        currentBatchId = data.batch_id;
        btnBatchDownloadAll.style.display = 'inline-flex';

        data.items.forEach(item => {
          const card = document.createElement('div');
          card.className = 'batch-card';
          card.innerHTML = `
            <img class="batch-thumb" src="${item.thumb}" alt="${item.filename}">
            <div class="batch-info">
              <div style="font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${item.filename}</div>
              <div style="color:var(--success); font-size:0.75rem; margin-top:4px;">Cleaned in ${item.elapsed_sec}s</div>
            </div>
          `;
          batchGrid.appendChild(card);
        });
      }
      hideLoading();
    } catch (err) {
      alert("Batch error: " + err);
      hideLoading();
    }
  });

  btnBatchDownloadAll.addEventListener('click', () => {
    if (!currentBatchId) return;
    window.location.href = `/api/download-batch/${currentBatchId}`;
  });

  // Loading Indicator Helpers
  function showLoading(text) {
    processingText.textContent = text;
    processingOverlay.style.display = 'flex';
  }
  function hideLoading() {
    processingOverlay.style.display = 'none';
  }
});
