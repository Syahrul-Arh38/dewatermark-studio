# 🌟 Dewatermark Pro Studio & AI Engine

An advanced, end-to-end watermark and stamp removal studio engineered with **Dual-Pass CLAHE Contrast Enhancement**, **Multi-Cluster Stamp Grouping**, **Morphological Stroke Segmentation**, and **Multi-Engine Inpainting (Edge-Guided + LaMa Neural)**.

Built as an open-source, high-fidelity alternative to commercial tools like **Dewatermark.ai**, with complete zero-cost offline capabilities and an interactive real-time web studio.

---

## ⚡ Core Advantages Over Basic OCR / Naive Scripts

| Feature | Naive Script (Colab) | Dewatermark Pro |
| :--- | :--- | :--- |
| **Multiple Stamps** | ❌ **Fatal Bug**: Averages coordinates across distant stamps, creating a giant hole in the center of the image. | ✅ **Spatial Clustering**: Groups fragments with DBSCAN/distance clustering; calculates individual enclosing circles per stamp. |
| **Stamp Radius** | ❌ Fixed 13.5% image dimension radius. | ✅ **Dynamic Enclosing Geometry**: Fits tightly around detected stamp boundaries. |
| **Non-Text / Stylized Watermarks** | ❌ Fails if text isn't in vocabulary. | ✅ **Morphological Stroke Analysis**: Isolates translucent logos and strokes regardless of language. |
| **Low-Contrast / Translucent Text** | ❌ Standard OCR skips low contrast text. | ✅ **Dual-Pass CLAHE**: Enhances luminance channels in LAB color space. |
| **User Control** | ❌ No manual touch-ups. | ✅ **Interactive Web Studio**: Manual brush & eraser, live before/after curtain slider, sensitivity sliders. |
| **Inpainting Seams** | ❌ Hard visible boundary edges. | ✅ **Feathered Bilateral Blending**: Smooth alpha gradient transition. |

---

## 🚀 Quickstart

### 1. Run the Interactive Web Studio
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000` to launch the interactive studio:
- **Load Demo Sample**: One-click demo test with realistic channel stamps & diagonal watermarks.
- **Auto Scan**: AI automatically identifies watermark strokes & circular stamps.
- **Manual Brush & Eraser**: Custom brush sizes to refine masks just like Dewatermark.ai.
- **Before / After Split Slider**: Real-time comparison curtain slider.
- **Batch Queue**: Upload multiple images and download as ZIP.

---

### 2. Command Line Batch Processing (CLI)
Drop-in replacement for Google Colab or local scripts:

```bash
# Test with a generated realistic sample
python cli.py --sample

# Batch process an entire directory of photos
python cli.py --input ./input_photos --output ./cleaned_photos --dilate 11 --sensitivity 0.50
```

---

## 🐍 Fixed Google Colab Script

If you run this on Google Colab with GPU, use this improved script that includes the multi-cluster bug fixes:

```python
# 1. Install dependencies
!pip install -q easyocr simple-lama-inpainting opencv-python-headless pillow

import os
import cv2
import numpy as np
import easyocr
from PIL import Image
from simple_lama_inpainting import SimpleLama

# 2. Inisialisasi Model AI
reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
lama = SimpleLama()

input_folder = '/content/drive/MyDrive/wm_input'
output_folder = '/content/drive/MyDrive/wm_output'
os.makedirs(output_folder, exist_ok=True)

def enhance_for_ocr(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def cluster_stamps(candidate_boxes, max_distance=100):
    clusters = []
    for box in candidate_boxes:
        bx, by, bw, bh = box
        bcx, bcy = bx + bw / 2, by + bh / 2
        placed = False
        for cluster in clusters:
            for cx, cy, cw, ch in cluster:
                if np.hypot(bcx - (cx + cw / 2), bcy - (cy + ch / 2)) <= max_distance:
                    cluster.append(box)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([box])
    return clusters

# Pemrosesan Batch
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
print(f"Memproses {len(files)} gambar...")

for idx, filename in enumerate(files, 1):
    img_path = os.path.join(input_folder, filename)
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        continue

    h, w, _ = img_cv.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Dual-pass scan
    res_normal = reader.readtext(img_cv, contrast_ths=0.05, adjust_contrast=0.7)
    res_enhanced = reader.readtext(enhance_for_ocr(img_cv), contrast_ths=0.03, text_threshold=0.4, low_text=0.2)
    
    wm_keywords = ['rfans', 'channel', 'bot', 'tg', 'album', '电', '报', '群', '搜', '索', '@']
    stamp_keywords = ['channel', 'album', 'bot', 'rfans']
    stamp_candidates = []

    for (bbox, text, prob) in (res_normal + res_enhanced):
        t_low = text.lower().replace(" ", "")
        if any(k in t_low for k in wm_keywords):
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            bx, by = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
            bw, bh = int(np.max(pts[:, 0]) - bx), int(np.max(pts[:, 1]) - by)
            if any(sk in t_low for sk in stamp_keywords) or bw > w * 0.08:
                stamp_candidates.append((bx, by, bw, bh))

    # Masking stempel per klaster independen (mencegah bug koordinat tengah!)
    if stamp_candidates:
        clusters = cluster_stamps(stamp_candidates, max_distance=int(min(h, w) * 0.20))
        for cluster in clusters:
            pts = []
            for (bx, by, bw, bh) in cluster:
                pts.extend([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]])
            (cx, cy), radius = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
            cv2.circle(mask, (int(cx), int(cy)), int(radius * 1.25), 255, -1)

    # Dilasi lembut & Inpaint LaMa
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)

    img_pil = Image.open(img_path).convert("RGB")
    mask_pil = Image.fromarray(mask).convert("L")
    result_pil = lama(img_pil, mask_pil)

    out_path = os.path.join(output_folder, filename)
    result_pil.save(out_path, quality=100, subsampling=0)
    print(f"[{idx}/{len(files)}] Berhasil dibersihkan: {filename}")
```

---

## 📂 Project Architecture

```
dewatermark-studio/
├── app.py                 # Flask server & REST API
├── cli.py                 # CLI Batch runner
├── requirements.txt       # Dependencies
├── core/
│   ├── detector.py        # Multi-cluster stamp & stroke detection engine
│   ├── inpainter.py       # Edge-guided & neural inpainting
│   ├── pipeline.py        # Unified orchestrator
│   └── utils.py           # Image I/O & sample generator
├── templates/
│   └── index.html         # Studio interface
├── static/
│   ├── css/style.css      # Cyber-glass theme & responsive design
│   └── js/studio.js       # Canvas editor & before/after slider
├── samples/               # Sample test files
└── output/                # Cleaned image outputs
```
