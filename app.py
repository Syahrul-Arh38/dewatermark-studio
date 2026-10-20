"""
app.py - Dewatermark AI Studio Web Server
A modern interactive web interface mimicking and surpassing Dewatermark.ai:
- Real-time interactive brush & mask editor
- Before/After split curtain comparison slider
- Dual-pass CLAHE + multi-cluster stamp auto detection
- Multi-engine inpainting
- Batch queue processing
"""

import os
import io
import time
import zipfile
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory

from core.pipeline import WatermarkPipeline
from core.utils import (
    bgr_to_base64,
    base64_to_bgr,
    base64_to_mask,
    save_image_lossless,
    create_sample_watermark_image
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
SAMPLES_FOLDER = os.path.join(os.path.dirname(__file__), 'samples')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

# CORS headers for remote Colab connection
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Auto-detect CUDA GPU (Google Colab / Nvidia GPU)
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

print(f"[*] Watermark Engine Acceleration: {'CUDA GPU (High Speed)' if HAS_CUDA else 'CPU (Edge-Guided)'}")
pipeline = WatermarkPipeline(use_gpu=HAS_CUDA)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample')
def get_sample():
    """Generate or retrieve a realistic sample image."""
    sample_path = os.path.join(SAMPLES_FOLDER, 'demo_sample.jpg')
    if not os.path.exists(sample_path):
        create_sample_watermark_image(sample_path)
        
    img = cv2.imread(sample_path)
    if img is None:
        create_sample_watermark_image(sample_path)
        img = cv2.imread(sample_path)
        
    return jsonify({
        'status': 'ok',
        'image': bgr_to_base64(img, 'jpg', quality=95),
        'filename': 'demo_sample.jpg',
        'width': img.shape[1],
        'height': img.shape[0]
    })


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """Detect watermark regions and return binary mask overlay."""
    data = request.get_json(force=True)
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400

    img_bgr = base64_to_bgr(image_b64)
    sensitivity = float(data.get('sensitivity', 0.5))
    dilation_px = int(data.get('dilation_px', 2))
    wm_mode = str(data.get('wm_mode', 'auto'))
    use_ocr = bool(data.get('use_ocr', True))
    use_heuristics = bool(data.get('use_heuristics', True))

    det_res = pipeline.detector.detect(
        img_bgr=img_bgr,
        wm_mode=wm_mode,
        use_ocr=use_ocr,
        use_heuristics=use_heuristics,
        sensitivity=sensitivity,
        dilation_px=dilation_px
    )

    mask = det_res['mask']
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return jsonify({
        'status': 'ok',
        'mask': bgr_to_base64(mask_bgr, 'png'),
        'stamp_count': det_res['stamp_count'],
        'detected_texts': det_res.get('detected_texts', []),
        'mode': det_res.get('mode', wm_mode),
        'pixels_masked': int(np.count_nonzero(mask))
    })


@app.route('/api/inpaint', methods=['POST'])
def api_inpaint():
    """Inpaint image using provided or detected mask."""
    data = request.get_json(force=True)
    image_b64 = data.get('image')
    mask_b64 = data.get('mask')

    if not image_b64:
        return jsonify({'error': 'No image provided'}), 400

    img_bgr = base64_to_bgr(image_b64)
    h, w = img_bgr.shape[:2]

    if mask_b64:
        mask = base64_to_mask(mask_b64, (h, w))
    else:
        # Fallback to auto detection if mask was not supplied
        det_res = pipeline.detector.detect(img_bgr)
        mask = det_res['mask']

    method = data.get('method', 'auto')
    feather = bool(data.get('feather', True))

    start_t = time.time()
    cleaned = pipeline.inpainter.inpaint(
        img_bgr=img_bgr,
        mask=mask,
        method=method,
        feather_edges=feather
    )
    elapsed = round(time.time() - start_t, 3)

    return jsonify({
        'status': 'ok',
        'cleaned': bgr_to_base64(cleaned, 'jpg', quality=98),
        'elapsed_sec': elapsed,
        'dimensions': {'width': w, 'height': h}
    })


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """Process multiple uploaded files or sequential single-file batch items."""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    method = request.form.get('method', 'auto')
    sensitivity = float(request.form.get('sensitivity', 0.5))
    dilation = int(request.form.get('dilation_px', 2))
    wm_mode = request.form.get('wm_mode', 'auto')

    results = []
    timestamp = int(time.time())
    batch_id = request.form.get('batch_id')
    if not batch_id or not batch_id.startswith('batch_'):
        batch_id = f"batch_{timestamp}"

    batch_out_dir = os.path.join(OUTPUT_FOLDER, batch_id)
    os.makedirs(batch_out_dir, exist_ok=True)

    for file in files:
        if not file.filename:
            continue
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue

        res = pipeline.process(
            img_bgr=img,
            use_ocr=True,
            use_heuristics=True,
            sensitivity=sensitivity,
            dilation_px=dilation,
            wm_mode=wm_mode,
            inpaint_method=method
        )

        out_name = f"cleaned_{file.filename}"
        out_path = os.path.join(batch_out_dir, out_name)
        save_image_lossless(res['cleaned_bgr'], out_path)

        # Generate lightweight thumbnail
        h_orig, w_orig = res['cleaned_bgr'].shape[:2]
        thumb_w = 160
        thumb_h = max(1, int(thumb_w * h_orig / max(1, w_orig)))
        thumb_bgr = cv2.resize(res['cleaned_bgr'], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        results.append({
            'filename': file.filename,
            'out_filename': out_name,
            'elapsed_sec': res['elapsed_sec'],
            'thumb': bgr_to_base64(thumb_bgr, 'jpg', quality=85),
            'download_url': f"/api/download-file/{batch_id}/{out_name}"
        })

    return jsonify({
        'status': 'ok',
        'processed_count': len(results),
        'batch_id': batch_id,
        'items': results
    })


@app.route('/api/download-file/<batch_id>/<filename>')
def download_single_batch_file(batch_id, filename):
    """Download an individual cleaned image from a batch."""
    safe_batch = os.path.basename(batch_id)
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_FOLDER, safe_batch, safe_name)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    from flask import send_file
    return send_file(file_path, as_attachment=True, download_name=safe_name)


@app.route('/api/download-batch/<batch_id>')
def download_batch_zip(batch_id):
    """Download batch results as a ZIP archive."""
    batch_dir = os.path.join(OUTPUT_FOLDER, batch_id)
    if not os.path.exists(batch_dir):
        return jsonify({'error': 'Batch not found'}), 404

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(batch_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zip_file.write(file_path, arcname=file)

    zip_buffer.seek(0)
    data = zip_buffer.getvalue()
    from flask import Response
    resp = Response(data, mimetype='application/zip')
    resp.headers['Content-Disposition'] = f'attachment; filename="{batch_id}.zip"'
    resp.headers['Content-Length'] = str(len(data))
    resp.headers['Accept-Ranges'] = 'none'
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


if __name__ == '__main__':
    port = 5000
    print(f"[*] Starting Dewatermark AI Studio on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, debug=False)
