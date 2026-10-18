"""
colab_server.py - Google Colab GPU Backend Server for Dewatermark Studio
Run this file directly in Google Colab with T4 GPU runtime enabled.
"""

import os
import sys
import time
import re
import subprocess
import base64
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from simple_lama_inpainting import SimpleLama

# 1. Cek Ketersediaan GPU CUDA
print("=" * 65)
if torch.cuda.is_available():
    print(f"🚀 CUDA GPU Aktif: {torch.cuda.get_device_name(0)}")
    print(f"⚡ VRAM Tersedia: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    USE_GPU = True
else:
    print("⚠️ PERINGATAN: GPU tidak terdeteksi! Pastikan Anda memilih Runtime -> T4 GPU di Colab.")
    USE_GPU = False
print("=" * 65)

# 2. Inisialisasi Model Inpainting LaMa (Utama untuk menghapus watermark)
print("[*] Memuat model Inpainting LaMa ke GPU...")
lama = SimpleLama()
print("✅ SimpleLaMa Inpainting berhasil dimuat ke GPU!")

# 3. EasyOCR (Opsional untuk deteksi teks - jika modul bidi belum ada, server tetap jalan 100%)
reader = None
try:
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=USE_GPU)
    print("✅ EasyOCR berhasil dimuat ke GPU!")
except Exception as e:
    print(f"ℹ️ EasyOCR belum aktif ({e}). Server LaMa Inpainting GPU tetap jalan 100%!")

app = Flask(__name__)
CORS(app)

def bgr_to_base64(img_bgr, ext='jpg', quality=95):
    if ext == 'jpg':
        _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = 'image/jpeg'
    else:
        _, buf = cv2.imencode('.png', img_bgr)
        mime = 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(buf).decode('utf-8')

def base64_to_bgr(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64_str), np.uint8), cv2.IMREAD_COLOR)

def enhance_for_ocr(img_bgr):
    """Multi-scale CLAHE untuk memunculkan teks transparan paling tipis/gelap"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_standard = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_ultra = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(4, 4))
    l_enh = cv2.addWeighted(clahe_standard.apply(l), 0.4, clahe_ultra.apply(l), 0.6, 0)
    return cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)

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

@app.route('/api/sample')
def api_sample():
    return jsonify({'status': 'ok', 'device': 'CUDA GPU' if USE_GPU else 'CPU'})

@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.get_json(force=True)
    img = base64_to_bgr(data['image'])
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    wm_keywords = ['rfans', 'channel', 'bot', 'tg', 'album', '电', '报', '群', '搜', '索', '@', 'fans', 't.me', 'vip']
    stamp_keywords = ['channel', 'album', 'bot', 'rfans']
    stamp_candidates = []

    if reader:
        res_normal = reader.readtext(img, contrast_ths=0.05, adjust_contrast=0.7)
        res_enhanced = reader.readtext(enhance_for_ocr(img), contrast_ths=0.02, text_threshold=0.25, low_text=0.15)
        all_res = res_normal + res_enhanced
        for (bbox, text, prob) in all_res:
            t_low = text.lower().replace(' ', '')
            if any(k in t_low for k in wm_keywords):
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
                bx, by = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                bw, bh = int(np.max(pts[:, 0]) - bx), int(np.max(pts[:, 1]) - by)
                if any(sk in t_low for sk in stamp_keywords) or bw > w * 0.08:
                    stamp_candidates.append((bx, by, bw, bh))

    if stamp_candidates:
        clusters = cluster_stamps(stamp_candidates, max_distance=int(min(h, w) * 0.20))
        for cluster in clusters:
            pts = []
            for (bx, by, bw, bh) in cluster:
                pts.extend([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]])
            (cx, cy), radius = cv2.minEnclosingCircle(np.array(pts, dtype=np.float32))
            cv2.circle(mask, (int(cx), int(cy)), int(radius * 1.25), 255, -1)

    dilation_px = int(data.get('dilation_px', 9))
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
        mask = cv2.dilate(mask, kernel, iterations=1)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return jsonify({
        'status': 'ok',
        'mask': bgr_to_base64(mask_bgr, 'png'),
        'stamp_count': len(stamp_candidates),
        'pixels_masked': int(np.count_nonzero(mask))
    })

@app.route('/api/inpaint', methods=['POST'])
def api_inpaint():
    data = request.get_json(force=True)
    img_bgr = base64_to_bgr(data['image'])
    h, w = img_bgr.shape[:2]

    mask_b64 = data.get('mask')
    if ',' in mask_b64:
        mask_b64 = mask_b64.split(',', 1)[1]
    raw_mask = cv2.imdecode(np.frombuffer(base64.b64decode(mask_b64), np.uint8), cv2.IMREAD_UNCHANGED)
    if len(raw_mask.shape) == 3 and raw_mask.shape[2] == 4:
        mask = raw_mask[:, :, 3]
    elif len(raw_mask.shape) == 3:
        mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
    else:
        mask = raw_mask
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    start_t = time.time()
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert('L')
    cleaned_pil = lama(img_pil, mask_pil)
    cleaned_bgr = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
    elapsed = round(time.time() - start_t, 3)

    return jsonify({
        'status': 'ok',
        'cleaned': bgr_to_base64(cleaned_bgr, 'jpg', 98),
        'elapsed_sec': elapsed,
        'dimensions': {'width': w, 'height': h}
    })

if __name__ == '__main__':
    # Start native cloudflared tunnel
    public_url = "http://127.0.0.1:5000"
    try:
        tunnel_proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://127.0.0.1:5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        for _ in range(25):
            line = tunnel_proc.stderr.readline()
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match:
                public_url = match.group(0)
                break
            time.sleep(0.3)
    except Exception as e:
        print(f"[!] Warning tunnel: {e}")

    print("\n" + "=" * 65)
    print("🎉 GOOGLE COLAB GPU SERVER BERHASIL AKTIF!")
    print(f"🔗 Public Tunnel URL: {public_url}")
    print("=" * 65)
    print("👉 Copy link 'https://....trycloudflare.com' di atas,")
    print("   lalu buka Web Studio Anda di menu:")
    print("   '⚡ GPU Acceleration' -> centang 'Google Colab GPU' -> Paste URL!")
    print("=" * 65 + "\n")

    app.run(host='0.0.0.0', port=5000)
