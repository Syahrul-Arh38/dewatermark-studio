"""
colab_server.py - Google Colab GPU Backend Server for Dewatermark Studio
Run this file directly in Google Colab (with T4 GPU runtime enabled):

1. Di Google Colab, pastikan Runtime diatur ke T4 GPU:
   Menu 'Runtime' -> 'Change runtime type' -> pilih 'T4 GPU'.

2. Jalankan:
   !pip install -q easyocr simple-lama-inpainting opencv-python-headless pillow flask flask-cors pycloudflared

3. Jalankan script ini:
   python colab_server.py

Script ini akan otomatis mengaktifkan SimpleLaMa & EasyOCR dengan akselerasi CUDA GPU
dan membagikan public link (https://xxxx.trycloudflare.com) untuk di-paste ke Web Studio Anda!
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from pycloudflared import try_cloudflare

# 1. Cek Ketersediaan GPU CUDA
print("=" * 60)
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"🚀 CUDA GPU Terdeteksi: {device_name}")
    print(f"⚡ VRAM Tersedia: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    USE_GPU = True
else:
    print("⚠️ PERINGATAN: GPU tidak terdeteksi! Pastikan Anda memilih Runtime -> T4 GPU di Colab.")
    USE_GPU = False
print("=" * 60)

# 2. Inisialisasi Model AI (EasyOCR + SimpleLaMa)
print("[*] Menginisialisasi Model AI di GPU...")
try:
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=USE_GPU)
except Exception as e:
    print(f"[!] EasyOCR notice: {e}")
    reader = None

try:
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()
    print("[*] Model LaMa Inpainting berhasil dimuat ke GPU!")
except Exception as e:
    print(f"[!] SimpleLama error: {e}")
    lama = None

app = Flask(__name__)
CORS(app)

import base64

def bgr_to_base64(img_bgr, ext="jpg", quality=95):
    if ext == "jpg":
        _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = "image/jpeg"
    else:
        _, buf = cv2.imencode('.png', img_bgr)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('utf-8')}"

def base64_to_bgr(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    data = base64.b64decode(b64_str)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

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

@app.route('/api/sample')
def api_sample():
    return jsonify({'status': 'ok', 'device': 'CUDA GPU' if USE_GPU else 'CPU'})

@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.get_json(force=True)
    img_bgr = base64_to_bgr(data['image'])
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    wm_keywords = ['rfans', 'channel', 'bot', 'tg', 'album', '电', '报', '群', '搜', '索', '@']
    stamp_keywords = ['channel', 'album', 'bot', 'rfans']
    stamp_candidates = []

    if reader:
        res_normal = reader.readtext(img_bgr, contrast_ths=0.05, adjust_contrast=0.7)
        res_enhanced = reader.readtext(enhance_for_ocr(img_bgr), contrast_ths=0.03, text_threshold=0.4, low_text=0.2)
        for (bbox, text, prob) in (res_normal + res_enhanced):
            t_low = text.lower().replace(" ", "")
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

    # Decode mask
    mask_b64 = data.get('mask')
    if "," in mask_b64:
        mask_b64 = mask_b64.split(",", 1)[1]
    mask_data = base64.b64decode(mask_b64)
    raw_mask = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_UNCHANGED)
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
    if lama:
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert("L")
        cleaned_pil = lama(img_pil, mask_pil)
        cleaned_bgr = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
    else:
        cleaned_bgr = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)

    elapsed = round(time.time() - start_t, 3)
    return jsonify({
        'status': 'ok',
        'cleaned': bgr_to_base64(cleaned_bgr, 'jpg', quality=98),
        'elapsed_sec': elapsed,
        'dimensions': {'width': w, 'height': h}
    })

if __name__ == '__main__':
    PORT = 5000
    print("\n[*] Membuka Cloudflare Tunnel untuk koneksi publik...")
    try:
        tunnel = try_cloudflare(port=PORT)
        print("\n" + "=" * 65)
        print(f"🎉 GOOGLE COLAB GPU SERVER BERHASIL AKTIF!")
        print(f"🔗 Public Tunnel URL: {tunnel.tunnel}")
        print("=" * 65)
        print("👉 Copy link 'https://....trycloudflare.com' di atas,")
        print("   lalu buka Web Studio Anda dan paste ke menu:")
        print("   '⚡ GPU Acceleration' -> centang 'Google Colab GPU' -> Paste URL!")
        print("=" * 65 + "\n")
    except Exception as e:
        print(f"[!] Cloudflare tunnel warning: {e}")

    app.run(host='0.0.0.0', port=PORT)
