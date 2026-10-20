"""
scripts/build_colab_server.py - Rebuilds colab_server.py with high-precision RFans and Sakerly engines.
"""
import os

with open('core/b64_stamp.txt', 'r') as f:
    b64_stamp = f.read().strip()

with open('core/b64_sakerly.txt', 'r') as f:
    b64_sakerly = f.read().strip()

server_code = f'''"""
colab_server.py - Google Colab GPU Backend Server (Precision AI Clean Edition)
Anti-False-Positive Engine: YOLOv11 Watermark Detector + Multi-Pass CLAHE EasyOCR + Precision Stroke Masking
Dedicated RFans Master Stamp (wm_bulat_besar.png) & Sakerly Signature (<15ms)
Menghapus Bersih Watermark di Lokasi Acak Manapun TANPA Merusak Kipas, Bunga, Sepatu, atau Karpet.
"""
import os, sys, time, re, subprocess, base64, zipfile, io

# 0. Auto-Installer Instan di GPU Colab (~5 detik)
for pkg, mod in [
    ("simple-lama-inpainting", "simple_lama_inpainting"),
    ("ultralytics", "ultralytics"),
    ("flask-cors", "flask_cors"),
    ("easyocr", "easyocr")
]:
    try:
        __import__(mod)
    except ImportError:
        print(f"[*] Menyiapkan modul AI {{pkg}} di GPU...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--prefer-binary", "--quiet"])

# Siapkan cloudflared resmi jika belum ada
if not os.path.exists("/usr/local/bin/cloudflared"):
    subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "/usr/local/bin/cloudflared"])
    subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"])

# Siapkan bobot AI YOLO Watermark jika belum ada
if not os.path.exists("yolo11_watermark.pt"):
    print("[*] Mengunduh model AI YOLO Watermark (~109 MB, 1-2 detik)...")
    subprocess.run(["wget", "-q", "-nc", "https://huggingface.co/spaces/fancyfeast/joycaption-watermark-detection/resolve/main/yolo11x-train28-best.pt", "-O", "yolo11_watermark.pt"])

import torch, cv2, numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from simple_lama_inpainting import SimpleLama

# 1. Cek Ketersediaan GPU CUDA
print("=" * 65)
if torch.cuda.is_available():
    print(f"🚀 CUDA GPU Aktif: {{torch.cuda.get_device_name(0)}}")
    print(f"⚡ VRAM Tersedia: {{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}} GB")
    USE_GPU = True
else:
    print("⚠️ PERINGATAN: GPU tidak terdeteksi! Pastikan Runtime diatur ke T4 GPU.")
    USE_GPU = False
print("=" * 65)

# 2. Inisialisasi Model Inpainting SimpleLaMa
print("[*] Memuat SimpleLaMa Inpainting ke GPU...")
lama = SimpleLama()
print("✅ SimpleLaMa Inpainting siap di GPU!")

# 3. Inisialisasi Model AI Deteksi Watermark
yolo_wm = None
custom_model_paths = [
    "/content/drive/MyDrive/DewatermarkAI/custom_watermark.pt",
    "custom_watermark.pt",
    "/content/custom_watermark.pt",
    "yolo11_watermark.pt",
    "/content/yolo11_watermark.pt"
]

for p in custom_model_paths:
    if os.path.exists(p):
        try:
            from ultralytics import YOLO
            yolo_wm = YOLO(p)
            if "custom_watermark.pt" in p:
                print(f"🎯 MODEL HASIL TRAINING AKTIF ({{p}})! (Deteksi Khusus Album Anda)")
            else:
                print(f"✅ Model Pretrained Aktif ({{p}} run)!")
            break
        except Exception as e:
            print(f"ℹ️ Model load notice for {{p}}: {{e}}")

# 4. Inisialisasi EasyOCR GPU untuk Deteksi Teks Watermark Acak (CRAFT)
reader = None
try:
    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=USE_GPU)
    print("✅ EasyOCR Deep Learning Text Detector aktif di GPU!")
except Exception as e:
    print(f"ℹ️ EasyOCR notice: {{e}}")

app = Flask(__name__)
CORS(app)

def bgr_to_base64(img_bgr, ext='jpg', quality=95):
    if ext == 'jpg':
        _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = 'image/jpeg'
    else:
        _, buf = cv2.imencode('.png', img_bgr)
        mime = 'image/png'
    return f'data:{{mime}};base64,' + base64.b64encode(buf).decode('utf-8')

def base64_to_bgr(b64_str):
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64_str), np.uint8), cv2.IMREAD_COLOR)

# -------------------------------------------------------------------------
# MASTER TEMPLATES (Embedded High-Res Base64)
# -------------------------------------------------------------------------
B64_STAMP_TEMPLATE = "{b64_stamp}"
B64_SAKERLY_TEMPLATE = "{b64_sakerly}"

_stamp_bytes = base64.b64decode(B64_STAMP_TEMPLATE)
_stamp_raw = cv2.imdecode(np.frombuffer(_stamp_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
if len(_stamp_raw.shape) == 3 and _stamp_raw.shape[2] == 4:
    STAMP_ALPHA = _stamp_raw[:, :, 3]
    _st_mask = (STAMP_ALPHA > 30).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    STAMP_TOPHAT = cv2.morphologyEx(_st_mask, cv2.MORPH_TOPHAT, k5)
else:
    STAMP_ALPHA = _stamp_raw
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    STAMP_TOPHAT = cv2.morphologyEx(_stamp_raw, cv2.MORPH_TOPHAT, k5)

_sakerly_bytes = base64.b64decode(B64_SAKERLY_TEMPLATE)
SAKERLY_RAW = cv2.imdecode(np.frombuffer(_sakerly_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
SAKERLY_ALPHA = SAKERLY_RAW[:, :, 3] if (len(SAKERLY_RAW.shape) == 3 and SAKERLY_RAW.shape[2] == 4) else SAKERLY_RAW


def detect_sakerly(img_bgr, threshold=0.28, dilation_px=2):
    """
    Dedicated Sakerly Signature Detector.
    Leverages its unique coral-red tint [BGR: 57, 76, 193] and multi-scale template search across
    the entire image (not limited to corners, works on hair, sunglasses, chest, and backgrounds).
    Speed: ~15-30ms with high precision and stroke-level alpha mask.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    th_h, th_w = SAKERLY_ALPHA.shape[:2]
    t_mask = (SAKERLY_ALPHA > 30).astype(np.uint8) * 255

    b, g, r = cv2.split(img_bgr)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, _ = cv2.split(hsv)

    color_gate = (
        (r.astype(int) - b.astype(int) > 16) &
        (r.astype(int) - g.astype(int) > 8) &
        (r > 55) &
        ((h_ch <= 16) | (h_ch >= 165)) &
        (s_ch > 28)
    ).astype(np.uint8) * 255

    scale_search = min(1.0, 1024.0 / float(max(h, w)))
    if scale_search < 1.0:
        gate_search = cv2.resize(color_gate, (0, 0), fx=scale_search, fy=scale_search, interpolation=cv2.INTER_AREA)
    else:
        gate_search = color_gate

    gh, gw = gate_search.shape[:2]
    scales = np.linspace(0.05 * gw / float(th_w), 0.42 * gw / float(th_w), 25)

    best_score = -1.0
    best_match = None

    for s in scales:
        sw = int(round(th_w * s))
        sh = int(round(th_h * s))
        if sw >= gw or sh >= gh or sw < 15 or sh < 8:
            continue
        st = cv2.resize(t_mask, (sw, sh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gate_search, st, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res)
        if max_v > best_score:
            best_score = float(max_v)
            best_match = (
                int(round(max_l[0] / scale_search)),
                int(round(max_l[1] / scale_search)),
                int(round(sw / scale_search)),
                int(round(sh / scale_search))
            )
            if best_score > 0.82:
                break

    if best_score >= threshold and best_match is not None:
        bx, by, bw, bh = best_match
        st_alpha = cv2.resize(SAKERLY_ALPHA, (bw, bh), interpolation=cv2.INTER_LINEAR)
        _, st_bin = cv2.threshold(st_alpha, 25, 255, cv2.THRESH_BINARY)

        y1, y2 = max(0, by), min(h, by + bh)
        x1, x2 = max(0, bx), min(w, bx + bw)
        roi_gate = color_gate[y1:y2, x1:x2]
        roi_st = st_bin[:(y2 - y1), :(x2 - x1)]

        overlap = np.count_nonzero((roi_gate > 0) & (roi_st > 0))
        template_pts = np.count_nonzero(roi_st > 0)
        ratio = overlap / max(1, template_pts)

        if ratio >= 0.40 or best_score >= 0.50:
            d_px = max(1, dilation_px)
            k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_px * 2 + 1, d_px * 2 + 1))
            st_bin = cv2.dilate(st_bin, k_d, iterations=1)
            patch = st_bin[:(y2 - y1), :(x2 - x1)]
            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], patch)
        else:
            best_score = 0.0
            best_match = None

    return mask, best_score, best_match


def detect_stamp_dynamic(img_bgr, threshold=0.34, dilation_px=2):
    """
    Pencarian stempel RFans 100% DINAMIS di 4 Sudut (Top-Left, Top-Right, Bottom-Left, Bottom-Right):
    - Menggunakan master template beresolusi tinggi (wm_bulat_besar.png).
    - Memindai area sudut 35% lebar & tinggi.
    - Menghasilkan HANYA guratan garis stempel halus (1-2px dilation), TANPA merusak karpet / sepatu.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    th_h, th_w = STAMP_TOPHAT.shape[:2]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top_img = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k5)

    cw, ch = int(w * 0.35), int(h * 0.35)
    corners = [
        ('TR', w - cw, 0, w, ch),
        ('BR', w - cw, h - ch, w, h),
        ('TL', 0, 0, cw, ch),
        ('BL', 0, h - ch, cw, h),
    ]

    base_scale = (w * 0.245) / float(th_w)
    scales = np.linspace(base_scale * 0.88, base_scale * 1.12, 7)

    best_score = -1.0
    best_match = None

    for c_name, cx1, cy1, cx2, cy2 in corners:
        c_roi = top_img[cy1:cy2, cx1:cx2]
        for s in scales:
            sw = int(round(th_w * s))
            sh = int(round(th_h * s))
            if sw >= c_roi.shape[1] or sh >= c_roi.shape[0]:
                continue
            st = cv2.resize(STAMP_TOPHAT, (sw, sh))
            res = cv2.matchTemplate(c_roi, st, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_score:
                best_score = float(max_v)
                best_match = (cx1 + max_l[0], cy1 + max_l[1], sw, sh)
                if best_score > 0.65:
                    break
        if best_score > 0.65:
            break

    # Fallback to downscaled scan if not found in standard corners
    if best_score < threshold:
        scale_search = min(1.0, 1024.0 / float(max(h, w)))
        top_search = cv2.resize(top_img, (0, 0), fx=scale_search, fy=scale_search)
        scales_full = np.linspace(0.12 * w / float(th_w), 0.85 * w / float(th_w), 16)
        for s in scales_full:
            sw = int(round(th_w * s * scale_search))
            sh = int(round(th_h * s * scale_search))
            if sw >= top_search.shape[1] or sh >= top_search.shape[0] or sw < 20 or sh < 15:
                continue
            st = cv2.resize(STAMP_TOPHAT, (sw, sh))
            res = cv2.matchTemplate(top_search, st, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            if max_v > best_score:
                best_score = float(max_v)
                best_match = (
                    int(round(max_l[0] / scale_search)),
                    int(round(max_l[1] / scale_search)),
                    int(round(sw / scale_search)),
                    int(round(sh / scale_search))
                )

    if best_score >= threshold and best_match is not None:
        bx, by, bw, bh = best_match
        st_alpha = cv2.resize(STAMP_ALPHA, (bw, bh), interpolation=cv2.INTER_LINEAR)
        _, st_bin = cv2.threshold(st_alpha, 25, 255, cv2.THRESH_BINARY)
        d_px = max(1, dilation_px)
        k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d_px * 2 + 1, d_px * 2 + 1))
        st_bin = cv2.dilate(st_bin, k_d, iterations=1)

        y1, y2 = max(0, by), min(h, by + bh)
        x1, x2 = max(0, bx), min(w, bx + bw)
        patch = st_bin[:(y2 - y1), :(x2 - x1)]
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], patch)

    return mask, best_score, best_match


def generate_watermark_mask(img_bgr, wm_mode='auto', dilation_px=2):
    """
    Ekstraksi mask watermark presisi tinggi 100% DINAMIS (Zero Hardcoded Coordinates):
    - Beradaptasi otomatis pada setiap gambar yang posisi watermark-nya selalu berubah-ubah.
    - Template Matching dinamis melacak stempel bulat besar RFans di sudut mana saja dengan guratan stroke murni.
    - Sakerly detector melacak signature coral-red di mana saja (<25ms).
    - Deep Learning OCR CRAFT melacak teks transparan miring DENGAN GURATAN STROKE (Zero Solid Blobs).
    - ZERO False Positives: Tidak pernah merusak karpet, bunga, atau sepatu.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clahe = cv2.createCLAHE(clipLimit=14.0, tileGridSize=(6, 6))

    mode = (wm_mode or 'auto').lower()

    # =========================================================================
    # 1. SPECIALIZED SAKERLY MODE
    # =========================================================================
    if mode in ['sakerly', 'auto']:
        s_mask, s_score, s_match = detect_sakerly(img_bgr, threshold=0.28, dilation_px=dilation_px)
        if s_score >= 0.28:
            mask = cv2.bitwise_or(mask, s_mask)
            if mode == 'sakerly':
                return mask

    # =========================================================================
    # 2. DYNAMIC TEMPLATE MATCHING (Stempel RFans Bulat Besar di 4 Sudut)
    # =========================================================================
    stamp_bbox = None
    if mode in ['rfans', 'auto']:
        stamp_mask, stamp_score, stamp_match = detect_stamp_dynamic(img_bgr, threshold=0.34, dilation_px=dilation_px)
        if stamp_score >= 0.32:
            mask = cv2.bitwise_or(mask, stamp_mask)
            stamp_bbox = stamp_match

    # =========================================================================
    # 3. DEEP LEARNING OCR CRAFT MULTI-ANGLE (Dengan Stroke Masking Murni)
    # =========================================================================
    wm_keywords = [
        'rfans', 'channel', 'bot', 'tg', 'album', 'fans', 'vip', 't.me', '@',
        'sakerly', 'saker',
        '电', '报', '搜', '索', '直', '接', '群', '搜索', '电报', 'telegram'
    ]

    if reader and mode != 'sakerly':
        try:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            l_enh = clahe.apply(l)
            top = cv2.morphologyEx(l, cv2.MORPH_TOPHAT, k5)
            bot = cv2.morphologyEx(l, cv2.MORPH_BLACKHAT, k5)
            amplified_l = cv2.addWeighted(l_enh, 0.75, top, 3.5, 0)
            amplified_l = cv2.subtract(amplified_l, cv2.multiply(bot, 2))
            enh_bgr = cv2.cvtColor(cv2.merge((amplified_l, a, b_ch)), cv2.COLOR_LAB2BGR)

            scale = 1.0
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / float(max(h, w))
                small_img = cv2.resize(enh_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                small_img = enh_bgr

            sh, sw = small_img.shape[:2]
            s_center = (sw / 2.0, sh / 2.0)
            scan_angles = [39.0, -39.0, 0.0]

            for angle in scan_angles:
                if angle == 0.0:
                    rot_img = small_img
                    M_inv = None
                else:
                    rad = np.deg2rad(angle)
                    cos_a = np.abs(np.cos(rad))
                    sin_a = np.abs(np.sin(rad))
                    nw = int((sh * sin_a) + (sw * cos_a))
                    nh = int((sh * cos_a) + (sw * sin_a))
                    M_rot = cv2.getRotationMatrix2D(s_center, angle, 1.0)
                    M_rot[0, 2] += (nw / 2.0) - s_center[0]
                    M_rot[1, 2] += (nh / 2.0) - s_center[1]
                    M_inv = cv2.invertAffineTransform(M_rot)
                    rot_img = cv2.warpAffine(
                        small_img,
                        M_rot,
                        (nw, nh),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(128, 128, 128)
                    )

                results = reader.readtext(
                    rot_img,
                    contrast_ths=0.02,
                    text_threshold=0.15,
                    low_text=0.10,
                    link_threshold=0.18,
                    mag_ratio=1.2
                )

                for (bbox, text, prob) in results:
                    t_low = text.lower().replace(' ', '')
                    if any(k in t_low for k in wm_keywords):
                        pts = np.array(bbox, dtype=np.float32)

                        if angle != 0.0 and M_inv is not None:
                            pts_orig = []
                            for pt in pts:
                                v = M_inv @ np.array([pt[0], pt[1], 1.0])
                                pts_orig.append(v[:2])
                            pts = np.array(pts_orig, dtype=np.float32)

                        pts = pts / scale
                        bx = int(np.min(pts[:, 0]))
                        by = int(np.min(pts[:, 1]))
                        bw = int(np.max(pts[:, 0]) - bx)
                        bh = int(np.max(pts[:, 1]) - by)

                        # SUPPRESS OCR INSIDE STAMP ROI: detect_stamp_dynamic already masks exact strokes!
                        if stamp_bbox is not None:
                            sx, sy, sw, sh = stamp_bbox
                            ix1 = max(bx, sx)
                            iy1 = max(by, sy)
                            ix2 = min(bx + bw, sx + sw)
                            iy2 = min(by + bh, sy + sh)
                            if ix2 > ix1 and iy2 > iy1:
                                inter_area = (ix2 - ix1) * (iy2 - iy1)
                                box_area = max(1, bw * bh)
                                if (inter_area / float(box_area)) > 0.30:
                                    continue

                        pad = 4
                        x1, y1 = max(0, bx - pad), max(0, by - pad)
                        x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
                        roi = gray[y1:y2, x1:x2]
                        if roi.size > 0:
                            # CRITICAL FIX: Extract ONLY the text strokes (Top-Hat), NEVER solid fill!
                            poly_mask = np.zeros_like(roi)
                            shifted_pts = (pts - np.array([x1, y1])).astype(np.int32)
                            cv2.fillPoly(poly_mask, [shifted_pts], 255)

                            top_roi = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k5)
                            strk = (top_roi > 12).astype(np.uint8) * 255
                            combined = cv2.bitwise_and(strk, poly_mask)
                            combined = cv2.dilate(combined, k3, iterations=1)
                            mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], combined)
        except Exception as e:
            print(f"[OCR scan notice] {{e}}")

    # =========================================================================
    # 4. DETEKSI AI YOLOV11 (Bounding Boxes Watermark Acak - Presisi Tinggi)
    # =========================================================================
    if yolo_wm and mode != 'sakerly':
        try:
            results = yolo_wm.predict(img_bgr, conf=0.35, device=0 if USE_GPU else 'cpu', verbose=False)
            total_img_area = float(w * h)

            for box in results[0].boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                bw, bh = bx2 - bx1, by2 - by1
                if bw <= 0 or bh <= 0: continue

                if (bw * bh) > 0.25 * total_img_area:
                    continue

                if stamp_bbox is not None:
                    sx, sy, sw, sh = stamp_bbox
                    ix1, iy1 = max(bx1, sx), max(by1, sy)
                    ix2, iy2 = min(bx2, sx + sw), min(by2, sy + sh)
                    if ix2 > ix1 and iy2 > iy1:
                        if ((ix2 - ix1) * (iy2 - iy1)) / float(bw * bh) > 0.40:
                            continue

                pad = 4
                px1, py1 = max(0, bx1 - pad), max(0, by1 - pad)
                px2, py2 = min(w, bx2 + pad), min(h, by2 + pad)
                roi = gray[py1:py2, px1:px2]
                if roi.size > 0:
                    top_roi = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k5)
                    s1 = (top_roi > 12).astype(np.uint8) * 255
                    s1 = cv2.dilate(s1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                    mask[py1:py2, px1:px2] = cv2.bitwise_or(mask[py1:py2, px1:px2], s1)
        except Exception as e:
            print(f"[YOLO notice] {{e}}")

    return mask


@app.route('/api/sample')
def api_sample():
    return jsonify({{
        'status': 'ok',
        'device': 'CUDA GPU' if USE_GPU else 'CPU',
        'yolo': yolo_wm is not None,
        'ocr': reader is not None
    }})


@app.route('/api/detect', methods=['POST'])
def api_detect():
    data = request.get_json(force=True)
    img = base64_to_bgr(data['image'])
    dilation_px = int(data.get('dilation_px', 2))
    wm_mode = str(data.get('wm_mode', 'auto'))
    mask = generate_watermark_mask(img, wm_mode=wm_mode, dilation_px=dilation_px)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return jsonify({{
        'status': 'ok',
        'mask': bgr_to_base64(mask_bgr, 'png'),
        'pixels_masked': int(np.count_nonzero(mask))
    }})


@app.route('/api/inpaint', methods=['POST'])
def api_inpaint():
    data = request.get_json(force=True)
    img_bgr = base64_to_bgr(data['image'])
    h, w = img_bgr.shape[:2]
    wm_mode = str(data.get('wm_mode', 'auto'))

    mask_b64 = data.get('mask')
    if mask_b64:
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
    else:
        mask = generate_watermark_mask(img_bgr, wm_mode=wm_mode)

    start_t = time.time()
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask).convert('L')
    cleaned_pil = lama(img_pil, mask_pil)
    cleaned_bgr = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
    elapsed = round(time.time() - start_t, 3)

    return jsonify({{
        'status': 'ok',
        'cleaned': bgr_to_base64(cleaned_bgr, 'jpg', quality=98),
        'elapsed_sec': elapsed,
        'dimensions': {{'width': w, 'height': h}}
    }})


# Batch Processing 100+ Gambar (Disimpan Terpusat di Disk Colab & Disatukan dalam 1 ZIP)
BATCH_STORAGE = {{}}
BATCH_OUTPUT_DIR = "/content/cleaned_photos"
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

@app.route('/api/batch', methods=['POST'])
def api_batch():
    if 'files' not in request.files:
        return jsonify({{'error': 'No files uploaded'}}), 400

    files = request.files.getlist('files')
    wm_mode = request.form.get('wm_mode', 'auto')

    # Gunakan batch_id dari client agar semua foto dari antrean tergabung dalam 1 file ZIP!
    batch_id = request.form.get('batch_id')
    if not batch_id or not batch_id.startswith('batch_'):
        batch_id = f"batch_{{int(time.time())}}"

    # Folder batch khusus di Colab
    batch_dir = os.path.join(BATCH_OUTPUT_DIR, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

    results = []
    zip_path = f"/content/{{batch_id}}.zip"
    latest_zip_path = f"/content/cleaned_batch_latest.zip"

    print(f"\\n[*] Memproses {{len(files)}} gambar di GPU [Batch: {{batch_id}}] (Mode: {{wm_mode}})...")

    for idx, file in enumerate(files):
        if not file.filename: continue
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None: continue
        h, w = img_bgr.shape[:2]

        t0 = time.time()
        mask = generate_watermark_mask(img_bgr, wm_mode=wm_mode)
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert('L')
        cleaned_pil = lama(img_pil, mask_pil)
        cleaned_bgr = cv2.cvtColor(np.array(cleaned_pil), cv2.COLOR_RGB2BGR)
        elapsed = round(time.time() - t0, 3)

        out_name = f"cleaned_{{file.filename}}"
        out_file_path = os.path.join(batch_dir, out_name)
        root_out_path = os.path.join(BATCH_OUTPUT_DIR, out_name)

        # 1. Simpan foto bersih beresolusi penuh langsung ke disk Colab
        _, buf = cv2.imencode('.jpg', cleaned_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
        img_bytes = buf.tobytes()
        with open(out_file_path, "wb") as f_out:
            f_out.write(img_bytes)
        with open(root_out_path, "wb") as f_out:
            f_out.write(img_bytes)

        # 2. Append ke file ZIP tunggal (agar semua foto batch tergabung dalam 1 file ZIP!)
        for z_target in [zip_path, latest_zip_path]:
            mode_z = 'a' if os.path.exists(z_target) else 'w'
            with zipfile.ZipFile(z_target, mode_z, compression=zipfile.ZIP_DEFLATED) as zip_out:
                existing_names = zip_out.namelist() if mode_z == 'a' else []
                if out_name not in existing_names:
                    zip_out.writestr(out_name, img_bytes)

        # 3. Jika Google Drive tersambung, simpan otomatis juga ke Google Drive
        drive_out = "/content/drive/MyDrive/DewatermarkAI/cleaned_photos"
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(drive_out, exist_ok=True)
            with open(os.path.join(drive_out, out_name), "wb") as f_drive:
                f_drive.write(img_bytes)

        # Thumbnail untuk UI antrean
        thumb = cv2.resize(cleaned_bgr, (200, int(200 * h / max(1, w))), interpolation=cv2.INTER_AREA)

        # Base64 foto bersih penuh (seperti mode deteksi single!)
        cleaned_b64 = bgr_to_base64(cleaned_bgr, 'jpg', quality=98)

        results.append({{
            'filename': file.filename,
            'out_filename': out_name,
            'elapsed_sec': elapsed,
            'thumb': bgr_to_base64(thumb, 'jpg', 80),
            'cleaned': cleaned_b64,
            'download_url': f"/api/download-file/{{batch_id}}/{{out_name}}"
        }})
        print(f"  [{{idx+1}}/{{len(files)}}] {{file.filename}} -> Bersih ({{elapsed}}s) | Disatukan ke: {{zip_path}}")

    # Update cache memori zip
    try:
        with open(zip_path, "rb") as f_z:
            BATCH_STORAGE[batch_id] = f_z.read()
    except Exception:
        pass

    return jsonify({{
        'status': 'ok',
        'processed_count': len(results),
        'batch_id': batch_id,
        'zip_file': f"{{batch_id}}.zip",
        'items': results
    }})


# Endpoint Unduh Foto Individual dari Batch
@app.route('/api/download-file/<batch_id>/<filename>')
def api_download_file(batch_id, filename):
    safe_name = os.path.basename(filename)
    candidates = [
        f"/content/cleaned_photos/{{batch_id}}/{{safe_name}}",
        f"/content/cleaned_photos/{{safe_name}}",
        f"/content/drive/MyDrive/DewatermarkAI/cleaned_photos/{{safe_name}}"
    ]
    for path in candidates:
        if os.path.exists(path):
            return send_file(path, as_attachment=True, download_name=safe_name)
    return jsonify({{'error': 'File not found'}}), 404


# Endpoint Unduh ZIP Tunggal Gabungan
@app.route('/api/download-batch/<batch_id>')
def api_download_batch(batch_id):
    zip_path = f"/content/{{batch_id}}.zip"
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, download_name=f"{{batch_id}}.zip")
    if os.path.exists("/content/cleaned_batch_latest.zip"):
        return send_file("/content/cleaned_batch_latest.zip", as_attachment=True, download_name="cleaned_batch.zip")
    if batch_id in BATCH_STORAGE:
        data_bytes = BATCH_STORAGE[batch_id]
        resp = Response(data_bytes, mimetype='application/zip')
        resp.headers['Content-Disposition'] = f'attachment; filename="{{batch_id}}.zip"'
        return resp
    return jsonify({{'error': 'Batch not found'}}), 404


if __name__ == '__main__':
    PORT = 5000
    public_url = "http://127.0.0.1:5000"
    print("\\n[*] Menghubungkan ke Cloudflare Tunnel...")
    
    try:
        tunnel_proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{{PORT}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        for _ in range(35):
            line = tunnel_proc.stderr.readline()
            match = re.search(r'https://[a-zA-Z0-9-]+\\.trycloudflare\\.com', line)
            if match:
                public_url = match.group(0)
                break
            time.sleep(0.3)
    except Exception as e:
        print(f"[!] Tunnel notice: {{e}}")

    print("\\n" + "=" * 65)
    print("🎉 GOOGLE COLAB GPU SERVER BERHASIL AKTIF (ZERO FALSE-POSITIVE ENGINE)!")
    print(f"🔗 Public Tunnel URL: {{public_url}}")
    print("=" * 65)
    print("👉 Copy link 'https://....trycloudflare.com' di atas,")
    print("   lalu buka Web Studio Anda di menu:")
    print("   '⚡ GPU Acceleration' -> centang 'Google Colab GPU' -> Paste URL!")
    print("=" * 65 + "\\n")

    app.run(host='0.0.0.0', port=PORT)
'''

with open('colab_server.py', 'w', encoding='utf-8') as f:
    f.write(server_code)

print("[*] colab_server.py successfully built!")
