import json, os

with open('colab_server.py', 'r', encoding='utf-8') as f:
    server_code = f.read()

# CELL 2: Training & Auto-Save to Google Drive
trainer_code = '''# =========================================================================
# CELL 2: AUTO-TRAIN WATERMARK AI (TRAINING 1 KALI UNTUK SELAMANYA)
# =========================================================================
# Model AI (.pt) akan otomatis disimpan permanen di Google Drive Anda:
# Google Drive/DewatermarkAI/custom_watermark.pt
# Sehingga BESOK, LUSA, ATAU KAPAN PUN ANDA BUKA LAGI, TIDAK PERLU TRAIN LAGI!
# =========================================================================

import os, sys, glob, shutil, random, base64
import cv2, numpy as np

# Ubah ke True jika suatu hari nanti Anda ingin melatih ulang dengan sampel baru
FORCE_RETRAIN = False

DRIVE_DIR = "/content/drive/MyDrive/DewatermarkAI"
DRIVE_MODEL = os.path.join(DRIVE_DIR, "custom_watermark.pt")
LOCAL_MODEL = "custom_watermark.pt"

# Cek apakah model sudah pernah dilatih sebelumnya
already_trained = False
if os.path.exists(DRIVE_MODEL):
    print("=" * 65)
    print("🎉 MODEL CUSTOM DITEMUKAN DI GOOGLE DRIVE!")
    print(f"📁 Path: {DRIVE_MODEL}")
    shutil.copy(DRIVE_MODEL, LOCAL_MODEL)
    print("⚡ Model langsung siap digunakan TANPA PERLU TRAINING ULANG!")
    print("=" * 65)
    already_trained = True
elif os.path.exists(LOCAL_MODEL):
    print("=" * 65)
    print("🎉 MODEL CUSTOM DITEMUKAN DI LOCAL DIRECTORY!")
    print("⚡ Langsung siap digunakan TANPA PERLU TRAINING ULANG!")
    print("=" * 65)
    already_trained = True

if already_trained and not FORCE_RETRAIN:
    print("\\n👉 Model sudah aktif! Langsung lewati cell ini dan jalankan CELL 3 (Start Server).\\n")
else:
    print("=" * 65)
    print("🚀 MEMULAI AUTO-TRAINING KHUSUS UNTUK WATERMARK ALBUM ANDA...")
    print("⏳ Estimasi waktu: ~2-3 menit di GPU T4 (Gratis)...")
    print("=" * 65)

    # Folder Sampel Watermark Pengguna di Google Drive (Tersimpan Permanen):
    DRIVE_SAMPLES_DIR = os.path.join(DRIVE_DIR, "watermark_samples")
    LOCAL_SAMPLES_DIR = "/content/watermark_samples"
    os.makedirs(DRIVE_SAMPLES_DIR, exist_ok=True)
    os.makedirs(LOCAL_SAMPLES_DIR, exist_ok=True)
    os.makedirs(DRIVE_DIR, exist_ok=True)

    DATASET_DIR = "/content/dataset_wm"
    IMG_TRAIN = os.path.join(DATASET_DIR, "images/train")
    IMG_VAL = os.path.join(DATASET_DIR, "images/val")
    LBL_TRAIN = os.path.join(DATASET_DIR, "labels/train")
    LBL_VAL = os.path.join(DATASET_DIR, "labels/val")

    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        os.makedirs(d, exist_ok=True)

    # 1. Master Template (Stempel Bulat Besar RFans & Teks Strip Diagonal & Sakerly)
    with open('core/b64_stamp.txt', 'r') as _f_s:
        B64_STAMP = _f_s.read().strip()
    with open('core/b64_sakerly.txt', 'r') as _f_sk:
        B64_SAKERLY = _f_sk.read().strip()

    stamp_raw = cv2.imdecode(np.frombuffer(base64.b64decode(B64_STAMP), np.uint8), cv2.IMREAD_UNCHANGED)
    stamp_img = stamp_raw[:, :, 3] if (len(stamp_raw.shape) == 3 and stamp_raw.shape[2] == 4) else stamp_raw

    sakerly_raw = cv2.imdecode(np.frombuffer(base64.b64decode(B64_SAKERLY), np.uint8), cv2.IMREAD_UNCHANGED)
    sakerly_img = sakerly_raw[:, :, 3] if (len(sakerly_raw.shape) == 3 and sakerly_raw.shape[2] == 4) else sakerly_raw

    # Buat template teks strip diagonal (@RFANS, TG@..., 电报直接搜索, t.me/SakerLy)
    text_strip_templates = []
    for txt in ["TG@RFans_Album_bot", "@RFANS CHANNEL", "电报直接搜索: @RFANS", "t.me/SakerLy", "t.me/album_cosplay"]:
        t_can = np.zeros((32, 280), dtype=np.uint8)
        cv2.putText(t_can, txt, (28, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 2, cv2.LINE_AA)
        cv2.circle(t_can, (14, 16), 10, 255, 2)
        text_strip_templates.append(('text_strip', t_can))

    # Kumpulkan semua template watermark (stempel + sakerly + teks strip + sampel user dari Google Drive)
    watermark_templates = [('stamp', stamp_img), ('sakerly', sakerly_img)] + text_strip_templates

    drive_samples = glob.glob(os.path.join(DRIVE_SAMPLES_DIR, "*.*"))
    local_samples = glob.glob(os.path.join(LOCAL_SAMPLES_DIR, "*.*"))
    all_user_samples = list(set(drive_samples + local_samples))

    print(f"📁 Memeriksa folder sampel Google Drive: {DRIVE_SAMPLES_DIR}")
    print(f"   Ditemukan {len(all_user_samples)} file crop watermark yang Anda kumpulkan.")

    for f in all_user_samples:
        u_raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if u_raw is None: continue
        # Ekstraksi otomatis bentuk huruf / watermark dari crop gambar:
        if len(u_raw.shape) == 3 and u_raw.shape[2] == 4:
            # Jika PNG transparan, gunakan langsung channel alphanya
            extracted_mask = u_raw[:, :, 3]
        else:
            u_gray = cv2.cvtColor(u_raw, cv2.COLOR_BGR2GRAY) if len(u_raw.shape) == 3 else u_raw
            # Ekstraksi guratan huruf kontras tinggi
            k_u = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            top_u = cv2.morphologyEx(u_gray, cv2.MORPH_TOPHAT, k_u)
            _, stroke_u = cv2.threshold(top_u, 10, 255, cv2.THRESH_BINARY)
            if np.count_nonzero(stroke_u) < 40:
                _, stroke_u = cv2.threshold(u_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            extracted_mask = stroke_u

        watermark_templates.append(('user_crop', extracted_mask))
        print(f"   [+] Sampel berhasil diekstrak huruf/bentuknya: {os.path.basename(f)}")

    # 2. Generator Sintetis 400 Gambar Latihan (Dengan Realistis Skin/Cloth & Sampel Negatif)
    print("[*] Membuat 400 data latihan cerdas (termasuk sampel negatif kulit/kain bersih agar tidak overfit)...")
    TOTAL_SAMPLES = 400
    for idx in range(TOTAL_SAMPLES):
        is_val = (idx % 5 == 0)
        img_out = IMG_VAL if is_val else IMG_TRAIN
        lbl_out = LBL_VAL if is_val else LBL_TRAIN

        H, W = 640, 640
        bg_type = random.choice(['skin_tone', 'cloth_dark', 'cloth_red', 'texture_wood', 'gradient'])
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        if bg_type == 'skin_tone':
            # Simulasi warna kulit (fair skin / warm tone)
            base_col = np.array([random.randint(170, 230), random.randint(185, 240), random.randint(210, 255)], dtype=np.uint8)
            canvas[:] = base_col
            noise = np.random.randint(-12, 12, (H, W, 3))
            canvas = np.clip(canvas.astype(int) + noise, 0, 255).astype(np.uint8)
        elif bg_type == 'cloth_red':
            # Simulasi kain kimono / baju merah
            canvas[:] = [random.randint(10, 40), random.randint(10, 40), random.randint(140, 230)]
        elif bg_type == 'cloth_dark':
            # Simulasi baju hitam / tatami gelap
            canvas[:] = [random.randint(20, 60), random.randint(20, 60), random.randint(20, 60)]
        elif bg_type == 'texture_wood':
            # Simulasi lantai kayu
            canvas[:] = [random.randint(30, 70), random.randint(60, 110), random.randint(100, 160)]
        else:
            c1 = np.random.randint(40, 220, 3)
            c2 = np.random.randint(40, 220, 3)
            X, Y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
            for c in range(3):
                canvas[:, :, c] = (c1[c] * (1 - Y) + c2[c] * Y).astype(np.uint8)

        labels = []
        # 25% Sampel Negatif (Tanpa Watermark sama sekali agar AI paham bahwa kulit/baju bersih BUKAN watermark!)
        is_negative_sample = (idx % 4 == 0)

        if not is_negative_sample:
            num_wm = random.choice([1, 2])
            for _ in range(num_wm):
                w_type, t_img = random.choice(watermark_templates)
                th, tw = t_img.shape[:2]

                target_w = random.randint(120, 260)
                target_h = int(th * (target_w / float(tw)))
                scaled = cv2.resize(t_img, (target_w, target_h))

                # Rotasi acak (-40 sampai +40 derajat)
                ang = random.uniform(-40.0, 40.0)
                M = cv2.getRotationMatrix2D((target_w / 2.0, target_h / 2.0), ang, 1.0)
                cos_a = np.abs(M[0, 0])
                sin_a = np.abs(M[0, 1])
                nw = int((target_h * sin_a) + (target_w * cos_a))
                nh = int((target_h * cos_a) + (target_w * sin_a))
                M[0, 2] += (nw / 2.0) - (target_w / 2.0)
                M[1, 2] += (nh / 2.0) - (target_h / 2.0)
                st_rot = cv2.warpAffine(scaled, M, (nw, nh))

                if nw >= W or nh >= H:
                    continue

                x = random.randint(5, W - nw - 5)
                y = random.randint(5, H - nh - 5)

                alpha = random.uniform(0.18, 0.70)
                roi = canvas[y:y+nh, x:x+nw].astype(float)
                mask_3c = np.repeat((st_rot > 60)[:, :, np.newaxis], 3, axis=2)
                canvas[y:y+nh, x:x+nw] = np.where(mask_3c, np.clip(roi * (1 - alpha) + 255 * alpha, 0, 255).astype(np.uint8), roi.astype(np.uint8))

                cx = (x + nw / 2.0) / float(W)
                cy = (y + nh / 2.0) / float(H)
                bw = nw / float(W)
                bh = nh / float(H)
                labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Simpan file gambar dan label YOLO
        img_fn = f"synth_{idx:04d}.jpg"
        cv2.imwrite(os.path.join(img_out, img_fn), canvas)
        with open(os.path.join(lbl_out, f"synth_{idx:04d}.txt"), "w") as f_lbl:
            f_lbl.write("\\n".join(labels) + "\\n")

    # 3. Buat File Konfigurasi YOLO dataset.yaml
    yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val
names:
  0: watermark
"""
    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f_y:
        f_y.write(yaml_content)

    # 4. Training YOLOv11 di GPU T4 (~1.5 Menit) - Transfer Learning dari Model Bawaan
    print("\\n[*] Memulai Transfer Learning YOLOv11 di GPU NVIDIA T4 (~1.5 menit)...")
    from ultralytics import YOLO
    base_weights = "yolo11n.pt"
    model = YOLO(base_weights)
    model.train(
        data=yaml_path,
        epochs=12,
        imgsz=640,
        batch=16,
        device=0,
        verbose=False,
        save=True
    )

    # 5. Simpan Permanen ke Google Drive & Lokal
    trained_weights = "runs/detect/train/weights/best.pt"
    if os.path.exists(trained_weights):
        shutil.copy(trained_weights, LOCAL_MODEL)
        shutil.copy(LOCAL_MODEL, DRIVE_MODEL)
        print("\\n" + "=" * 65)
        print("🎉 TRAINING SUKSES! MODEL TELAH TERSIMPAN PERMANEN!")
        print(f"💾 Lokasi Google Drive: {DRIVE_MODEL}")
        print("⚡ Mulai besok dan seterusnya, Anda TIDAK PERLU training ulang lagi!")
        print("=" * 65 + "\\n")
        try:
            from google.colab import files
            files.download(LOCAL_MODEL)
        except Exception:
            pass
'''

notebook_content = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# 🚀 Dewatermark Pro — Google Colab GPU Backend Server (v2.2 Precision AI Edition)\n',
                '\n',
                'Akselerasi GPU Cloud untuk **SimpleLaMa Neural Inpainting** dan Pembersihan Watermark Presisi Tinggi.\n',
                '\n',
                '### ⚡ UPDATE TERBARU (Landscape & Portrait Precision, Sakerly & Unified Batch):\n',
                '- **Multi-Aspect Support (Landscape & Portrait)**: Deteksi stempel RFans kini 100% otomatis beradaptasi pada orientasi foto (Landscape, Portrait, maupun resolusi web) dengan multi-scale matching dinamis berbasis sisi pendek & panjang. Menghapus bersih stempel bulat di foto landscape tanpa terlewat!\n',
                '- **Batch Download & Unified ZIP**: Seluruh hasil proses batch otomatis disimpan di folder `/content/cleaned_photos/` di Google Colab dan disatukan ke dalam **1 file ZIP tunggal** (`/content/cleaned_batch_latest.zip` & `/content/{batch_id}.zip`). Tidak ada lagi file ZIP yang terpisah-pisah!\n',
                '- **Anti-Gagal Download**: Web Studio kini dilengkapi tombol unduh per-foto (`⬇️ Unduh`) langsung dari memori browser, serta tombol `Download Semua (ZIP)` menggunakan client-side JSZip instan dan server fallback.\n',
                '- **Stempel Bulat RFans**: 100% menggunakan master template bersih `wm_bulat_besar.png` (stroke-level 1-2px, anti-blobs, tekstur karpet & sepatu tetap 100% utuh tanpa kabut buram).\n',
                '- **Sakerly Signature**: Full-image pyramid search untuk tanda tangan coral-red `t.me/Sakerly` di mana saja (kacamata hitam, rambut, baju, sudut) dalam <20ms.\n',
                '- **Anti-Solid Box**: Poligon balok solid OCR telah dihapus total. Area stempel bulat dilindungi dari poligon OCR.\n',
                '- **Zero Training Required**: Template matematis langsung aktif otomatis tanpa perlu training lagi.\n',
                '\n',
                '### 🚀 Cara Menjalankan (Hanya 2 Langkah Cepat!):\n',
                '1. **Langkah 1**: Klik menu **Runtime** ➔ **Change runtime type** ➔ Pilih **T4 GPU** ➔ **Save**.\n',
                '2. **Langkah 2**: Jalankan **CELL 1 (Setup Modul)** (~15 detik).\n',
                '3. **Langkah 3**: Jalankan **CELL 2 (Start GPU Server)**.\n',
                '4. Copy link `https://...trycloudflare.com` yang muncul di bagian bawah, lalu tempel ke Web Studio lokal Anda di menu **⚡ GPU Acceleration**!\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# CELL 1: Siapkan Modul LaMa Inpainting GPU (~15 detik)\n',
                '!pip install --no-deps simple-lama-inpainting ultralytics --quiet\n',
                '!pip install flask-cors easyocr --quiet\n',
                '\n',
                'import os, sys\n',
                'print("✅ Modul LaMa GPU Acceleration berhasil terpasang di Colab!")\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in server_code.splitlines()]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '---\n',
                '### 📥 [FITUR BARU] Cek Folder Hasil Foto & Unduh ZIP Langsung dari Colab (Opsional)\n',
                'Semua foto yang telah dibersihkan oleh server otomatis disimpan di disk Colab `/content/cleaned_photos/` dan disatukan ke dalam satu file ZIP `/content/cleaned_batch_latest.zip`.\n',
                'Anda dapat menjalankan cell di bawah ini kapan saja untuk melihat file atau mengunduh langsung dari Colab:\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# CELL 3: Cek Folder Foto Bersih & Unduh ZIP Tunggal Langsung dari Colab\n',
                'import os\n',
                'from google.colab import files\n',
                '\n',
                'print("=" * 65)\n',
                'print("📁 FOLDER PENYIMPANAN COLAB: /content/cleaned_photos")\n',
                'print("=" * 65)\n',
                'if os.path.exists("/content/cleaned_photos"):\n',
                '    items = os.listdir("/content/cleaned_photos")\n',
                '    print(f"Total file tersimpan: {len(items)} file")\n',
                '    !ls -lh /content/cleaned_photos\n',
                'else:\n',
                '    print("Belum ada foto yang diproses. Silakan jalankan batch dari Web Studio.")\n',
                '\n',
                '# Unduh ZIP Tunggal Gabungan Hasil Batch Terbaru\n',
                'if os.path.exists("/content/cleaned_batch_latest.zip"):\n',
                '    print("\\n⬇️ Mengunduh 1 file ZIP gabungan batch terbaru ke komputer Anda...")\n',
                '    files.download("/content/cleaned_batch_latest.zip")\n',
                'else:\n',
                '    print("\\n💡 Tip: Jalankan proses Batch di Web Studio terlebih dahulu untuk men-generate file ZIP.")\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '---\n',
                '### 📦 [OPSIONAL] Cell Training Tambahan (Tidak Diperlukan untuk RFans / Sakerly)\n',
                'Hanya jalankan cell di bawah ini jika suatu hari Anda ingin melatih model AI baru untuk watermark asing yang belum ada di daftar template.\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [line + '\n' for line in trainer_code.splitlines()]
        }
    ],
    'metadata': {
        'accelerator': 'GPU',
        'colab': {
            'provenance': []
        },
        'kernelspec': {
            'display_name': 'Python 3',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 0
}

with open('Dewatermark_GPU_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)

print('Dewatermark_GPU_Colab.ipynb successfully built with Auto-Trainer & Google Drive persistence!')
