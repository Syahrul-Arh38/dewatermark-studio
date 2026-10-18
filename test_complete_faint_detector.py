import cv2
import numpy as np

def detect_transparent_and_diagonal_watermarks(img_bgr, sensitivity=0.5):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Multi-scale CLAHE for transparent text
    clahe = cv2.createCLAHE(clipLimit=14.0, tileGridSize=(6, 6))
    enh = clahe.apply(gray)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th_enh = cv2.morphologyEx(enh, cv2.MORPH_TOPHAT, k5)
    
    # 2. Telegram paper plane icon template matching (anchor for repeated tiles)
    icon = cv2.imread('samples/telegram_icon.png')
    if icon is not None:
        icon_scaled = cv2.resize(icon, (0, 0), fx=0.29, fy=0.29)
        icon_gray = cv2.cvtColor(icon_scaled, cv2.COLOR_BGR2GRAY)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        icon_th = cv2.morphologyEx(icon_gray, cv2.MORPH_TOPHAT, k3)
        orig_th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k3)
        
        res = cv2.matchTemplate(orig_th, icon_th, cv2.TM_CCOEFF_NORMED)
        # Match threshold based on sensitivity
        th_match = 0.42 - (sensitivity - 0.5) * 0.1
        locs = np.where(res > th_match)
        pts = list(zip(locs[1], locs[0]))
        
        # Cluster nearby points
        clustered_pts = []
        for (x, y) in pts:
            if not any(np.hypot(x - cx, y - cy) < 20 for (cx, cy) in clustered_pts):
                clustered_pts.append((x, y))
                
        for (ix, iy) in clustered_pts:
            # Trajectory of watermark text: diagonal ~30 deg, width 130px, height 70px
            x1, y1 = max(0, ix - 6), max(0, iy - 6)
            x2, y2 = min(w, ix + 145), min(h, iy + 85)
            
            roi_th = th_enh[y1:y2, x1:x2]
            if roi_th.size > 0:
                # Mask out diagonal bounding strip
                strip_mask = np.zeros_like(roi_th)
                # Triangle / oriented polygon covering text trajectory
                poly_pts = np.array([
                    [0, 0],
                    [130, 65],
                    [130, 95],
                    [0, 30]
                ], dtype=np.int32)
                cv2.fillPoly(strip_mask, [poly_pts], 255)
                
                _, strk = cv2.threshold(roi_th, 14, 255, cv2.THRESH_BINARY)
                strk = cv2.bitwise_and(strk, strip_mask)
                # Dilate 1px
                strk = cv2.dilate(strk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], strk)
                
    # 3. Flat background faint text sweep (for smooth sky & dark fabric where std < 20)
    mean = cv2.blur(gray.astype(float), (9, 9))
    sq_mean = cv2.blur((gray.astype(float))**2, (9, 9))
    std = np.sqrt(np.maximum(0, sq_mean - mean**2))
    
    # In flat background regions, detect high-pass faint strokes
    flat_gate = (std < 18).astype(np.uint8) * 255
    th_val = int(14 * (1.1 - sensitivity * 0.3))
    _, faint_strokes = cv2.threshold(th_enh, th_val, 255, cv2.THRESH_BINARY)
    faint_strokes = cv2.bitwise_and(faint_strokes, flat_gate)
    
    # Filter small noise vs text characters
    num, labels, stats, _ = cv2.connectedComponentsWithStats(faint_strokes)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if 4 <= area <= 180 and max(bw, bh) <= 28:
            mask[labels == i] = 255
            
    return mask

if __name__ == '__main__':
    orig = cv2.imread(r'C:\Users\Administrator\.gemini\antigravity-ide\brain\f8c36c6f-14f0-4f58-b3f7-b2ddc5a07fe4\.user_uploaded\media_1792302928507.jpg')
    dw_mask = cv2.imread('samples/dewatermark_full_binary_mask.png', 0)
    
    faint_mask = detect_transparent_and_diagonal_watermarks(orig, sensitivity=0.6)
    
    # Check coverage in Sky and Cloth
    sky_m = faint_mask[104:201, 450:555]
    cloth_m = faint_mask[542:617, 354:511]
    
    print('Sky watermark pixels detected:', np.count_nonzero(sky_m))
    print('Cloth watermark pixels detected:', np.count_nonzero(cloth_m))
    print('Total faint mask pixels:', np.count_nonzero(faint_mask))
    
    cv2.imwrite('samples/test_auto_faint_mask.png', faint_mask)
    
    # Inpaint test
    cleaned = cv2.inpaint(orig, faint_mask, 2, cv2.INPAINT_TELEA)
    cv2.imwrite('samples/test_auto_faint_cleaned_sky.jpg', cleaned[104:201, 450:555])
    cv2.imwrite('samples/test_auto_faint_cleaned_cloth.jpg', cleaned[542:617, 354:511])
    print('Inpainted crops saved successfully!')
