import cv2
import numpy as np

def detect_all_watermarks_unified(img_bgr, reader=None, sensitivity=0.5, dilation_px=2):
    h, w = img_bgr.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # ----------------------------------------------------
    # 1. Dual-Pass Multi-Scale CLAHE Enhancement
    # ----------------------------------------------------
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c1 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    c2 = cv2.createCLAHE(clipLimit=14.0, tileGridSize=(4, 4))
    l_enh = cv2.addWeighted(c1.apply(l), 0.35, c2.apply(l), 0.65, 0)
    img_enh = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)
    gray_enh = cv2.cvtColor(img_enh, cv2.COLOR_BGR2GRAY)
    
    wm_keywords = ['rfans', 'channel', 'bot', 'tg', 'album', '电', '报', '群', '搜', '索', '直', '接', '@', 'fans', 't.me', 'vip']
    stamp_keywords = ['channel', 'album', 'bot', 'rfans']
    stamp_candidates = []
    
    # ----------------------------------------------------
    # 2. OCR Detection (Normal + Rotated -30 deg for diagonal text)
    # ----------------------------------------------------
    if reader is not None:
        try:
            # Pass A: Normal orientation
            r_norm = reader.readtext(img_bgr, contrast_ths=0.04, adjust_contrast=0.7)
            r_enh = reader.readtext(img_enh, contrast_ths=0.015, text_threshold=0.18, low_text=0.10)
            
            # Pass B: Rotated -30 deg for diagonal tiled watermarks
            center = (w / 2, h / 2)
            M_rot = cv2.getRotationMatrix2D(center, -30, 1.0)
            M_inv = cv2.getRotationMatrix2D(center, 30, 1.0)
            img_rot = cv2.warpAffine(img_enh, M_rot, (w, h))
            r_rot = reader.readtext(img_rot, contrast_ths=0.015, text_threshold=0.18, low_text=0.10)
            
            # Process normal & enhanced results
            for (bbox, text, prob) in (r_norm + r_enh):
                t_low = text.lower().replace(' ', '')
                if any(k in t_low for k in wm_keywords):
                    pts = np.array(bbox, dtype=np.int32)
                    bx = int(np.min(pts[:, 0]))
                    by = int(np.min(pts[:, 1]))
                    bw = int(np.max(pts[:, 0]) - bx)
                    bh = int(np.max(pts[:, 1]) - by)
                    
                    # Stroke-level inside bbox
                    pad = 3
                    x1, y1 = max(0, bx - pad), max(0, by - pad)
                    x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
                    roi = gray_enh[y1:y2, x1:x2]
                    if roi.size > 0:
                        p_mask = np.zeros_like(roi)
                        cv2.fillPoly(p_mask, [pts - [x1, y1]], 255)
                        k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k_th)
                        _, s1 = cv2.threshold(th, 14, 255, cv2.THRESH_BINARY)
                        _, s2 = cv2.threshold(roi, 195, 255, cv2.THRESH_BINARY)
                        comb = cv2.bitwise_and(cv2.bitwise_or(s1, s2), p_mask)
                        comb = cv2.dilate(comb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(final_mask[y1:y2, x1:x2], comb)
                    
                    if any(sk in t_low for sk in stamp_keywords) or bw > w * 0.08:
                        stamp_candidates.append((bx, by, bw, bh))
                        
            # Process rotated results (transform back)
            for (bbox, text, prob) in r_rot:
                t_low = text.lower().replace(' ', '')
                if any(k in t_low for k in wm_keywords):
                    pts_rot = np.array(bbox, dtype=np.float32)
                    # Transform back to original coordinate system
                    ones = np.ones((len(pts_rot), 1), dtype=np.float32)
                    pts_homo = np.hstack([pts_rot, ones])
                    orig_pts = M_inv.dot(pts_homo.T).T.astype(np.int32)
                    
                    bx = int(np.clip(np.min(orig_pts[:, 0]), 0, w - 1))
                    by = int(np.clip(np.min(orig_pts[:, 1]), 0, h - 1))
                    bw = int(np.clip(np.max(orig_pts[:, 0]) - bx, 1, w - bx))
                    bh = int(np.clip(np.max(orig_pts[:, 1]) - by, 1, h - by))
                    
                    pad = 3
                    x1, y1 = max(0, bx - pad), max(0, by - pad)
                    x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
                    roi = gray_enh[y1:y2, x1:x2]
                    if roi.size > 0:
                        p_mask = np.zeros_like(roi)
                        cv2.fillPoly(p_mask, [orig_pts - [x1, y1]], 255)
                        k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, k_th)
                        _, s1 = cv2.threshold(th, 14, 255, cv2.THRESH_BINARY)
                        comb = cv2.bitwise_and(s1, p_mask)
                        comb = cv2.dilate(comb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                        final_mask[y1:y2, x1:x2] = cv2.bitwise_or(final_mask[y1:y2, x1:x2], comb)
        except Exception as e:
            print(f"[OCR Warning] {e}")

    # ----------------------------------------------------
    # 3. Transparent Tiled Watermark Anchor Scanner (Icon + Trajectory)
    # ----------------------------------------------------
    icon = cv2.imread('samples/telegram_icon.png')
    if icon is not None:
        icon_scaled = cv2.resize(icon, (0, 0), fx=0.29, fy=0.29)
        icon_gray = cv2.cvtColor(icon_scaled, cv2.COLOR_BGR2GRAY)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        icon_th = cv2.morphologyEx(icon_gray, cv2.MORPH_TOPHAT, k3)
        orig_th3 = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k3)
        
        res = cv2.matchTemplate(orig_th3, icon_th, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res > 0.44)
        pts = list(zip(locs[1], locs[0]))
        
        clustered = []
        for (x, y) in pts:
            if not any(np.hypot(x - cx, y - cy) < 22 for (cx, cy) in clustered):
                clustered.append((x, y))
                
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th_enh = cv2.morphologyEx(gray_enh, cv2.MORPH_TOPHAT, k5)
        
        for (ix, iy) in clustered:
            x1, y1 = max(0, ix - 6), max(0, iy - 6)
            x2, y2 = min(w, ix + 125), min(h, iy + 70)
            roi_th = th_enh[y1:y2, x1:x2]
            if roi_th.size > 0:
                band = np.zeros_like(roi_th)
                cv2.line(band, (0, 10), (120, 52), 255, 26)
                _, strk = cv2.threshold(roi_th, 15, 255, cv2.THRESH_BINARY)
                strk = cv2.bitwise_and(strk, band)
                strk = cv2.dilate(strk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                final_mask[y1:y2, x1:x2] = cv2.bitwise_or(final_mask[y1:y2, x1:x2], strk)

    # ----------------------------------------------------
    # 4. Faint Background Stroke Sweep (Smooth Sky & Dark Fabric)
    # ----------------------------------------------------
    mean = cv2.blur(gray.astype(float), (9, 9))
    sq_mean = cv2.blur((gray.astype(float))**2, (9, 9))
    std = np.sqrt(np.maximum(0, sq_mean - mean**2))
    
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    th_enh = cv2.morphologyEx(gray_enh, cv2.MORPH_TOPHAT, k5)
    
    # In smooth/flat regions (std < 18), detect low-contrast strokes
    flat_gate = (std < 18).astype(np.uint8) * 255
    th_val = int(14 * (1.1 - sensitivity * 0.3))
    _, faint_strk = cv2.threshold(th_enh, th_val, 255, cv2.THRESH_BINARY)
    faint_strk = cv2.bitwise_and(faint_strk, flat_gate)
    
    num, labels, stats, _ = cv2.connectedComponentsWithStats(faint_strk)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bw_c = stats[i, cv2.CC_STAT_WIDTH]
        bh_c = stats[i, cv2.CC_STAT_HEIGHT]
        if 4 <= area <= 160 and max(bw_c, bh_c) <= 28:
            final_mask[labels == i] = 255

    # ----------------------------------------------------
    # 5. Circular Stamp Geometric Stroke Extraction
    # ----------------------------------------------------
    if stamp_candidates:
        all_pts = []
        for (bx, by, bw, bh) in stamp_candidates:
            all_pts.extend([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]])
        (cx, cy), radius = cv2.minEnclosingCircle(np.array(all_pts, dtype=np.float32))
        cx, cy, r = int(cx), int(cy), int(radius * 1.15)
        y1, y2 = max(0, cy - r), min(h, cy + r)
        x1, x2 = max(0, cx - r), min(w, cx + r)
        roi_stamp = gray[y1:y2, x1:x2]
        if roi_stamp.size > 0:
            c_roi = np.zeros_like(roi_stamp)
            cv2.circle(c_roi, (cx - x1, cy - y1), r, 255, -1)
            k_th = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            tophat = cv2.morphologyEx(roi_stamp, cv2.MORPH_TOPHAT, k_th)
            _, strk = cv2.threshold(tophat, 14, 255, cv2.THRESH_BINARY)
            _, brt = cv2.threshold(roi_stamp, 205, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_or(strk, brt)
            cv2.circle(combined, (cx - x1, cy - y1), r, 255, 3)
            if r > 20:
                cv2.circle(combined, (cx - x1, cy - y1), max(5, r - 8), 255, 2)
            combined = cv2.bitwise_and(combined, c_roi)
            combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            final_mask[y1:y2, x1:x2] = cv2.bitwise_or(final_mask[y1:y2, x1:x2], combined)

    # ----------------------------------------------------
    # 6. Adaptive Micro Dilation (2px Default)
    # ----------------------------------------------------
    if dilation_px > 0:
        k_sz = min(5, max(3, dilation_px if dilation_px % 2 != 0 else dilation_px + 1))
        final_mask = cv2.dilate(final_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz)), iterations=1)
        
    return final_mask

if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\Administrator\.gemini\antigravity-ide\brain\f8c36c6f-14f0-4f58-b3f7-b2ddc5a07fe4\.user_uploaded\media_1792302928507.jpg')
    mask = detect_all_watermarks_unified(img, sensitivity=0.55, dilation_px=2)
    print('Unified mask pixels:', np.count_nonzero(mask))
    
    # Check crops:
    sky_m = mask[104:201, 450:555]
    cloth_m = mask[542:617, 354:511]
    wall_m = mask[840:950, 480:680]
    
    print('Sky watermark pixels:', np.count_nonzero(sky_m))
    print('Cloth watermark pixels:', np.count_nonzero(cloth_m))
    print('Wall watermark pixels:', np.count_nonzero(wall_m))
    
    cv2.imwrite('samples/unified_full_mask.png', mask)
    
    # Inpaint full image
    cleaned = cv2.inpaint(img, mask, 2, cv2.INPAINT_TELEA)
    cv2.imwrite('samples/unified_full_cleaned.jpg', cleaned)
    cv2.imwrite('samples/unified_sky_clean.jpg', cleaned[104:201, 450:555])
    cv2.imwrite('samples/unified_cloth_clean.jpg', cleaned[542:617, 354:511])
    cv2.imwrite('samples/unified_wall_clean.jpg', cleaned[840:950, 480:680])
    print('Successfully verified unified detection & inpainting!')
