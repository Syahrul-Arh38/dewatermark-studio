import cv2
import numpy as np

def extract_precise_strokes(img_bgr, pts_list=[], stamp_circles=[], sensitivity=0.5):
    """
    Ekstraksi guratan watermark tingkat piksel (stroke-level).
    TIDAK PERNAH memblok penuh (solid block), melainkan hanya mengambil tinta tulisan (2-3px)
    dan garis lingkaran cincin stempel, sehingga tekstur tembok dan karpet 100% terjaga.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Stroke extraction for bounding boxes (misal teks pada baju/langit)
    for pts in pts_list:
        pts = np.array(pts, dtype=np.int32)
        bx, by, bw, bh = cv2.boundingRect(pts)
        pad = 4
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
        
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            continue
            
        poly_mask = np.zeros_like(roi_gray)
        shifted_pts = pts - np.array([x1, y1])
        cv2.fillPoly(poly_mask, [shifted_pts], 255)
        poly_mask = cv2.dilate(poly_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        
        # Ekstraksi guratan terang (Top-Hat + Luminance)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, k)
        th_val = int(25 * (1.2 - sensitivity * 0.4))
        _, strk = cv2.threshold(tophat, th_val, 255, cv2.THRESH_BINARY)
        _, brt = cv2.threshold(roi_gray, 190, 255, cv2.THRESH_BINARY)
        combined_roi = cv2.bitwise_or(strk, brt)
        combined_roi = cv2.bitwise_and(combined_roi, poly_mask)
        
        # Dilasi mikro 1-2px saja agar tidak ngeblok
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_roi = cv2.dilate(combined_roi, k_dil, iterations=1)
        
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], combined_roi)
        
    # 2. Stroke extraction for circular stamps (Stempel Telegram / RFans)
    for (cx, cy, r) in stamp_circles:
        cx, cy, r = int(cx), int(cy), int(r)
        y1, y2 = max(0, cy - r - 6), min(h, cy + r + 6)
        x1, x2 = max(0, cx - r - 6), min(w, cx + r + 6)
        
        roi_gray = gray[y1:y2, x1:x2]
        if roi_gray.size == 0:
            continue
            
        circle_roi = np.zeros_like(roi_gray)
        cv2.circle(circle_roi, (cx - x1, cy - y1), r + 2, 255, -1)
        
        # Garis cincin lingkaran luar & dalam (HANYA setebal 2-3px, BUKAN SOLID -1!)
        ring_mask = np.zeros_like(roi_gray)
        cv2.circle(ring_mask, (cx - x1, cy - y1), r, 255, 3)
        if r > 20:
            cv2.circle(ring_mask, (cx - x1, cy - y1), max(5, r - 8), 255, 2)
            
        # Guratan huruf di dalam stempel
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, k)
        th_val = int(14 * (1.1 - sensitivity * 0.3))
        _, strk = cv2.threshold(tophat, th_val, 255, cv2.THRESH_BINARY)
        _, brt = cv2.threshold(roi_gray, 205, 255, cv2.THRESH_BINARY)
        
        stamp_stroke = cv2.bitwise_or(strk, brt)
        stamp_stroke = cv2.bitwise_or(stamp_stroke, ring_mask)
        stamp_stroke = cv2.bitwise_and(stamp_stroke, circle_roi)
        
        # Filter bintik noise sangat kecil
        num, labels, stats, _ = cv2.connectedComponentsWithStats(stamp_stroke)
        cleaned_stamp = np.zeros_like(stamp_stroke)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                cleaned_stamp[labels == i] = 255
                
        # Dilasi halus 1-2px
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_stamp = cv2.dilate(cleaned_stamp, k_dil, iterations=1)
        
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], cleaned_stamp)
        
    return mask

if __name__ == '__main__':
    img_path = r"C:\Users\Administrator\.gemini\antigravity-ide\brain\f8c36c6f-14f0-4f58-b3f7-b2ddc5a07fe4\.user_uploaded\media_1792302928507.jpg"
    img = cv2.imread(img_path)
    
    # Stamp bottom right: cx=585, cy=935, r=85
    # Chinese watermark on dress: pts around x: 385..440, y: 540..580
    # RFANS watermark on dress: pts around x: 445..500, y: 530..565
    pts1 = [[385, 540], [440, 540], [440, 580], [385, 580]]
    pts2 = [[445, 530], [500, 530], [500, 565], [445, 565]]
    pts3 = [[440, 115], [500, 115], [500, 155], [440, 155]] # transparent sky
    
    mask = extract_precise_strokes(img, pts_list=[pts1, pts2, pts3], stamp_circles=[(585, 935, 85)])
    cv2.imwrite("samples/canggih_full_mask.png", mask)
    
    # Inpainting
    cleaned = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    cv2.imwrite("samples/canggih_full_result.jpg", cleaned)
    
    # Wall crop verification
    wall_res = cleaned[840:950, 480:680]
    cv2.imwrite("samples/canggih_wall_verified.jpg", wall_res)
    print("Done! Total stroke pixels:", np.count_nonzero(mask))
