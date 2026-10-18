"""
core/detector.py - Watermark Detection & Segmentation Engine
Addresses flaws in naive OCR approaches:
1. Multi-cluster stamp grouping (prevents multi-stamp centroid averaging bugs).
2. Dynamic radius calculation based on enclosing geometry.
3. Dual-pass CLAHE contrast enhancement for low-contrast/semi-transparent text.
4. Morphological stroke & gradient frequency segmentation for arbitrary watermarks.
5. EasyOCR integration with graceful fallback.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False


def enhance_for_ocr(img_bgr: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """Enhance low-contrast and semi-transparent watermarks using CLAHE in LAB space."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def cluster_boxes(boxes: List[Tuple[int, int, int, int]], max_distance: float) -> List[List[Tuple[int, int, int, int]]]:
    """
    Cluster spatial bounding boxes (x, y, w, h) into separate groups.
    Solves the bug where multiple distant watermarks/stamps are averaged together.
    """
    if not boxes:
        return []

    clusters: List[List[Tuple[int, int, int, int]]] = []
    
    for box in boxes:
        bx, by, bw, bh = box
        bcx, bcy = bx + bw / 2, by + bh / 2
        
        placed = False
        for cluster in clusters:
            # Check distance to any box in this cluster
            for c_box in cluster:
                cx, cy, cw, ch = c_box
                cc_x, cc_y = cx + cw / 2, cy + ch / 2
                dist = np.hypot(bcx - cc_x, bcy - cc_y)
                if dist <= max_distance:
                    cluster.append(box)
                    placed = True
                    break
            if placed:
                break
                
        if not placed:
            clusters.append([box])
            
    return clusters


def detect_stamps_and_geometry(
    img_bgr: np.ndarray,
    candidate_boxes: List[Tuple[int, int, int, int]],
    pad_ratio: float = 0.15
) -> np.ndarray:
    """
    Given candidate stamp fragments (e.g. text from 'channel', 'bot', 'album'),
    cluster them spatially and create precise enclosing circles/hulls per individual stamp.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not candidate_boxes:
        return mask

    # Cluster distance threshold: 18% of minimum dimension
    max_dist = max(30, int(min(h, w) * 0.18))
    clusters = cluster_boxes(candidate_boxes, max_distance=max_dist)
    
    for cluster in clusters:
        # Collect all corner points for this specific cluster
        points = []
        for (bx, by, bw, bh) in cluster:
            points.append([bx, by])
            points.append([bx + bw, by])
            points.append([bx + bw, by + bh])
            points.append([bx, by + bh])
            
        pts = np.array(points, dtype=np.float32)
        (center_x, center_y), radius = cv2.minEnclosingCircle(pts)
        
        # Apply proportional padding to cover outer ring and icon badges
        radius = int(radius * (1.0 + pad_ratio))
        # Ensure radius is within a realistic stamp bound
        max_allowed_radius = int(min(h, w) * 0.35)
        min_allowed_radius = int(min(h, w) * 0.04)
        radius = np.clip(radius, min_allowed_radius, max_allowed_radius)
        
        cx, cy = int(center_x), int(center_y)
        cv2.circle(mask, (cx, cy), int(radius), 255, -1)
        
    return mask


def detect_watermark_strokes(
    img_bgr: np.ndarray,
    sensitivity: float = 0.5,
    detect_light: bool = True,
    detect_dark: bool = True
) -> np.ndarray:
    """
    Multi-scale morphological & frequency-based stroke detector.
    Detects semi-transparent watermarks, translucent logos, and overlaid text
    independent of OCR or language.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Scale kernel size based on resolution
    base_k = max(7, int(min(h, w) * 0.015))
    if base_k % 2 == 0:
        base_k += 1
    
    stroke_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Morphological Top-Hat (bright strokes on darker background)
    # Morphological Black-Hat (dark strokes on lighter background)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (base_k, base_k))
    
    if detect_light:
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_rect)
        thresh_val = int(255 * (1.0 - sensitivity * 0.6))
        _, th_light = cv2.threshold(top_hat, 25, 255, cv2.THRESH_BINARY)
        stroke_mask = cv2.bitwise_or(stroke_mask, th_light)
        
    if detect_dark:
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_rect)
        _, th_dark = cv2.threshold(black_hat, 25, 255, cv2.THRESH_BINARY)
        stroke_mask = cv2.bitwise_or(stroke_mask, th_dark)
        
    # High-pass edge gradient to preserve stroke boundaries
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0))
    
    _, edge_mask = cv2.threshold(abs_grad, int(40 * (1.1 - sensitivity)), 255, cv2.THRESH_BINARY)
    
    # Intersect strokes with edges to prevent masking smooth background gradients
    combined = cv2.bitwise_and(stroke_mask, edge_mask)
    
    # Remove small isolated speckle noise (connected components filtering)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    min_area = max(10, int((h * w) * 0.00003))
    max_area = int((h * w) * 0.25)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            clean_mask[labels == i] = 255
            
    return clean_mask


class WatermarkDetector:
    """Unified detector coordinating OCR, Stamp clustering, and Stroke segmentation."""
    
    def __init__(self, languages: Optional[List[str]] = None, use_gpu: bool = False):
        self.languages = languages or ['ch_sim', 'en']
        self.use_gpu = use_gpu
        self._reader = None
        self.default_wm_keywords = ['rfans', 'channel', 'bot', 'tg', 'album', '电', '报', '群', '搜', '索', '@', 't.me']
        self.stamp_keywords = ['channel', 'album', 'bot', 'rfans', 'group']

    @property
    def reader(self):
        if self._reader is None and EASYOCR_AVAILABLE:
            try:
                self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            except Exception as e:
                print(f"[WatermarkDetector] EasyOCR init notice: {e}")
                self._reader = None
        return self._reader

    def detect(
        self,
        img_bgr: np.ndarray,
        use_ocr: bool = True,
        use_heuristics: bool = True,
        sensitivity: float = 0.5,
        dilation_px: int = 9,
        custom_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Main detection entrypoint.
        Returns:
            {
                'mask': np.ndarray (binary mask 0 or 255),
                'stamp_count': int,
                'detected_regions': list of bbox,
                'method': str
            }
        """
        h, w = img_bgr.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        candidate_stamp_boxes: List[Tuple[int, int, int, int]] = []
        detected_texts: List[str] = []
        
        keywords = custom_keywords or self.default_wm_keywords
        
        # 1. OCR Detection (if available & enabled)
        if use_ocr and self.reader is not None:
            img_enhanced = enhance_for_ocr(img_bgr)
            try:
                res_normal = self.reader.readtext(img_bgr, contrast_ths=0.05, adjust_contrast=0.7)
                res_enhanced = self.reader.readtext(img_enhanced, contrast_ths=0.03, text_threshold=0.4, low_text=0.2)
                all_results = res_normal + res_enhanced
                
                for (bbox, text, prob) in all_results:
                    t_low = text.lower().replace(" ", "")
                    if any(k in t_low for k in keywords):
                        pts = np.array(bbox, dtype=np.int32)
                        cv2.fillPoly(final_mask, [pts], 255)
                        
                        bx = int(np.min(pts[:, 0]))
                        by = int(np.min(pts[:, 1]))
                        bw = int(np.max(pts[:, 0]) - bx)
                        bh = int(np.max(pts[:, 1]) - by)
                        
                        detected_texts.append(text)
                        
                        if any(sk in t_low for sk in self.stamp_keywords) or bw > w * 0.07:
                            candidate_stamp_boxes.append((bx, by, bw, bh))
            except Exception as e:
                print(f"[WatermarkDetector] OCR scan warning: {e}")

        # 2. Structural & Morphological Stroke Detection (General / Non-OCR)
        if use_heuristics:
            stroke_mask = detect_watermark_strokes(img_bgr, sensitivity=sensitivity)
            final_mask = cv2.bitwise_or(final_mask, stroke_mask)
            
        # 3. Stamp Geometric Enclosure (fixes center averaging bug)
        stamp_mask = detect_stamps_and_geometry(img_bgr, candidate_stamp_boxes)
        final_mask = cv2.bitwise_or(final_mask, stamp_mask)
        
        # 4. Adaptive Dilation (Smooth edges and wrap anti-aliasing)
        if dilation_px > 0:
            k_size = max(3, dilation_px)
            if k_size % 2 == 0:
                k_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            final_mask = cv2.dilate(final_mask, kernel, iterations=1)
            
        return {
            'mask': final_mask,
            'stamp_count': len(candidate_stamp_boxes),
            'detected_texts': detected_texts,
            'easyocr_available': EASYOCR_AVAILABLE and (self._reader is not None)
        }
