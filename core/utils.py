"""
core/utils.py - Image I/O, base64 encoding, and sample test generator
"""

import os
import cv2
import base64
import numpy as np
from PIL import Image
from typing import Tuple


def bgr_to_base64(img_bgr: np.ndarray, format_ext: str = "png", quality: int = 95) -> str:
    """Encode OpenCV BGR image to base64 data URI."""
    if format_ext.lower() in ("jpg", "jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', img_bgr, encode_params)
        mime = "image/jpeg"
    else:
        success, buffer = cv2.imencode('.png', img_bgr)
        mime = "image/png"
        
    if not success:
        raise ValueError("Failed to encode image to buffer")
        
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:{mime};base64,{b64_str}"


def base64_to_bgr(base64_str: str) -> np.ndarray:
    """Decode base64 data URI to OpenCV BGR image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image from base64 string")
    return img_bgr


def base64_to_mask(base64_str: str, target_shape: Tuple[int, int]) -> np.ndarray:
    """Decode base64 data URI to single-channel binary mask matching target shape."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        return np.zeros(target_shape, dtype=np.uint8)
        
    # If 4 channels (RGBA from canvas), extract alpha channel or threshold
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        mask = mask[:, :, 3]
    elif len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask


def save_image_lossless(img_bgr: np.ndarray, file_path: str):
    """Save image with 100% quality and 0 subsampling for JPEG, or lossless PNG."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(file_path, img_bgr, [
            int(cv2.IMWRITE_JPEG_QUALITY), 100,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
        ])
    elif ext == '.webp':
        cv2.imwrite(file_path, img_bgr, [int(cv2.IMWRITE_WEBP_QUALITY), 100])
    else:
        cv2.imwrite(file_path, img_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])


def create_sample_watermark_image(output_path: str) -> str:
    """
    Generate an aesthetic photographic test image containing realistic watermarks:
    1. A circular Telegram channel stamp in the corner (like @rfans / channel).
    2. A diagonal semi-transparent watermark across the center.
    3. Bottom text copyright tag.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create an appealing scenic background gradient with textured shapes
    w, h = 1000, 750
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Vibrant sunset gradient
    for y in range(h):
        r = int(240 - (y / h) * 160)
        g = int(120 - (y / h) * 60)
        b = int(70 + (y / h) * 110)
        img[y, :] = (b, g, r)
        
    # Add scenic hills / landscape
    cv2.circle(img, (200, 900), 500, (40, 30, 25), -1)
    cv2.circle(img, (800, 850), 450, (60, 45, 35), -1)
    cv2.circle(img, (500, 920), 400, (80, 60, 45), -1)
    
    # Sun
    cv2.circle(img, (500, 320), 85, (120, 230, 255), -1)
    
    # Water reflection
    cv2.rectangle(img, (0, 520), (w, h), (110, 60, 40), -1)
    for y in range(520, h, 6):
        cv2.line(img, (0, y), (w, y), (140, 80, 50), 1)

    # Add realistic watermarks onto copy
    watermarked = img.copy()
    overlay = watermarked.copy()
    
    # 1. Circular Telegram/RFans style stamp (bottom-right)
    stamp_cx, stamp_cy, stamp_r = 860, 620, 65
    cv2.circle(overlay, (stamp_cx, stamp_cy), stamp_r, (255, 255, 255), 3)
    cv2.circle(overlay, (stamp_cx, stamp_cy), stamp_r - 8, (255, 255, 255), 1)
    # Stamp text
    cv2.putText(overlay, "RFANS BOT", (stamp_cx - 50, stamp_cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, "CHANNEL VIP", (stamp_cx - 52, stamp_cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    # Little paper plane icon representation
    plane_pts = np.array([[stamp_cx, stamp_cy - 30], [stamp_cx - 15, stamp_cy - 45], [stamp_cx + 15, stamp_cy - 40]], dtype=np.int32)
    cv2.polylines(overlay, [plane_pts], True, (255, 255, 255), 1)
    
    # 2. Semi-transparent diagonal watermark across the middle
    diag_text = "PREVIEW COPY - DO NOT DISTRIBUTE"
    cv2.putText(overlay, diag_text, (180, 380), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 3. Top left channel tag
    cv2.putText(overlay, "t.me/album_exclusive_cos", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Alpha blend the overlay at 60% opacity
    cv2.addWeighted(overlay, 0.65, watermarked, 0.35, 0, watermarked)
    
    save_image_lossless(watermarked, output_path)
    return output_path
