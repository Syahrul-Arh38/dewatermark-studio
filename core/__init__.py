"""
Core Watermark Engine Package
"""
from .detector import WatermarkDetector, enhance_for_ocr, detect_stamps_and_geometry
from .inpainter import InpaintingEngine
from .pipeline import WatermarkPipeline
from .utils import bgr_to_base64, base64_to_bgr, base64_to_mask, save_image_lossless

__all__ = [
    'WatermarkDetector',
    'InpaintingEngine',
    'WatermarkPipeline',
    'enhance_for_ocr',
    'detect_stamps_and_geometry',
    'bgr_to_base64',
    'base64_to_bgr',
    'base64_to_mask',
    'save_image_lossless'
]
