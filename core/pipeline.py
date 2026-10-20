"""
core/pipeline.py - Unified Pipeline connecting Detector and Inpainter
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List
from .detector import WatermarkDetector
from .inpainter import InpaintingEngine
from .utils import bgr_to_base64


class WatermarkPipeline:
    """End-to-end processing pipeline for watermark detection and removal."""

    def __init__(self, use_gpu: bool = False):
        self.detector = WatermarkDetector(use_gpu=use_gpu)
        self.inpainter = InpaintingEngine(use_gpu=use_gpu)

    def process(
        self,
        img_bgr: np.ndarray,
        custom_mask: Optional[np.ndarray] = None,
        use_ocr: bool = True,
        use_heuristics: bool = True,
        sensitivity: float = 0.5,
        dilation_px: int = 2,
        wm_mode: str = "auto",
        inpaint_method: str = "auto",
        feather_edges: bool = True
    ) -> Dict[str, Any]:
        """
        Execute detection + inpainting.
        If custom_mask is provided (e.g. from user manual brush), it is merged with or overrides detection.
        """
        start_time = time.time()
        h, w = img_bgr.shape[:2]

        detection_info = {}
        if custom_mask is not None:
            mask = custom_mask.copy()
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            detection_info = {'source': 'manual_brush'}
        else:
            det_res = self.detector.detect(
                img_bgr=img_bgr,
                wm_mode=wm_mode,
                use_ocr=use_ocr,
                use_heuristics=use_heuristics,
                sensitivity=sensitivity,
                dilation_px=dilation_px
            )
            mask = det_res['mask']
            detection_info = det_res

        # Execute inpainting
        cleaned_bgr = self.inpainter.inpaint(
            img_bgr=img_bgr,
            mask=mask,
            method=inpaint_method,
            feather_edges=feather_edges
        )

        elapsed = round(time.time() - start_time, 3)

        return {
            'cleaned_bgr': cleaned_bgr,
            'mask': mask,
            'elapsed_sec': elapsed,
            'detection_info': detection_info,
            'dimensions': {'width': w, 'height': h}
        }
