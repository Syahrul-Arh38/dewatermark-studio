"""
core/inpainter.py - High-Fidelity Watermark Inpainting Engine
Features:
1. High-Fidelity Edge-Guided Inpainting (Fast, zero-dependency, preserves gradients).
2. Feathered Boundary Blending (Eliminates harsh inpainting seam artifacts).
3. Pluggable SimpleLama / LaMa ONNX Neural Inpainting engine.
4. Alpha Matting Reconstruction for semi-transparent watermarks.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Union, Dict, Any

try:
    from simple_lama_inpainting import SimpleLama
    SIMPLE_LAMA_AVAILABLE = True
except ImportError:
    SimpleLama = None
    SIMPLE_LAMA_AVAILABLE = False


class InpaintingEngine:
    """Unified inpainting engine with neural and geometric modes."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._lama = None

    @property
    def lama(self):
        if self._lama is None and SIMPLE_LAMA_AVAILABLE:
            try:
                self._lama = SimpleLama()
            except Exception as e:
                print(f"[InpaintingEngine] LaMa loading note: {e}")
                self._lama = None
        return self._lama

    def inpaint(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        method: str = "auto",
        feather_edges: bool = True
    ) -> np.ndarray:
        """
        Inpaint image given a binary mask (255 = area to remove).
        Methods:
          - 'auto': tries LaMa if available, falls back gracefully to Edge-Guided
          - 'lama': forces LaMa neural inpainting
          - 'fast_telea': OpenCV Fast Marching
          - 'navier_stokes': Fluid-dynamics edge propagation
          - 'edge_guided': Multi-pass pyramid inpaint with bilateral smoothing
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return img_bgr.copy()

        # Ensure single-channel 8-bit mask
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 1. Check for Neural LaMa
        if method in ("auto", "lama") and self.lama is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_mask = Image.fromarray(mask_bin)
                res_pil = self.lama(pil_img, pil_mask)
                res_rgb = np.array(res_pil)
                return cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[InpaintingEngine] LaMa failed, falling back to Edge-Guided: {e}")

        # 2. Select geometric inpainting
        if method == "navier_stokes":
            return cv2.inpaint(img_bgr, mask_bin, inpaintRadius=5, flags=cv2.INPAINT_NS)
        elif method == "fast_telea":
            return cv2.inpaint(img_bgr, mask_bin, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        else:
            # Default: Advanced Edge-Guided Multi-Stage Inpainting
            return self._edge_guided_inpaint(img_bgr, mask_bin, feather_edges=feather_edges)

    def _edge_guided_inpaint(
        self,
        img_bgr: np.ndarray,
        mask_bin: np.ndarray,
        feather_edges: bool = True
    ) -> np.ndarray:
        """
        High-fidelity multi-stage inpainting:
        1. Base Telea pass for texture seed
        2. Navier-Stokes pass for structural continuity
        3. Bilateral filter on reconstructed zone to remove pixelation
        4. Feathered alpha blending along mask perimeter to guarantee zero seam lines
        """
        h, w = img_bgr.shape[:2]
        
        # Inpaint radius scaled to resolution
        inpaint_r = max(3, min(9, int(min(h, w) * 0.005)))
        
        # Dual pass inpainting
        inp_telea = cv2.inpaint(img_bgr, mask_bin, inpaintRadius=inpaint_r, flags=cv2.INPAINT_TELEA)
        inp_ns = cv2.inpaint(img_bgr, mask_bin, inpaintRadius=inpaint_r + 2, flags=cv2.INPAINT_NS)
        blended = cv2.addWeighted(inp_telea, 0.6, inp_ns, 0.4, 0)
        
        # Texture smoothing inside mask zone only
        smoothed = cv2.bilateralFilter(blended, d=7, sigmaColor=35, sigmaSpace=35)
        
        # Restore untouched background completely
        result = img_bgr.copy()
        result[mask_bin > 0] = smoothed[mask_bin > 0]
        
        # Feather mask edges to avoid any hard visible border
        if feather_edges:
            feather_size = max(5, int(min(h, w) * 0.006))
            if feather_size % 2 == 0:
                feather_size += 1
            alpha_mask = cv2.GaussianBlur(mask_bin.astype(np.float32) / 255.0, (feather_size, feather_size), 0)
            alpha_3ch = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
            result = (result.astype(np.float32) * alpha_3ch + img_bgr.astype(np.float32) * (1.0 - alpha_3ch)).astype(np.uint8)
            
        return result

    def restore_semi_transparent(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        estimated_alpha: float = 0.35,
        watermark_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Inverse Alpha Matting:
        I = alpha * W + (1 - alpha) * B
        => B = (I - alpha * W) / (1 - alpha)
        Preserves original underlying pixels without hallucination!
        """
        if mask is None or np.count_nonzero(mask) == 0:
            return img_bgr.copy()

        h, w = img_bgr.shape[:2]
        result = img_bgr.copy().astype(np.float32)
        
        wm = np.array(watermark_color, dtype=np.float32)
        alpha = np.clip(estimated_alpha, 0.05, 0.95)
        
        mask_idx = mask > 127
        # Invert alpha blending for masked pixels
        restored = (result[mask_idx] - alpha * wm) / (1.0 - alpha)
        result[mask_idx] = np.clip(restored, 0, 255)
        
        return result.astype(np.uint8)
