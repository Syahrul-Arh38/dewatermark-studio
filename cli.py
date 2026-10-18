#!/usr/bin/env python3
"""
cli.py - Batch Watermark Removal Command Line Tool
Drop-in replacement for the Google Colab script with bug fixes and enhancements:
- Dynamic multi-cluster stamp detection (no more center-averaging glitches)
- Dual-pass CLAHE contrast enhancement
- Multi-engine inpainting (Edge-guided, LaMa, Navier-Stokes)
- 100% Lossless preservation

Usage:
  python cli.py --input ./input_folder --output ./output_folder
  python cli.py --sample   # generates and cleans a test image
"""

import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.pipeline import WatermarkPipeline
from core.utils import save_image_lossless, create_sample_watermark_image


def parse_args():
    parser = argparse.ArgumentParser(description="Dewatermark Pro CLI - Batch Watermark Cleaner")
    parser.add_argument("--input", "-i", type=str, default="./samples", help="Input directory or image file")
    parser.add_argument("--output", "-o", type=str, default="./output", help="Output directory")
    parser.add_argument("--dilate", "-d", type=int, default=11, help="Dilation pixel radius (default: 11)")
    parser.add_argument("--sensitivity", "-s", type=float, default=0.5, help="Detection sensitivity 0.1 - 1.0 (default: 0.5)")
    parser.add_argument("--method", "-m", type=str, default="auto", choices=["auto", "edge_guided", "lama", "navier_stokes"], help="Inpainting method")
    parser.add_argument("--no-ocr", action="store_true", help="Disable EasyOCR scan")
    parser.add_argument("--sample", action="store_true", help="Generate a sample watermarked image and clean it")
    return parser.parse_args()


def process_file(pipeline: WatermarkPipeline, input_path: str, output_path: str, args):
    img = cv2.imread(input_path)
    if img is None:
        print(f"[SKIP] Cannot read image: {input_path}")
        return False
        
    res = pipeline.process(
        img_bgr=img,
        use_ocr=not args.no_ocr,
        use_heuristics=True,
        sensitivity=args.sensitivity,
        dilation_px=args.dilate,
        inpaint_method=args.method
    )
    
    save_image_lossless(res['cleaned_bgr'], output_path)
    print(f"[OK] {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({res['elapsed_sec']}s)")
    return True


def main():
    args = parse_args()
    
    if args.sample:
        sample_path = "./samples/watermark_sample.jpg"
        out_sample = "./output/watermark_sample_cleaned.jpg"
        print(f"[*] Generating realistic watermark sample: {sample_path}")
        create_sample_watermark_image(sample_path)
        args.input = sample_path

    os.makedirs(args.output, exist_ok=True)
    pipeline = WatermarkPipeline()

    if os.path.isfile(args.input):
        filename = os.path.basename(args.input)
        out_file = os.path.join(args.output, f"cleaned_{filename}")
        print(f"[*] Processing single file: {args.input}")
        process_file(pipeline, args.input, out_file, args)
        print(f"[DONE] Cleaned image saved to: {out_file}")
        return

    # Batch directory
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    all_files = [f for f in glob.glob(os.path.join(args.input, "*")) if f.lower().endswith(valid_exts)]

    if not all_files:
        print(f"[!] No supported images found in '{args.input}'.")
        print(f"    Tip: Run 'python cli.py --sample' to generate and test with a sample image.")
        return

    print(f"[*] Starting batch processing of {len(all_files)} images...")
    start_all = time.time()
    success_count = 0

    for idx, filepath in enumerate(all_files, 1):
        filename = os.path.basename(filepath)
        out_path = os.path.join(args.output, filename)
        print(f"[{idx}/{len(all_files)}] Processing {filename}...")
        if process_file(pipeline, filepath, out_path, args):
            success_count += 1

    total_time = round(time.time() - start_all, 2)
    print(f"\n[COMPLETE] Successfully processed {success_count}/{len(all_files)} images in {total_time}s.")
    print(f"Output folder: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
