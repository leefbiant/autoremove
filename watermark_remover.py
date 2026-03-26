#!/usr/bin/env python3
"""
Automatic watermark remover (offline, local model only).

Features:
1) Auto-detect text watermark in bottom-right area and generate mask.
2) Auto-select device: prefer MPS, fallback to CPU.
3) Local-only inpainting with cached diffusers model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import easyocr
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(f"easyocr import failed: {exc}")

try:
    from diffusers import StableDiffusionInpaintPipeline
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit(f"diffusers import failed: {exc}")


Box = Tuple[int, int, int, int]  # x0, y0, x1, y1 (inclusive)


def pick_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clamp_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Box:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def merge_boxes(boxes: Sequence[Box]) -> Optional[Box]:
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def get_bottom_right_roi(w: int, h: int, ratio_w: float = 0.45, ratio_h: float = 0.40) -> Box:
    x0 = int(round(w * (1.0 - ratio_w)))
    y0 = int(round(h * (1.0 - ratio_h)))
    return clamp_box(x0, y0, w - 1, h - 1, w, h)


def ocr_detect_text_boxes(
    image_bgr: np.ndarray,
    roi: Box,
    languages: Sequence[str],
    min_confidence: float,
) -> List[Box]:
    h, w = image_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = roi
    crop = image_bgr[ry0 : ry1 + 1, rx0 : rx1 + 1]
    reader = easyocr.Reader(list(languages), gpu=False, verbose=False)
    raw = reader.readtext(crop, detail=1, paragraph=False)
    boxes: List[Box] = []

    for item in raw:
        poly, _text, conf = item
        if conf < min_confidence:
            continue
        xs = [int(round(p[0])) for p in poly]
        ys = [int(round(p[1])) for p in poly]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        bx0 += rx0
        bx1 += rx0
        by0 += ry0
        by1 += ry0
        boxes.append(clamp_box(bx0, by0, bx1, by1, w, h))

    # Fallback: if confidence filter removed all, keep low-confidence candidates.
    if not boxes and raw:
        for item in raw:
            poly, _text, _conf = item
            xs = [int(round(p[0])) for p in poly]
            ys = [int(round(p[1])) for p in poly]
            bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
            bx0 += rx0
            bx1 += rx0
            by0 += ry0
            by1 += ry0
            boxes.append(clamp_box(bx0, by0, bx1, by1, w, h))

    return boxes


def has_strong_text(
    image_bgr: np.ndarray,
    roi: Box,
    min_confidence: float = 0.45,
    languages: Sequence[str] = ("ch_sim", "en"),
) -> bool:
    rx0, ry0, rx1, ry1 = roi
    crop = image_bgr[ry0 : ry1 + 1, rx0 : rx1 + 1]
    reader = easyocr.Reader(list(languages), gpu=False, verbose=False)
    raw = reader.readtext(crop, detail=1, paragraph=False)
    for _poly, _text, conf in raw:
        if conf >= min_confidence:
            return True
    return False


def build_mask_from_boxes(
    image_bgr: np.ndarray,
    w: int,
    h: int,
    boxes: Sequence[Box],
    expand_px: int,
    gaussian_blur_px: int,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        x0 -= expand_px
        y0 -= expand_px
        x1 += expand_px
        y1 += expand_px
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, w, h)

        # Pixel-level text extraction via top-hat to suppress smooth bright background.
        region = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, hat_kernel)
        thr = max(12.0, float(np.percentile(tophat, 88)))
        binary = (tophat >= thr).astype(np.uint8) * 255

        # Constrain to brighter pixels; watermark text is generally bright.
        bright_thr = max(140.0, float(np.percentile(gray, 60)))
        bright = (gray >= bright_thr).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(binary, bright)

        k = np.ones((3, 3), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k, iterations=1)
        candidate = cv2.dilate(candidate, k, iterations=1)

        # Remove extremely large connected regions (usually non-text structures).
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), connectivity=8)
        filtered = np.zeros_like(candidate)
        region_area = max(1, gray.shape[0] * gray.shape[1])
        max_comp = max(64, int(region_area * 0.08))
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if 8 <= area <= max_comp:
                filtered[labels == lab] = 255

        ratio = float(np.count_nonzero(filtered)) / float(region_area)
        if ratio < 0.0015:
            # If extraction fails, fallback to rectangle.
            cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)
        else:
            mask[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(mask[y0 : y1 + 1, x0 : x1 + 1], filtered)

    if gaussian_blur_px > 0:
        # Keep odd kernel size.
        k = gaussian_blur_px if gaussian_blur_px % 2 == 1 else gaussian_blur_px + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    return mask


def fallback_mask_from_default_region(w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    # Conservative fallback: bottom-right rectangle.
    x0 = int(round(w * 0.84))
    y0 = int(round(h * 0.90))
    x1 = w - 1
    y1 = h - 1
    cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)
    return mask


def rectangle_mask_from_boxes(w: int, h: int, boxes: Sequence[Box], expand_px: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        x0 -= expand_px
        y0 -= expand_px
        x1 += expand_px
        y1 += expand_px
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, w, h)
        cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)
    return mask


def crop_by_mask_with_padding(image: Image.Image, mask: Image.Image, pad: int = 64) -> Tuple[Image.Image, Image.Image, Box]:
    m = np.array(mask.convert("L"))
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        raise ValueError("mask is empty")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(image.width - 1, x1 + pad)
    y1 = min(image.height - 1, y1 + pad)
    box: Box = (x0, y0, x1, y1)
    crop = (x0, y0, x1 + 1, y1 + 1)
    return image.crop(crop), mask.crop(crop), box


def load_local_inpaint_pipeline(model_ids: Sequence[str], device: str) -> Tuple[StableDiffusionInpaintPipeline, str]:
    last_err: Optional[Exception] = None
    for model_id in model_ids:
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, local_files_only=True)
            # Watermark removal is an image restoration task; disable safety checker
            # to avoid false-positive black outputs.
            pipe.safety_checker = None
            pipe.requires_safety_checker = False
            pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to(device)
            return pipe, model_id
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"cannot load local inpaint model from {list(model_ids)}; last error: {last_err}")


def inpaint_image(
    image: Image.Image,
    mask: Image.Image,
    device: str,
    steps: int,
    guidance_scale: float,
    seed: int,
    model_ids: Sequence[str],
) -> Tuple[Image.Image, str]:
    crop_img, crop_mask, (x0, y0, x1, y1) = crop_by_mask_with_padding(image, mask, pad=64)
    original_crop_size = crop_img.size

    # Fixed working size for stable diffusion inpainting.
    work_size = (512, 512)
    crop_img_512 = crop_img.resize(work_size, Image.Resampling.LANCZOS)
    crop_mask_512 = crop_mask.resize(work_size, Image.Resampling.NEAREST)

    pipe, model_id = load_local_inpaint_pipeline(model_ids, device=device)
    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)
    out = pipe(
        prompt="clean natural background, no text, no watermark",
        negative_prompt="text, watermark, logo, letters, symbols, artifacts",
        image=crop_img_512,
        mask_image=crop_mask_512,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    out = out.resize(original_crop_size, Image.Resampling.LANCZOS)

    # Keep unmasked area from original crop to reduce hallucination outside mask.
    original_crop_np = np.array(crop_img.convert("RGB"), dtype=np.float32)
    out_crop_np = np.array(out.convert("RGB"), dtype=np.float32)
    mask_np = np.array(crop_mask.convert("L"), dtype=np.float32) / 255.0
    mask_np = np.expand_dims(mask_np, axis=2)
    blended_np = original_crop_np * (1.0 - mask_np) + out_crop_np * mask_np
    blended = Image.fromarray(np.clip(blended_np, 0, 255).astype(np.uint8))

    result = image.copy()
    result.paste(blended, (x0, y0))
    return result, model_id


def cv2_inpaint_fallback(image: Image.Image, mask: Image.Image, radius: int = 3) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    m = np.array(mask.convert("L"))
    out = cv2.inpaint(bgr, m, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto watermark removal with local inpainting model.")
    parser.add_argument("--input", type=Path, default=Path("2.png"), help="Input image path")
    parser.add_argument("--output", type=Path, default=Path("2_out.png"), help="Output image path")
    parser.add_argument(
        "--mask-output",
        type=Path,
        default=Path("2_auto_mask.png"),
        help="Generated mask output path",
    )
    parser.add_argument(
        "--manual-mask",
        type=Path,
        default=None,
        help="Optional manual mask path; if provided, skip auto detection",
    )
    parser.add_argument("--steps", type=int, default=20, help="Inpainting steps")
    parser.add_argument("--guidance", type=float, default=7.0, help="Classifier free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.10,
        help="OCR confidence threshold for text boxes in auto-mask mode",
    )
    parser.add_argument(
        "--expand",
        type=int,
        default=12,
        help="Expand pixels around detected boxes when generating mask",
    )
    parser.add_argument(
        "--roi-w",
        type=float,
        default=0.45,
        help="Bottom-right ROI width ratio for text detection",
    )
    parser.add_argument(
        "--roi-h",
        type=float,
        default=0.40,
        help="Bottom-right ROI height ratio for text detection",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Local model id; can pass multiple, fallback in order",
    )
    parser.add_argument(
        "--disable-cv2-fallback",
        action="store_true",
        help="Disable cv2 inpaint fallback when SD output still contains obvious text artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input.exists():
        print(f"[ERROR] input not found: {args.input}")
        return 2

    image = Image.open(args.input).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    fallback_mask_np: Optional[np.ndarray] = None

    if args.manual_mask is not None:
        if not args.manual_mask.exists():
            print(f"[ERROR] manual mask not found: {args.manual_mask}")
            return 2
        mask_img = Image.open(args.manual_mask).convert("L")
        mask_np = np.array(mask_img)
        fallback_mask_np = mask_np.copy()
        print(f"[INFO] using manual mask: {args.manual_mask}")
    else:
        roi = get_bottom_right_roi(w, h, ratio_w=args.roi_w, ratio_h=args.roi_h)
        boxes = ocr_detect_text_boxes(
            image_bgr=image_bgr,
            roi=roi,
            languages=["ch_sim", "en"],
            min_confidence=args.min_conf,
        )
        if boxes:
            mask_np = build_mask_from_boxes(
                image_bgr=image_bgr,
                w=w,
                h=h,
                boxes=boxes,
                expand_px=max(0, args.expand),
                gaussian_blur_px=9,
            )
            fallback_mask_np = rectangle_mask_from_boxes(w, h, boxes, expand_px=max(8, args.expand))
            merged = merge_boxes(boxes)
            print(f"[INFO] auto-detected text boxes: {len(boxes)}, merged_box={merged}")
        else:
            mask_np = fallback_mask_from_default_region(w, h)
            fallback_mask_np = mask_np.copy()
            print("[WARN] no text box detected; using conservative bottom-right fallback mask")

        Image.fromarray(mask_np).save(args.mask_output)
        print(f"[INFO] auto mask saved: {args.mask_output}")

    if np.count_nonzero(mask_np) == 0:
        print("[ERROR] empty mask; nothing to inpaint")
        return 2

    device = pick_device()
    print(f"[INFO] selected device: {device} (prefer mps, fallback cpu)")

    model_ids = args.model_id or [
        "runwayml/stable-diffusion-inpainting",
        "sd2-community/stable-diffusion-2-inpainting",
    ]

    result, used_model = inpaint_image(
        image=image,
        mask=Image.fromarray(mask_np),
        device=device,
        steps=max(1, args.steps),
        guidance_scale=args.guidance,
        seed=args.seed,
        model_ids=model_ids,
    )

    # Quality gate: if obvious text is still present in mask ROI, fallback to cv2 inpaint.
    if not args.disable_cv2_fallback:
        mask_img = Image.fromarray(mask_np)
        try:
            roi_box = get_bottom_right_roi(w, h, ratio_w=args.roi_w, ratio_h=args.roi_h)
            result_bgr = cv2.cvtColor(np.array(result.convert("RGB")), cv2.COLOR_RGB2BGR)
            if has_strong_text(result_bgr, roi=roi_box, min_confidence=0.05):
                print("[WARN] strong text detected after SD inpaint, fallback to cv2 inpaint")
                fallback_mask_img = Image.fromarray(fallback_mask_np if fallback_mask_np is not None else mask_np)
                result = cv2_inpaint_fallback(image=image, mask=fallback_mask_img, radius=2)
        except Exception as exc:
            print(f"[WARN] fallback quality gate skipped: {exc}")

    result.save(args.output)
    print(f"[INFO] inpaint model: {used_model}")
    print(f"[INFO] output saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
