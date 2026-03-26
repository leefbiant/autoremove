from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import easyocr
from diffusers import StableDiffusionInpaintPipeline

from webapp.config import DEFAULT_MODEL_IDS

Box = Tuple[int, int, int, int]

_OCR_CACHE: Dict[Tuple[str, ...], easyocr.Reader] = {}
_OCR_LOCK = Lock()

_PIPELINE_CACHE: Dict[Tuple[str, str], StableDiffusionInpaintPipeline] = {}
_PIPELINE_LOCK = Lock()


@dataclass
class AutoMaskResult:
    mask: np.ndarray
    fallback_mask: np.ndarray
    box_count: int
    merged_box: Optional[Box]


@dataclass
class RemoveResult:
    image: Image.Image
    device: str
    model_id: str
    used_cv2_fallback: bool


def get_reader(languages: Sequence[str]) -> easyocr.Reader:
    key = tuple(languages)
    with _OCR_LOCK:
        reader = _OCR_CACHE.get(key)
        if reader is None:
            reader = easyocr.Reader(list(key), gpu=False, verbose=False)
            _OCR_CACHE[key] = reader
        return reader


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
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def get_bottom_right_roi(w: int, h: int, ratio_w: float = 0.45, ratio_h: float = 0.40) -> Box:
    x0 = int(round(w * (1.0 - ratio_w)))
    y0 = int(round(h * (1.0 - ratio_h)))
    return clamp_box(x0, y0, w - 1, h - 1, w, h)


def ocr_detect_text_boxes(image_bgr: np.ndarray, roi: Box, min_confidence: float) -> List[Box]:
    h, w = image_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = roi
    crop = image_bgr[ry0 : ry1 + 1, rx0 : rx1 + 1]
    raw = get_reader(("ch_sim", "en")).readtext(crop, detail=1, paragraph=False)
    boxes: List[Box] = []
    for poly, _txt, conf in raw:
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

    if not boxes and raw:
        for poly, _txt, _conf in raw:
            xs = [int(round(p[0])) for p in poly]
            ys = [int(round(p[1])) for p in poly]
            bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
            bx0 += rx0
            bx1 += rx0
            by0 += ry0
            by1 += ry0
            boxes.append(clamp_box(bx0, by0, bx1, by1, w, h))
    return boxes


def build_mask_from_boxes(image_bgr: np.ndarray, w: int, h: int, boxes: Sequence[Box], expand_px: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        x0, y0, x1, y1 = clamp_box(x0 - expand_px, y0 - expand_px, x1 + expand_px, y1 + expand_px, w, h)
        region = image_bgr[y0 : y1 + 1, x0 : x1 + 1]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, hat_kernel)
        thr = max(12.0, float(np.percentile(tophat, 88)))
        binary = (tophat >= thr).astype(np.uint8) * 255
        bright_thr = max(140.0, float(np.percentile(gray, 60)))
        bright = (gray >= bright_thr).astype(np.uint8) * 255
        candidate = cv2.bitwise_and(binary, bright)
        k = np.ones((3, 3), np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k, iterations=1)
        candidate = cv2.dilate(candidate, k, iterations=1)

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
            cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)
        else:
            mask[y0 : y1 + 1, x0 : x1 + 1] = np.maximum(mask[y0 : y1 + 1, x0 : x1 + 1], filtered)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask


def rectangle_mask_from_boxes(w: int, h: int, boxes: Sequence[Box], expand_px: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x0, y0, x1, y1) in boxes:
        x0, y0, x1, y1 = clamp_box(x0 - expand_px, y0 - expand_px, x1 + expand_px, y1 + expand_px, w, h)
        cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)
    return mask


def default_corner_mask(w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = int(round(w * 0.84))
    y0 = int(round(h * 0.90))
    cv2.rectangle(mask, (x0, y0), (w - 1, h - 1), color=255, thickness=-1)
    return mask


def detect_auto_mask(image: Image.Image, min_confidence: float = 0.10, expand: int = 12, roi_w: float = 0.45, roi_h: float = 0.40) -> AutoMaskResult:
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    roi = get_bottom_right_roi(w, h, ratio_w=roi_w, ratio_h=roi_h)
    boxes = ocr_detect_text_boxes(image_bgr=image_bgr, roi=roi, min_confidence=min_confidence)
    if boxes:
        mask = build_mask_from_boxes(image_bgr=image_bgr, w=w, h=h, boxes=boxes, expand_px=max(0, expand))
        fallback = rectangle_mask_from_boxes(w=w, h=h, boxes=boxes, expand_px=max(8, expand))
    else:
        mask = default_corner_mask(w, h)
        fallback = mask.copy()
    return AutoMaskResult(mask=mask, fallback_mask=fallback, box_count=len(boxes), merged_box=merge_boxes(boxes))


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
    crop = (x0, y0, x1 + 1, y1 + 1)
    return image.crop(crop), mask.crop(crop), (x0, y0, x1, y1)


def _get_pipeline(model_ids: Sequence[str], device: str) -> Tuple[StableDiffusionInpaintPipeline, str]:
    last_err: Optional[Exception] = None
    for model_id in model_ids:
        key = (model_id, device)
        with _PIPELINE_LOCK:
            pipe = _PIPELINE_CACHE.get(key)
        if pipe is not None:
            return pipe, model_id
        try:
            loaded = StableDiffusionInpaintPipeline.from_pretrained(model_id, local_files_only=True)
            loaded.safety_checker = None
            loaded.requires_safety_checker = False
            loaded.set_progress_bar_config(disable=True)
            loaded = loaded.to(device)
            with _PIPELINE_LOCK:
                _PIPELINE_CACHE[key] = loaded
            return loaded, model_id
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"cannot load local inpaint model from {list(model_ids)}; last error: {last_err}")


def _has_detectable_text(image: Image.Image, roi: Box, min_confidence: float = 0.05) -> bool:
    image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    rx0, ry0, rx1, ry1 = roi
    crop = image_bgr[ry0 : ry1 + 1, rx0 : rx1 + 1]
    raw = get_reader(("ch_sim", "en")).readtext(crop, detail=1, paragraph=False)
    for _poly, _txt, conf in raw:
        if conf >= min_confidence:
            return True
    return False


def _cv2_inpaint(image: Image.Image, mask: np.ndarray, radius: int = 2) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.inpaint(bgr, mask.astype(np.uint8), inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def _sd_inpaint(image: Image.Image, mask_np: np.ndarray, device: str, steps: int, guidance_scale: float, seed: int, model_ids: Sequence[str]) -> Tuple[Image.Image, str]:
    mask = Image.fromarray(mask_np.astype(np.uint8)).convert("L")
    crop_img, crop_mask, (x0, y0, _x1, _y1) = crop_by_mask_with_padding(image, mask, pad=64)
    original_crop_size = crop_img.size
    crop_img_512 = crop_img.resize((512, 512), Image.Resampling.LANCZOS)
    crop_mask_512 = crop_mask.resize((512, 512), Image.Resampling.NEAREST)

    pipe, model_id = _get_pipeline(model_ids=model_ids, device=device)
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

    original_crop_np = np.array(crop_img.convert("RGB"), dtype=np.float32)
    out_crop_np = np.array(out.convert("RGB"), dtype=np.float32)
    alpha = np.array(crop_mask.convert("L"), dtype=np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    blended_np = original_crop_np * (1.0 - alpha) + out_crop_np * alpha
    blended = Image.fromarray(np.clip(blended_np, 0, 255).astype(np.uint8))

    result = image.copy()
    result.paste(blended, (x0, y0))
    return result, model_id


def remove_watermark(
    image: Image.Image,
    mask_np: np.ndarray,
    fallback_mask_np: Optional[np.ndarray] = None,
    steps: int = 20,
    guidance_scale: float = 7.0,
    seed: int = 42,
    model_ids: Optional[Sequence[str]] = None,
    disable_cv2_fallback: bool = False,
    roi_w: float = 0.45,
    roi_h: float = 0.40,
) -> RemoveResult:
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_np) == 0:
        raise ValueError("mask is empty")

    if fallback_mask_np is None:
        fallback_mask_np = mask_np
    fallback_mask_np = (fallback_mask_np > 0).astype(np.uint8) * 255

    device = pick_device()
    result, used_model = _sd_inpaint(
        image=image,
        mask_np=mask_np,
        device=device,
        steps=max(1, int(steps)),
        guidance_scale=float(guidance_scale),
        seed=int(seed),
        model_ids=model_ids or DEFAULT_MODEL_IDS,
    )

    used_cv2_fallback = False
    if not disable_cv2_fallback:
        roi = get_bottom_right_roi(image.width, image.height, ratio_w=roi_w, ratio_h=roi_h)
        if _has_detectable_text(result, roi=roi, min_confidence=0.05):
            result = _cv2_inpaint(image=image, mask=fallback_mask_np, radius=2)
            used_cv2_fallback = True

    return RemoveResult(image=result, device=device, model_id=used_model, used_cv2_fallback=used_cv2_fallback)

