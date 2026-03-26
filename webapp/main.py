from __future__ import annotations

import io
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from webapp.config import DATA_DIR, STATIC_DIR
from webapp.models import JobResponse, MaskSource, ProcessRequest
from webapp.remover import detect_auto_mask, remove_watermark
from webapp.store import JobStore

app = FastAPI(title="Local Watermark Remover")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

store = JobStore(DATA_DIR)
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wm-worker")
futures: Dict[str, Future] = {}
futures_lock = Lock()


def _job_to_response(job: dict) -> JobResponse:
    job_id = job["id"]
    return JobResponse(
        id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        message=job.get("message", ""),
        error=job.get("error", ""),
        device=job.get("device"),
        model_id=job.get("model_id"),
        mask_source=job.get("mask_source"),
        input_url=f"/api/jobs/{job_id}/assets/input" if job.get("input_file") else None,
        auto_mask_url=f"/api/jobs/{job_id}/assets/auto_mask" if job.get("auto_mask_file") else None,
        manual_mask_url=f"/api/jobs/{job_id}/assets/manual_mask" if job.get("manual_mask_file") else None,
        output_url=f"/api/jobs/{job_id}/assets/output" if job.get("output_file") else None,
    )


def _get_job_or_404(job_id: str) -> dict:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _save_upload_as_rgb(upload_bytes: bytes, output_path) -> None:
    image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
    image.save(output_path)


def _normalize_uploaded_mask(mask_bytes: bytes, target_size) -> np.ndarray:
    raw = Image.open(io.BytesIO(mask_bytes))
    if "A" in raw.getbands():
        alpha = np.array(raw.getchannel("A"), dtype=np.uint8)
        gray = np.array(raw.convert("L"), dtype=np.uint8)
        merged = np.maximum(alpha, gray)
    else:
        merged = np.array(raw.convert("L"), dtype=np.uint8)

    mask = Image.fromarray(merged).resize(target_size, Image.Resampling.NEAREST)
    mask_np = (np.array(mask, dtype=np.uint8) > 10).astype(np.uint8) * 255
    return mask_np


def _ensure_auto_mask(job_id: str, req: ProcessRequest) -> dict:
    job = _get_job_or_404(job_id)
    if job.get("auto_mask_file") and store.file_path(job_id, job["auto_mask_file"]).exists():
        return job

    input_file = job.get("input_file")
    if not input_file:
        raise RuntimeError("input file missing")
    input_path = store.file_path(job_id, input_file)
    image = _load_image(input_path)
    auto = detect_auto_mask(
        image=image,
        min_confidence=req.min_confidence,
        expand=req.expand,
        roi_w=req.roi_w,
        roi_h=req.roi_h,
    )
    auto_mask_path = store.file_path(job_id, "auto_mask.png")
    fallback_mask_path = store.file_path(job_id, "fallback_mask.png")
    Image.fromarray(auto.mask).save(auto_mask_path)
    Image.fromarray(auto.fallback_mask).save(fallback_mask_path)
    box_info = f"{auto.merged_box}" if auto.merged_box else "None"
    return store.update_job(
        job_id,
        auto_mask_file="auto_mask.png",
        fallback_mask_file="fallback_mask.png",
        message=f"auto mask generated, boxes={auto.box_count}, merged_box={box_info}",
    )


def _run_process_job(job_id: str, req: ProcessRequest) -> None:
    try:
        job = store.update_job(job_id, status="running", error="", message="processing...")
        input_path = store.file_path(job_id, job["input_file"])
        image = _load_image(input_path)

        if req.mask_source == MaskSource.manual:
            manual_file = job.get("manual_mask_file")
            if not manual_file:
                raise RuntimeError("manual mask not found, upload manual mask first")
            mask_path = store.file_path(job_id, manual_file)
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            fallback_np = mask_np.copy()
        else:
            job = _ensure_auto_mask(job_id, req)
            mask_path = store.file_path(job_id, job["auto_mask_file"])
            fallback_path = store.file_path(job_id, job["fallback_mask_file"])
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            fallback_np = np.array(Image.open(fallback_path).convert("L"), dtype=np.uint8)

        result = remove_watermark(
            image=image,
            mask_np=mask_np,
            fallback_mask_np=fallback_np,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            disable_cv2_fallback=req.disable_cv2_fallback,
            roi_w=req.roi_w,
            roi_h=req.roi_h,
        )
        output_path = store.file_path(job_id, "output.png")
        result.image.save(output_path)
        msg = "completed"
        if result.used_cv2_fallback:
            msg += " (cv2 fallback used)"
        store.update_job(
            job_id,
            status="completed",
            message=msg,
            output_file="output.png",
            device=result.device,
            model_id=result.model_id,
            mask_source=req.mask_source.value,
        )
    except Exception as exc:
        detail = f"{exc}\n{traceback.format_exc(limit=3)}"
        store.update_job(job_id, status="failed", error=detail, message="processing failed")
    finally:
        with futures_lock:
            futures.pop(job_id, None)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/jobs", response_model=JobResponse)
async def create_job(file: UploadFile = File(...)) -> JobResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only image upload is supported")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty upload")

    job = store.create_job()
    job_id = job["id"]
    input_path = store.file_path(job_id, "input.png")
    try:
        _save_upload_as_rgb(payload, input_path)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")

    job = store.update_job(job_id, input_file="input.png", message="image uploaded")
    return _job_to_response(job)


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    return _job_to_response(_get_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/assets/{asset}")
def get_asset(job_id: str, asset: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    p = store.resolve_asset(job, asset)
    if p is None:
        raise HTTPException(status_code=404, detail="asset not found")
    media = "image/png"
    if asset == "input":
        media = "image/png"
    return FileResponse(p, media_type=media)


@app.post("/api/jobs/{job_id}/auto-mask", response_model=JobResponse)
def generate_auto_mask(job_id: str, req: Optional[ProcessRequest] = None) -> JobResponse:
    job = _get_job_or_404(job_id)
    input_file = job.get("input_file")
    if not input_file:
        raise HTTPException(status_code=400, detail="input image missing")
    request = req or ProcessRequest()
    job = _ensure_auto_mask(job_id, request)
    return _job_to_response(job)


@app.post("/api/jobs/{job_id}/manual-mask", response_model=JobResponse)
async def upload_manual_mask(job_id: str, file: UploadFile = File(...)) -> JobResponse:
    job = _get_job_or_404(job_id)
    input_file = job.get("input_file")
    if not input_file:
        raise HTTPException(status_code=400, detail="input image missing")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty mask")

    input_image = _load_image(store.file_path(job_id, input_file))
    mask_np = _normalize_uploaded_mask(payload, input_image.size)
    if np.count_nonzero(mask_np) == 0:
        raise HTTPException(status_code=400, detail="manual mask is empty")

    manual_path = store.file_path(job_id, "manual_mask.png")
    Image.fromarray(mask_np).save(manual_path)
    job = store.update_job(job_id, manual_mask_file="manual_mask.png", message="manual mask uploaded")
    return _job_to_response(job)


@app.post("/api/jobs/{job_id}/process", response_model=JobResponse)
def process_job(job_id: str, req: ProcessRequest) -> JobResponse:
    job = _get_job_or_404(job_id)
    if not job.get("input_file"):
        raise HTTPException(status_code=400, detail="input image missing")

    with futures_lock:
        existing = futures.get(job_id)
        if existing is not None and not existing.done():
            raise HTTPException(status_code=409, detail="job is already running")
        updated = store.update_job(job_id, status="queued", error="", message="queued")
        futures[job_id] = executor.submit(_run_process_job, job_id, req)

    return _job_to_response(updated)


@app.on_event("shutdown")
def shutdown_event() -> None:
    executor.shutdown(wait=False, cancel_futures=False)
