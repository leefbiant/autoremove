from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MaskSource(str, Enum):
    auto = "auto"
    manual = "manual"


class ProcessRequest(BaseModel):
    mask_source: MaskSource = MaskSource.auto
    steps: int = Field(default=20, ge=1, le=60)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=20.0)
    seed: int = 42
    min_confidence: float = Field(default=0.10, ge=0.0, le=1.0)
    expand: int = Field(default=12, ge=0, le=128)
    roi_w: float = Field(default=0.45, gt=0.05, le=1.0)
    roi_h: float = Field(default=0.40, gt=0.05, le=1.0)
    disable_cv2_fallback: bool = False


class JobResponse(BaseModel):
    id: str
    status: str
    created_at: str
    updated_at: str
    message: str = ""
    error: str = ""
    device: Optional[str] = None
    model_id: Optional[str] = None
    mask_source: Optional[str] = None
    input_url: Optional[str] = None
    auto_mask_url: Optional[str] = None
    manual_mask_url: Optional[str] = None
    output_url: Optional[str] = None

