from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "webapp" / "static"
DATA_DIR = BASE_DIR / "data" / "jobs"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_IDS = [
    "runwayml/stable-diffusion-inpainting",
    "sd2-community/stable-diffusion-2-inpainting",
]

