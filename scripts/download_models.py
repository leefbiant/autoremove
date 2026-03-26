#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from diffusers import StableDiffusionInpaintPipeline


def load_manifest(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("manifest.models must be a list")
    return models


def iter_model_ids(models: Iterable[dict], include_optional: bool) -> Iterable[str]:
    for item in models:
        model_id = item.get("id")
        if not model_id:
            continue
        optional = bool(item.get("optional", False))
        if optional and not include_optional:
            continue
        yield model_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Download local inpainting models from manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("models/manifest.json"),
        help="Path to model manifest json",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download optional models",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"manifest not found: {args.manifest}")

    models = load_manifest(args.manifest)
    ids = list(iter_model_ids(models, include_optional=args.include_optional))
    if not ids:
        print("No models to download.")
        return 0

    for model_id in ids:
        print(f"[MODEL] downloading: {model_id}")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
        del pipe
        print(f"[MODEL] downloaded: {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

