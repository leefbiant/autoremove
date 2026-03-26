import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        return self.root_dir / job_id

    def job_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def create_job(self) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        now = utc_now_iso()
        job = {
            "id": job_id,
            "status": "uploaded",
            "created_at": now,
            "updated_at": now,
            "message": "",
            "error": "",
            "device": None,
            "model_id": None,
            "mask_source": None,
            "input_file": None,
            "auto_mask_file": None,
            "fallback_mask_file": None,
            "manual_mask_file": None,
            "output_file": None,
        }
        with self._lock:
            d = self.job_dir(job_id)
            d.mkdir(parents=True, exist_ok=True)
            self._save_unlocked(job_id, job)
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        p = self.job_json_path(job_id)
        if not p.exists():
            return None
        with self._lock:
            return json.loads(p.read_text(encoding="utf-8"))

    def update_job(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        with self._lock:
            job = self._load_unlocked(job_id)
            if job is None:
                raise KeyError(f"job not found: {job_id}")
            job.update(updates)
            job["updated_at"] = utc_now_iso()
            self._save_unlocked(job_id, job)
            return job

    def file_path(self, job_id: str, file_name: str) -> Path:
        return self.job_dir(job_id) / file_name

    def resolve_asset(self, job: Dict[str, Any], asset: str) -> Optional[Path]:
        key_map = {
            "input": "input_file",
            "auto_mask": "auto_mask_file",
            "manual_mask": "manual_mask_file",
            "output": "output_file",
        }
        key = key_map.get(asset)
        if key is None:
            return None
        file_name = job.get(key)
        if not file_name:
            return None
        p = self.file_path(job["id"], file_name)
        if not p.exists():
            return None
        return p

    def _load_unlocked(self, job_id: str) -> Optional[Dict[str, Any]]:
        p = self.job_json_path(job_id)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _save_unlocked(self, job_id: str, job: Dict[str, Any]) -> None:
        p = self.job_json_path(job_id)
        p.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

