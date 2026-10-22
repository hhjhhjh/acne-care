from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any


def synchronize_torch_device(device: str, torch_runtime: Any) -> None:
    normalized_device = device.strip().lower()
    if normalized_device == "mps":
        if torch_runtime.backends.mps.is_available():
            torch_runtime.mps.synchronize()
        return

    if _is_cuda_device(normalized_device) and torch_runtime.cuda.is_available():
        torch_runtime.cuda.synchronize()


def build_model_metadata(model_id: str, weights: Path) -> dict[str, str]:
    normalized_model_id = model_id.strip()
    if not normalized_model_id:
        raise ValueError("model id must not be blank")
    return {
        "id": normalized_model_id,
        "weights": str(weights),
        "weights_sha256": _file_sha256(weights),
    }


def _is_cuda_device(device: str) -> bool:
    if device == "cuda" or device.startswith("cuda:"):
        return True
    return bool(device) and all(part.strip().isdigit() for part in device.split(","))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
