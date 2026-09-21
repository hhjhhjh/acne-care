from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LegacyModelArtifact:
    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("sha256 must be a 64-character hexadecimal string")


def verify_legacy_model(artifact: LegacyModelArtifact) -> None:
    if not artifact.path.is_file():
        raise FileNotFoundError(f"legacy model file does not exist: {artifact.path}")

    actual_size = artifact.path.stat().st_size
    if actual_size != artifact.size_bytes:
        raise ValueError(
            f"legacy model size mismatch: expected {artifact.size_bytes}, got {actual_size}"
        )

    digest = sha256()
    with artifact.path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != artifact.sha256.lower():
        raise ValueError(
            f"legacy model SHA-256 mismatch: expected {artifact.sha256}, "
            f"got {actual_sha256}"
        )

