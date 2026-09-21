from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from acne_care.domain import Detection


@runtime_checkable
class AcneDetector(Protocol):
    """Contract shared by the legacy and future detector implementations."""

    @property
    def model_id(self) -> str:
        """Return a stable identifier that is stored with every analysis."""

    def detect(self, image: object) -> Sequence[Detection]:
        """Detect acne in one image and return pixel-coordinate boxes."""

