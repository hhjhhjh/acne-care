from dataclasses import dataclass


CAPTURE_OPTIONS = {
    "capture_lighting": ("밝고 고른 조명", "어두운 조명", "그림자가 진 조명"),
    "capture_angle": ("정면", "왼쪽 측면", "오른쪽 측면"),
    "capture_distance": ("가까이", "보통", "멀리"),
    "capture_filter": ("사용 안 함", "사용함"),
}
CAPTURE_NOTE_LIMIT = 160
CAPTURE_COLUMNS = (*CAPTURE_OPTIONS, "capture_note")


@dataclass(frozen=True)
class CaptureDetails:
    capture_lighting: str | None = None
    capture_angle: str | None = None
    capture_distance: str | None = None
    capture_filter: str | None = None
    capture_note: str | None = None

    def __post_init__(self) -> None:
        for field, options in CAPTURE_OPTIONS.items():
            value = (getattr(self, field) or "").strip()
            if value and value not in options:
                raise ValueError("촬영 조건은 목록에 있는 항목에서 선택해 주세요.")
            object.__setattr__(self, field, value or None)

        note = self.capture_note or ""
        if "\n" in note or "\r" in note:
            raise ValueError("메모는 한 줄로 입력해 주세요.")
        note = note.strip()
        if len(note) > CAPTURE_NOTE_LIMIT:
            raise ValueError(f"메모는 {CAPTURE_NOTE_LIMIT}자 이하로 입력해 주세요.")
        object.__setattr__(self, "capture_note", note or None)
