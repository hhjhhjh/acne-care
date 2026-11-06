from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from acne_care.adapters.sqlite_repository import SQLiteAnalysisRepository
from acne_care.api.settings import PROJECT_ROOT, Settings
from acne_care.application.analysis import AnalysisService, InvalidImage, ModelInfo
from acne_care.application.comparison import compare_analyses
from acne_care.ports.acne_detector import AcneDetector


def _public_record(record):
    result = {key: value for key, value in record.items() if key not in {"original_path", "annotated_path"}}
    result["original_image_url"] = f"/api/analyses/{record['id']}/image?view=original"
    result["annotated_image_url"] = f"/api/analyses/{record['id']}/image?view=annotated"
    return result


def create_app(settings: Settings | None = None, *, detector: AcneDetector | None = None,
               model_info: ModelInfo | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    if (detector is None) != (model_info is None):
        raise ValueError("injected detector and model metadata must be provided together")
    repository = SQLiteAnalysisRepository(settings.database_path)

    @asynccontextmanager
    async def lifespan(app):
        active_detector, active_info = detector, model_info
        if active_detector is None:
            from acne_care.adapters.yolo_detector import YOLOAcneDetector
            active_detector = YOLOAcneDetector(settings.weights_path, model_id=settings.model_id,
                                               expected_sha256=settings.expected_sha256,
                                               confidence_threshold=settings.confidence_threshold,
                                               image_size=settings.image_size, device=settings.device, nms_iou=settings.nms_iou)
            active_info = active_detector.info
        repository.initialize()
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        app.state.analysis_service = AnalysisService(active_detector, active_info, repository,
                                                      settings.storage_dir, max_pixels=settings.max_pixels)
        app.state.model_info = active_info
        yield

    app = FastAPI(title="acne-care", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_request_checks(request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.url.netloc:
                return JSONResponse(status_code=403, content={"detail": "이 화면에서 직접 분석을 요청해 주세요."})
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_upload_bytes + 65536:
                return JSONResponse(status_code=413, content={"detail": "사진은 10MB 이하로 선택해 주세요."})
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/health")
    def health():
        return {"ready": True, "model": asdict(app.state.model_info), "max_upload_bytes": settings.max_upload_bytes}

    @app.post("/api/analyses", status_code=201)
    def analyze(file: UploadFile = File(...), captured_on: date = Form(...)):
        try:
            data = file.file.read(settings.max_upload_bytes + 1)
        finally:
            file.file.close()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, "사진은 10MB 이하로 선택해 주세요.")
        if captured_on > date.today():
            raise HTTPException(422, "촬영일은 오늘 또는 이전 날짜를 선택해 주세요.")
        try:
            record = app.state.analysis_service.analyze(data, captured_on)
        except InvalidImage as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logging.getLogger(__name__).exception("Image analysis failed; no result returned")
            raise HTTPException(500, "분석을 완료하지 못했습니다. 잠시 후 다시 시도해 주세요.") from error
        return _public_record(record)

    @app.get("/api/analyses")
    def list_analyses(limit: int = Query(12, ge=1, le=100), offset: int = Query(0, ge=0)):
        listing = repository.list(limit, offset)
        return {**listing, "items": [_public_record(item) for item in listing["items"]]}

    def get_record(analysis_id):
        try:
            canonical = str(UUID(analysis_id))
        except ValueError:
            raise HTTPException(404, "분석 기록을 찾을 수 없습니다.")
        record = repository.get(canonical)
        if record is None:
            raise HTTPException(404, "분석 기록을 찾을 수 없습니다.")
        return record

    @app.get("/api/analyses/{analysis_id}")
    def analysis_detail(analysis_id: str):
        return _public_record(get_record(analysis_id))

    @app.get("/api/comparisons")
    def compare_records(previous_id: str, current_id: str):
        previous, current = get_record(previous_id), get_record(current_id)
        try:
            comparison = compare_analyses(previous, current)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"previous": _public_record(previous), "current": _public_record(current),
                "comparison": comparison}

    @app.get("/api/analyses/{analysis_id}/image")
    def analysis_image(analysis_id: str, view: str = Query("annotated", pattern="^(original|annotated)$")):
        record = get_record(analysis_id)
        image_path = settings.storage_dir / record[f"{view}_path"]
        if not image_path.is_file():
            raise HTTPException(404, "저장된 이미지 파일을 찾을 수 없습니다.")
        return FileResponse(image_path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=PROJECT_ROOT / "web", html=True), name="web")
    return app
