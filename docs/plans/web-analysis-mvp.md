# 사진 분석 웹앱 구현 계획

## 구현 범위

현재 YOLO11n control 모델로 사진 업로드, 탐지 결과 표시, SQLite 저장과 기록 조회를 구현한다.
맥북에서 실행하는 개인용 첫 버전이며 휴대폰 크기의 화면도 지원한다. 로그인,
촬영 조건 자동 판정과 변화 추이 그래프는 이번 구현 범위에 포함하지 않는다.
얼굴을 5개 부위로 나누는 분석은 프로젝트 개발 범위에서 제외한다.
후속 개발 방향은 [웹 서비스 개선 계획](web-service-improvements.md)을 참고한다.

## 구성

화면은 `web/`, HTTP API는 `backend/src/acne_care/api/`, 분석 절차는 `application/`,
모델과 SQLite 구현은 `adapters/`에 둔다. 기존 `AcneDetector` 계약과 `Detection`을 재사용한다.
학습 코드는 서비스에서 호출하지 않는다. FastAPI가 화면과 API를 같은 주소에서 제공한다.

SQLite의 `analyses`에는 촬영일, 분석 시각, 이미지 크기, 파일 위치와 모델 설정을 저장한다.
`detections`에는 분석 ID, 좌표, confidence와 클래스 ID를 저장한다. 전체 개수는 탐지 행에서
계산한다. 모델 ID, 가중치 SHA-256, threshold와 입력 크기는 분석 당시 값으로 보존한다.
사진은 DB 외부 폴더에 보관하고 EXIF 방향을 반영한 뒤 메타데이터를 제거해 저장한다.

## 개발 순서

- 데이터 저장과 조회 테스트를 먼저 작성하고 SQLite 저장소를 구현한다.
- 이미지 검증과 모델 호출, 결과 이미지 생성 및 저장을 분석 서비스로 구현한다.
- 업로드, 목록, 상세와 이미지 조회 API를 연결하고 오류 응답을 검증한다.
- 업로드 미리보기, 결과 상자와 confidence, 기록 목록을 보여 주는 반응형 화면을 구현한다.
- 가짜 탐지기를 주입한 자동 테스트와 실제 control 모델의 추론을 각각 확인한다.
- 실제 브라우저에서 사진 분석과 기록 재열기, 서버 재시작 후 보존을 확인한다.

## 성공 기준

사용자가 촬영일과 사진을 제출하면 실제 탐지 결과가 표시되고, 새로고침과 서버 재시작 후에도
동일 기록을 다시 열 수 있어야 한다. 잘못된 파일, 용량 초과와 모델 오류는 성공 결과로 저장하지
않는다. 테스트용 결과와 실제 모델 실행 증거를 구분한다. 사용자 파일과 DB는 Git에 포함하지 않는다.

## 실행과 검증

구체적인 실행 옵션은 `python -m acne_care.api --help`를 기준으로 확인한다. 자동 테스트는
`PYTHONPATH=backend/src .venv/training/bin/python -m unittest discover -s backend/tests`로 실행한다.
모델은 기존 control 가중치를 사용하며 학습 데이터와 학교 서버에 의존하지 않는다.

기존 학습 가상환경에서 아래 명령으로 서비스 의존성을 추가하고 실행한다.

```bash
.venv/training/bin/python -m pip install -r backend/requirements-service.txt
PYTHONPATH=backend/src .venv/training/bin/python -m acne_care.api --port 8000
```

브라우저에서 `http://127.0.0.1:8000/`을 연다. 서버를 다시 시작할 때도 같은 명령을 사용한다.
기본 저장 위치는 `uploads/service/`이며 사진 폴더와 `acne-care.sqlite3`를 함께 보관해야 한다.
임시 실행과 테스트에는 `ACNE_STORAGE_DIR`로 별도 폴더를 지정한다. 기존 DB와 사진을 지우면
분석 기록을 복원할 수 없으므로 Git 브랜치를 정리할 때도 이 폴더는 삭제하지 않는다.

선택 모델은 `models/detector-baseline.yaml`, 실행 장치는 `api/settings.py`를 기준으로 확인한다. `ACNE_WEIGHTS_PATH`,
`ACNE_MODEL_ID`, `ACNE_MODEL_SHA256`, `ACNE_DEVICE`로 명시적으로 변경할 수 있다. 모델을
교체해도 이미 저장한 기록의 모델 정보와 탐지 결과는 바뀌지 않는다. 이번 실행은 로그인 없는
개인용 로컬 모드이므로 서버는 `127.0.0.1`에만 연결한다.

파일 업로드와 테스트 구성은 [FastAPI 공식 문서](https://fastapi.tiangolo.com/tutorial/request-files/)와
[테스트 안내](https://fastapi.tiangolo.com/tutorial/testing/)를 참고한다.
