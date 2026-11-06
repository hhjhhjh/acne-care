# 웹 서비스 실행 안내

## 실행 환경

웹 서비스는 저장소 최상위 폴더에서 실행한다. Python 3.11 이상과 Ultralytics 및 서비스
의존성이 필요하다. 아래 명령은 가상환경을 새로 만들 때 사용한다. 기존 학습 환경이 있다면
가상환경 생성과 학습 의존성 설치를 건너뛰고 서비스 의존성만 추가한다.

```bash
python3 -m venv .venv/training
.venv/training/bin/python -m pip install -r backend/requirements-training.txt
.venv/training/bin/python -m pip install -r backend/requirements-service.txt
```

`python3 --version`으로 Python 버전을 확인한다. 기본 실행 장치는 CPU이며 로컬 사진 분석에
학교 서버 접속이나 데이터셋이 필요하지 않다. 설치 과정에서는 패키지 다운로드가 필요하다.

## 모델 준비

가중치는 [모델 관리 안내](../models/README.md)에 따라 별도로 준비한다. 기본 파일 위치와
SHA-256 확인값은 [모델 설정](../models/detector-baseline.yaml)을 기준으로 한다.
파일이 없거나 확인값이 다르면 서버는 시작 단계에서 오류를 낸다. 미학습 초기 가중치
`yolo11n.pt`는 서비스용 학습 결과를 대신하지 않는다.

## 서버 실행

```bash
PYTHONPATH=backend/src .venv/training/bin/python -m acne_care.api --port 8000
```

서버를 실행한 컴퓨터의 브라우저에서 `http://127.0.0.1:8000/`을 연다. 사용 가능한 실행 옵션은
`PYTHONPATH=backend/src .venv/training/bin/python -m acne_care.api --help`로 확인한다.
기본 서버는 로그인 없는 로컬 전용이며 `127.0.0.1`에만 연결한다. 휴대폰 화면 크기 대응과
실제 휴대폰 접속은 구분한다. 휴대폰 접속 환경과 직접 촬영은 후속 개발 범위이다.

## 저장과 설정

기본 저장 위치는 `uploads/service/`이다. `acne-care.sqlite3`에는 분석과 탐지 결과를 저장하고,
분석 ID별 폴더에는 방향 보정과 메타데이터 제거를 거친 사진 및 탐지 결과 사진을 저장한다.
DB에 기록한 사진 경로는 저장 폴더 기준 상대 경로이므로 백업과 복원 시 DB와 사진 폴더를
함께 보관한다. Git 브랜치 정리는 이 폴더를 삭제하는 절차가 아니다.

저장 경로와 모델을 바꿀 때는 아래 환경변수를 명시적으로 지정한다. 기본값의 정본은
[`api/settings.py`](../backend/src/acne_care/api/settings.py)와 모델 설정 파일이다.

| 환경변수 | 용도 |
|---|---|
| `ACNE_STORAGE_DIR` | 사진을 저장할 폴더를 지정한다. 별도 DB 경로를 지정하지 않으면 DB도 이 폴더에 둔다. |
| `ACNE_DATABASE_PATH` | DB 파일의 위치를 별도로 지정한다. |
| `ACNE_WEIGHTS_PATH` | 실행할 모델 가중치의 파일 경로를 지정한다. |
| `ACNE_MODEL_ID` | 분석 결과에 보존할 모델 식별자를 지정한다. |
| `ACNE_MODEL_SHA256` | 실행 전 가중치의 확인값을 지정한다. |
| `ACNE_DEVICE` | 추론에 사용할 실행 장치를 지정한다. |

다른 가중치로 교체할 때는 모델 식별자와 SHA-256도 함께 지정한다. 모델을 교체해도 이미
저장된 기록의 모델 정보와 탐지 결과는 바뀌지 않는다. 임시 검증에는 별도 저장 폴더를 사용한다.

## 구성과 검증

FastAPI가 `web/`의 화면과 HTTP API를 같은 주소에서 제공한다. `application/`은 이미지 검증과
탐지·저장을 진행하고 `adapters/`는 YOLO와 SQLite를 연결한다. 탐지 형식과 계약은 기존
`domain/`과 `ports/`를 사용한다. 웹 서비스가 학습 코드를 호출하지 않도록 분리한다.

```bash
PYTHONPATH=backend/src .venv/training/bin/python -m unittest discover -s backend/tests
node --check web/app.js
```

JavaScript 문법 검사에만 Node.js가 필요하다. 웹앱 실행에는 프런트엔드 빌드가 필요하지 않다.
자동 테스트는 임시 저장소와 테스트용 탐지기를 사용한다. 실제 모델과 브라우저 동작을
확인할 때는 별도 저장 폴더에서 업로드, 기록 재열기와 서버 재시작 후 보존을 확인한다.
