# acne-care

얼굴 사진에서 여드름을 탐지하고 사진과 분석 결과를 기록하는 대학 수업 프로젝트이다. 시간에 따른 변화를 확인하는 서비스를 목표로 하며 의료 진단이나 질환 판정을 제공하지 않는다.

## 웹앱 기능

사진을 업로드하면 YOLO11n으로 여드름을 탐지한다. 결과 화면에는 탐지 위치, confidence와 전체 개수를 표시한다. confidence는 모델의 탐지 점수이다.

촬영일은 오늘 날짜를 기본값으로 제공하며 다른 날짜를 선택할 수 있다. 사진과 결과는 SQLite 및 로컬 사진 폴더에 보관하고 분석 기록에서 다시 조회한다.

휴대폰 화면 크기에 대응하는 웹앱이며 기본 실행은 맥북의 로컬 전용이다. 서버를 실행한 컴퓨터에서 접속한다. 얼굴을 5개 부위로 나누는 기능은 개발 범위에서 제외했다. 개발 범위와 후속 방향은 [웹 서비스 개선 계획](docs/plans/web-service-improvements.md)에 둔다.

## 로컬 실행

Python 3.11 이상과 학습된 모델 가중치가 필요하다. 가중치는 Git에 포함하지 않으므로 저장소를 복제한 뒤 별도로 준비해야 한다. 사용할 파일과 확인 방법은 [모델 관리 안내](models/README.md)를 따른다. 웹앱 추론에는 학습 데이터셋이나 학교 GPU 접속이 필요하지 않다.

기존 `.venv/training` 환경이 준비되어 있다면 저장소 최상위 폴더에서 다음 명령으로 실행한다. 새 환경 구성과 저장 위치 변경은 [웹 서비스 실행 안내](docs/web-service.md)를 참고한다.

```bash
.venv/training/bin/python -m pip install -r backend/requirements-service.txt
PYTHONPATH=backend/src .venv/training/bin/python -m acne_care.api --port 8000
```

브라우저에서 `http://127.0.0.1:8000/`을 연다. 이 주소는 서버를 실행한 컴퓨터에서 사용한다. 기본 저장 위치는 `uploads/service/`이며 기록을 보관하려면 DB와 사진 폴더를 함께 유지해야 한다.

## 개발 구조

모델 학습과 웹 서비스 실행을 분리한다. 탐지기와 저장소는 인터페이스를 통해 분석 로직에 연결한다.

| 경로 | 역할 |
|---|---|
| `web/` | 사진 업로드, 결과와 기록 조회 화면을 제공한다. |
| `backend/src/acne_care/api/` | HTTP 요청을 처리하고 웹 화면을 제공한다. |
| `backend/src/acne_care/application/` | 이미지 검증, 탐지와 결과 저장 절차를 담당한다. |
| `backend/src/acne_care/adapters/` | YOLO 실행과 SQLite 저장소를 구현한다. |
| `backend/src/acne_care/domain/` 및 `ports/` | 탐지 결과의 형식과 탐지기 계약을 정의한다. |
| `ml/` | 데이터 준비, 학습과 평가 실행 코드를 둔다. |

자동 테스트는 다음 명령으로 실행한다.

```bash
PYTHONPATH=backend/src .venv/training/bin/python -m unittest discover -s backend/tests
```

## 한계와 자료

로그인 없는 개인용 로컬 프로토타입이다. 탐지 누락이나 오탐이 있을 수 있으며 confidence는 의학적 확률을 뜻하지 않는다. 사진, 데이터셋, DB와 모델 가중치는 Git 추적 대상에서 제외한다.

기존 `acne-CV`는 초기 비교 자료로 활용하며 현재 서비스는 독립된 구조로 개발한다. 얼굴 위치만으로 장기 상태나 질환의 원인을 추정하는 문구는 사용하지 않는다.

- [v2·v3 비교 실험](docs/experiments/yolo11n-dataset-v3-comparison.md)에서 기본 모델의 학습 조건과 평가 결과를 확인할 수 있다.
- [데이터 관리 안내](data/README.md)에서 데이터 출처와 라벨 검토 방법을 확인할 수 있다.
- [이전 프로젝트 조사](docs/legacy-acne-cv-audit.md)와 [보관 자료](references/acne-cv/README.md)에서 재사용 판단의 근거를 확인할 수 있다.
