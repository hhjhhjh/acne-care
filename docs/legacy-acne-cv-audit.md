# acne-CV 재사용 조사

## 조사 결론

`acne-CV`에서 바로 사용할 수 있는 핵심 산출물은 KerasCV YOLOv8 가중치이다. 얼굴 부위 landmark 인덱스, 좌표 환산 방식, 업로드부터 결과 기록까지의 화면 흐름은 새 구현의 참고 자료로 사용할 수 있다. Flask 애플리케이션과 데이터베이스 코드는 결합도가 높고 안전하지 않은 동작이 포함되어 있어 그대로 가져오지 않는다.

조사 기준 저장소는 `/Users/hwangjeonghyun/Desktop/github/acne-CV`이며, 확인한 Git HEAD는 `9d7fc56c`이다.

## 중복 폴더 판정

| 경로 | 역할 판정 | 처리 |
|---|---|---|
| `app/` | 업로드와 추론만 있는 초기 Flask 프로토타입이다. 47MB의 유효한 모델 가중치가 있다. | 가중치의 출처로 사용한다. Python 코드는 참고만 한다. |
| `app_db/` | 데이터베이스 기능을 붙인 중간 프로토타입이다. 모델 코드는 `app/`과 같다. 가중치 파일은 안내문뿐이라 실행할 수 없다. | 코드와 화면을 옮기지 않는다. |
| `PythonProject1/app_db/` | 기록 화면까지 포함한 최종 웹 프로토타입 후보이다. 유효한 가중치가 있다. | 사용자 흐름과 데이터 항목만 참고한다. |
| `model/` | 얼굴 부위 crop과 탐지 결과를 확인한 실험 노트북 모음이다. 유효한 가중치가 있다. | 실험 의도를 참고하되 제품 코드에서 실행하지 않는다. |
| `templates/`, `new-templates/` | 웹 화면 시안의 여러 사본이다. 일부 파일은 최종 Flask 경로와 연결되지 않는다. | 디자인 참고 자료로만 취급한다. |
| `cute/` | Windows 가상환경 전체가 저장된 폴더이다. | 재사용하지 않는다. 새 환경을 lockfile로 만든다. |
| `app_image/` | 음식 추천과 화면 시안에 쓰인 이미지 모음이다. | 현재 프로젝트 범위와 맞지 않아 옮기지 않는다. |
| `uploads/` | 사용자가 올린 것으로 보이는 얼굴 사진이다. | 개인정보일 수 있으므로 옮기지 않는다. |
| `DB.ipynb` | CSV를 SQLite로 옮기는 단발성 실험이다. 참조하는 CSV는 저장소에 없다. | 재사용하지 않는다. |

## 그대로 활용할 자료

### 모델 가중치

유효한 가중치는 HDF5 파일이며 크기는 47,016,328바이트이다. 다음 세 파일의 SHA-256 값은 모두 같다.

- `app/model/yolo_acne_detection.weights.h5`
- `model/yolo_acne_detection.weights.h5`
- `PythonProject1/app_db/model/yolo_acne_detection.weights.h5`

SHA-256은 `0aa53871402aa540ebebd6368151eb2d21cf4fa9fa1a41b742ac3facc18f0f35`이다. `app_db/model/yolo_acne_detection.weights.h5`는 “app 폴더에서 다운 받으시길”이라는 안내문이므로 제외한다.

가중치는 저장소에 직접 커밋하지 않는다. `scripts/import_legacy_model.sh`가 원본과 해시를 검증한 뒤 `models/legacy/acne-cv/`로 복사한다.

### 모델 구조

가중치를 불러오려면 이전 코드와 같은 모델 구조가 필요하다.

- 프레임워크는 TensorFlow와 KerasCV이다.
- detector는 `keras_cv.models.YOLOV8Detector`이다.
- backbone은 `yolo_v8_xs_backbone`이며 입력 rescaling을 포함한다.
- 클래스는 한 개이다.
- bounding box 형식은 `xyxy`이다.
- FPN depth는 5이다.
- 추론 입력 크기는 640×640이다.
- 이전 서비스의 confidence threshold는 0.4이다.

정확한 정보는 `references/acne-cv/model-manifest.yaml`에 한 번만 기록한다.

## 수정한 뒤 활용할 자료

### 얼굴 부위 landmark 인덱스

이마, 코, 볼과 턱을 구성하는 MediaPipe Face Mesh 인덱스가 남아 있다. 이전 구현은 인덱스로 polygon을 만들지 않고 각 부위의 최소·최대 좌표를 구해 직사각형 crop을 만들었다.

`app/model/model.py`와 `PythonProject1/app_db/model/model.py`의 유일한 차이는 왼쪽 볼과 오른쪽 볼 이름이다. `PythonProject1`에서 두 이름을 서로 바꿨지만 촬영 이미지의 미러링 여부가 기록되어 있지 않다. 새 구현에서는 방향이 표시된 테스트 이미지로 확인하기 전까지 어느 정의도 확정값으로 사용하지 않는다.

원본 인덱스는 `references/acne-cv/face-region-landmarks.yaml`에 보존한다. 이 파일은 검증 전 참고 자료이며 제품 설정이 아니다.

### 좌표 환산 방식

이전 코드는 얼굴 부위 crop을 640×640으로 늘려서 추론한 뒤 다음 계산으로 원본 좌표를 복원했다.

```text
원본 x = crop 시작 x + 예측 x × crop 너비 / 640
원본 y = crop 시작 y + 예측 y × crop 높이 / 640
```

좌표 환산 원리는 재사용할 수 있다. 다만 각 crop의 가로세로 비율을 강제로 바꾼 점과 서로 겹치는 crop에서 같은 여드름을 중복 탐지할 가능성은 개선해야 한다. 새 구현은 얼굴 전체에서 한 번 탐지하고 탐지 중심점이나 겹치는 면적으로 얼굴 부위를 판정하는 방식을 우선 검토한다.

### 사용자 흐름과 기록 항목

이전 웹 서비스에는 사진 업로드, 분석 결과, 가장 많이 탐지된 얼굴 부위와 과거 기록 화면이 있다. 이 흐름은 새 웹 서비스의 초기 사용자 흐름으로 활용할 수 있다.

이전 데이터베이스의 `uploaded_at`, `total_acne_count`, `max_acne_part`는 새 데이터 모델의 참고 항목으로 사용할 수 있다. 새 모델에는 사용자 식별자, 얼굴 부위별 개수, 모델 버전, threshold, 얼굴 부위 정의 버전과 촬영 품질 결과가 추가되어야 한다.

## 옮기지 않은 코드와 데이터

다음 자료는 새 프로젝트에 포함하지 않았다.

- `Base.metadata.drop_all(engine)`은 서버가 시작될 때마다 기존 분석 기록을 삭제한다.
- Flask `secret_key`가 소스에 고정되어 있다.
- 원본 파일명을 기준으로 업로드 파일을 저장하여 이름 충돌과 개인정보 관리 문제가 생길 수 있다.
- 모델, 얼굴 분석, 데이터베이스와 HTTP 처리가 하나의 실행 경로에 결합되어 있다.
- 화면을 다시 열 때 `/save_result`를 호출하므로 같은 결과가 중복 저장될 수 있다.
- 얼굴 미검출 결과가 정상 결과와 다른 구조인데 API에서 이를 처리하지 않아 서버 오류가 날 수 있다.
- 얼굴 부위와 장기의 건강 상태를 연결하고 음식을 추천하는 문구는 의학적 근거를 확인하지 못했다.
- 기존 SQLite 파일과 `uploads/`의 이미지는 사용자 정보일 수 있다.
- `.idea`, `__pycache__`, Windows 가상환경과 중복 HTML/CSS 파일은 새 프로젝트에 필요하지 않다.

## 찾지 못한 자료

다음 자료는 현재 `acne-CV`의 파일과 Git 이력에서 확인하지 못했다.

- 모델 학습 스크립트
- `data.yaml`과 학습 데이터 분할 기록
- epoch, batch size, optimizer와 augmentation 설정
- Precision, Recall, mAP 평가 결과
- Python 의존성 목록이나 재현 가능한 lockfile
- 모델 가중치와 Kaggle 데이터셋을 연결하는 명시적인 기록

따라서 기존 가중치는 출처가 완전히 재현되는 모델로 표시하지 않고 `legacy baseline candidate`로 관리한다. Kaggle test set에서 다시 평가한 뒤 baseline 채택 여부를 결정한다.

## 조사 이후 처리 결과

Kaggle 데이터셋을 내려받아 파일 대응, 라벨 범위와 분할 간 완전 중복을 검사했다. train과 test에 같은 이미지 한 쌍이 있어 test 쪽 파일을 비교 집합에서 제외했다. TensorFlow 2.16.2와 KerasCV 0.9.0으로 이전 가중치를 복원했고, 남은 test 이미지 47장의 기준 성능을 `docs/experiments/legacy-baseline.md`에 기록했다.

탐지 결과의 자료형, 탐지기 인터페이스, 데이터 검사와 평가 지표는 `backend/src/acne_care/`에 독립 모듈로 다시 작성했다. 얼굴 부위 landmark는 아직 검증 전 참고 자료로 유지한다. 다음 작업은 현재 관리되는 YOLO 구현으로 새 baseline을 학습한 뒤 같은 비교 집합에서 이전 모델과 비교하는 것이다.
