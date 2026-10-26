# 데이터 관리

초기 기준 데이터는 Osman Kagan Kurnaz가 공개한 [Acne Dataset in YOLOv8 Format](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format)이다.

로컬에서 확인한 데이터는 이미지와 YOLO 라벨이 각각 927개이다. 분할은 train 823개, validation 56개, test 48개이며 클래스는 `Acne` 한 개이다.

## 출처와 라이선스

Kaggle 데이터 카드에는 640×640 stretch resize와 사전 augmentation이 적용되었다고 적혀 있으며 라이선스는 Apache 2.0으로 표시되어 있다. 다운로드 파일의 `data.yaml`은 Roboflow의 `skin-detection-uvj1f` 버전 8을 출처로 적고 라이선스를 CC BY 4.0으로 표시한다.

두 라이선스 표시가 일치하지 않는다. 출처를 추가로 확인하기 전까지 데이터 파일을 재배포하지 않고, 보고서와 결과물에는 Kaggle 작성자와 `data.yaml`에 기록된 Roboflow 프로젝트를 함께 표시한다.

## 로컬 저장 위치

다운로드한 원본은 `data/raw/acne-dataset-in-yolov8-format/`에 둔다. `data/raw/`는 Git에서 제외되어 있다.

## 라벨 수정 데이터셋 생성

검토가 끝난 라벨은 원본에 덮어쓰지 않는다. 다음 명령은 편집 결과를 적용한 독립 데이터셋을
`data/processed/acne-dataset-v2/`에 만든다. 같은 명령을 다시 실행하면 기존 생성물을 교체하므로
이전 실행에서 남은 파일이 섞이지 않는다.

```bash
PYTHONPATH=backend/src .venv/training/bin/python ml/dataset/build_dataset_v2.py \
  --source-root data/raw/acne-dataset-in-yolov8-format/data-2 \
  --audit-json runs/dataset-audit/audit.json \
  --label-edits docs/experiments/label-edits-2026-10-08.json \
  --exclude-file data/splits/acne-yolov8-v1-exclusions.txt \
  --output-root data/processed/acne-dataset-v2
```

2026년 10월 8일 편집 결과를 적용하면 26장의 라벨이 바뀐다. 검토에서 제외한 Validation 이미지
2장과 기존 분할 중복 Test 이미지 1장을 제외한다. 생성 결과는 Train 823장, Validation 54장,
Test 47장이다. `data/processed/`는 Git에서 제외되므로 생성 코드와 편집 JSON을 기준으로 다시 만든다.

## 학습 전 검사

이미지와 라벨의 대응, 클래스 값과 bounding box 범위를 검사했다. 파일 내용이 완전히 같은 이미지 한 쌍이 train과 test에 걸쳐 발견되었다.

- `train/images/acne-299_jpeg.rf.193780200dffa83e6069b20d7eb65086.jpg`
- `test/images/acne-112_jpeg.rf.a03c796422719e964f46be1c118861bc.jpg`

두 이미지는 같지만 라벨 수는 각각 11개와 21개로 서로 다르다. 따라서 test 쪽 이미지를 [제외 목록](splits/acne-yolov8-v1-exclusions.txt)에 기록했고, 모델 비교에는 나머지 test 이미지 47개를 사용한다.

현재 검사는 파일이 완전히 같은 경우만 확인했다. 변형된 유사 이미지와 동일 인물의 분할 간 중복 여부는 아직 확인하지 못했다. 이 검사가 끝나기 전에는 47개 비교 집합의 성능을 외부 일반화 성능으로 해석하지 않는다.

검사를 마치기 전에는 제공된 test 분할의 성능을 최종 성능으로 보고하지 않는다.

## 외부 데이터 후보

추가 데이터는 `data/raw/candidates/` 아래에 원본별로 분리해서 둔다. 이 경로도 Git에서 제외한다.
기존 `dataset-v2`에 바로 복사하지 않고, 출처와 이용 조건, 라벨 정의, 구조 오류, 분할 중복,
기존 데이터와의 이미지 중복을 먼저 검사한다.

2026년 10월 22일에는 Kaggle의
[`Acne | Computer Vision`](https://www.kaggle.com/datasets/imtkaggleteam/acne-computer-vision)을
첫 후보로 검사했다. 아직 학습 데이터로 승인하거나 병합하지 않았다. 검사 결과와 보류 근거는
[`external-dataset-candidate-audit.md`](../docs/experiments/external-dataset-candidate-audit.md)에 기록했다.

다음 명령은 후보의 Roboflow CSV 주석을 검사하고 `dataset-v2`와 이미지 중복을 비교한다.
상세 결과에는 원본 파일 이름이 들어갈 수 있으므로 Git에서 제외된 `runs/` 아래에 저장한다.

```bash
PYTHONPATH=backend/src .venv/training/bin/python ml/dataset/audit_external_candidate.py \
  --candidate-root data/raw/candidates/acne-computer-vision \
  --reference-root data/processed/acne-dataset-v2 \
  --output runs/dataset-audit/acne-computer-vision-candidate.json \
  --kaggle-ref imtkaggleteam/acne-computer-vision \
  --declared-license "CC BY 4.0"
```

`acne`과 `pimple`을 단일 `Acne` 클래스로 변환한 50장 표본과 검토 화면은 다음과 같이 만든다.
고정 시드로 `pimple` 포함 이미지 25장과 `acne`만 있는 이미지 25장을 선택한다. `spot` 상자는
표본 라벨에 넣지 않는다.

```bash
PYTHONPATH=backend/src .venv/training/bin/python ml/dataset/prepare_external_candidate_review.py \
  --source-root data/raw/candidates/acne-computer-vision \
  --output-root runs/dataset-audit/external-candidate-review-dataset \
  --sample-size 50 \
  --seed 42

PYTHONPATH=backend/src .venv/training/bin/python ml/dataset/prepare_label_review.py \
  --dataset-root runs/dataset-audit/external-candidate-review-dataset \
  --output-directory runs/dataset-audit/external-candidate-review
```
