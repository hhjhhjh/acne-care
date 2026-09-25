# 탐지 모델 threshold 비교

## 결론

acne-care의 탐지 baseline으로 YOLO11n을 선택한다. confidence threshold는 validation에서 선택한 0.20을 사용한다. 기존 acne-CV 모델은 성능 비교용 legacy 모델로 유지한다.

## 선택 방법

두 모델은 validation 이미지 56장에서 confidence threshold 0.05부터 0.90까지 0.05 간격으로 비교했다. F1이 가장 높은 값을 선택하고, 동률이면 Count MAE가 낮은 값과 Recall이 높은 값 순서로 선택했다.

| 모델 | 선택 threshold | Validation Precision | Validation Recall | Validation F1 | Validation Count MAE |
|---|---:|---:|---:|---:|---:|
| acne-CV legacy | 0.30 | 0.636 | 0.533 | 0.580 | 3.464 |
| YOLO11n | 0.20 | 0.602 | 0.596 | 0.599 | 3.661 |

## Clean test 최종 비교

선택한 threshold를 고정한 뒤 중복 이미지를 제외한 test 이미지 47장에서 평가했다.

| 지표 | acne-CV legacy | YOLO11n |
|---|---:|---:|
| Confidence threshold | 0.30 | 0.20 |
| Precision | **0.683** | 0.662 |
| Recall | 0.486 | **0.597** |
| F1 | 0.568 | **0.628** |
| Count MAE | 4.660 | **4.106** |
| COCO mAP | 0.232 | **0.297** |
| mAP@0.5 | 0.560 | **0.651** |
| mAP@0.75 | 0.142 | **0.220** |

YOLO11n의 Precision은 0.022 낮지만 Recall은 0.111, F1은 0.060 높다. 이미지별 개수 오차와 mAP도 개선됐다. 피부 상태 모니터링에서는 병변 누락과 개수 오차가 중요하므로 YOLO11n을 선택한다.

legacy mAP는 공정한 비교를 위해 prediction decoder의 최소 confidence를 0.01로 낮춘 뒤 다시 계산했다. 서비스에서 사용하는 threshold는 0.30이며 mAP 계산용 최소 confidence와는 역할이 다르다.

선택 결과의 단일 출처는 `models/detector-baseline.yaml`이다. 세부 결과는 다음 파일에 있다.

- `docs/experiments/threshold-sweep-legacy-val.json`
- `docs/experiments/threshold-sweep-yolo11n-val.json`
- `docs/experiments/threshold-selected-legacy-test.json`
- `docs/experiments/threshold-selected-yolo11n-test.json`

현재 데이터 규모가 작고 변형된 유사 이미지나 동일 인물의 분할 중복 여부는 확인하지 못했다. 따라서 이 결정은 현재 데이터에서 사용할 프로젝트 baseline 선택이며 외부 촬영 환경의 최종 성능을 뜻하지 않는다.
