# YOLO11n 데이터 증강 대조 실험

## 결론

추가한 대비, 회전, 안전 크롭 증강은 Validation의 종합 성능을 개선하지 못했다. 세 모델 모두
기준 모델보다 F1이 낮고 Count MAE가 높았다. 따라서 추가 증강 모델은 채택하지 않고
`yolo11n-dataset-v2-control-seed42`를 기준 모델로 유지한다.

세 모델은 Test에서 평가하지 않았다. 모델 선택과 증강 조건 비교에는 Validation 54장만 사용했다.

## 대조 조건

모든 모델은 YOLO11n, dataset-v2, 입력 크기 640, seed 42, 50 epoch, batch size 8,
patience 15를 사용했다. 기준 모델의 증강 설정은 그대로 두고 실험마다 조건 하나만 추가했다.

| 조건 | 추가 설정 | 적용 확률 |
|---|---|---:|
| Control | 추가 설정 없음 | - |
| Contrast | 밝기는 유지하고 대비를 최대 ±15% 변경 | 0.5 |
| Rotation | 최대 ±10° 회전 | 기본 학습 변환 |
| Safe crop | 모든 상자를 보존하는 640×640 크롭 | 0.3 |

기준 모델에도 Ultralytics 기본 증강이 적용되어 있다. 주요 설정은 밝기 `0.4`, 이동 `0.1`, 크기
변경 `0.5`, 좌우 반전 `0.5`, mosaic `1.0`이다. 추가 실험은 이 설정 위에 각 변환을 적용했다.

학습은 NVIDIA RTX A5000에서 수행했다. Python은 3.12.3, Ultralytics는 8.4.157,
PyTorch는 2.10.0+cu130을 사용했다. 세 증강 모델의 최고 성능 epoch는 Contrast 50,
Rotation 49, Safe crop 49이다.

## Validation 결과

54장에 포함된 정답 객체 677개로 비교했다. Confidence threshold는 각 모델에서 F1이 가장 높은
값을 선택했다. F1이 같으면 Count MAE가 낮은 값과 Recall이 높은 값을 차례로 사용했다.

| 조건 | Threshold | TP | FP | FN | Precision | Recall | F1 | Count MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Control | 0.30 | 418 | 209 | 259 | 0.667 | 0.617 | **0.641** | **3.296** |
| Contrast | 0.25 | 438 | 293 | 239 | 0.599 | **0.647** | 0.622 | 4.074 |
| Rotation | 0.35 | 372 | 171 | 305 | **0.685** | 0.549 | 0.610 | 3.667 |
| Safe crop | 0.30 | 412 | 238 | 265 | 0.634 | 0.609 | 0.621 | 3.574 |

| 조건 | mAP50-95 | mAP50 | mAP75 |
|---|---:|---:|---:|
| Control | 0.301 | **0.678** | 0.228 |
| Contrast | **0.303** | 0.671 | **0.252** |
| Rotation | 0.293 | 0.661 | 0.210 |
| Safe crop | 0.296 | 0.662 | 0.219 |

Contrast는 Control보다 True Positive가 20개 많고 Recall이 0.030 높았다. 그러나 False
Positive가 84개 늘어 Precision은 0.068, F1은 0.019 낮아졌다. mAP50-95는 0.002 높았지만
Validation 규모와 다른 지표의 하락을 고려하면 성능 개선으로 판단하기 어렵다.

Rotation은 False Positive가 가장 적었지만 False Negative가 305개로 가장 많았다. Safe crop은
선택 threshold가 Control과 같지만 F1과 세 mAP가 모두 낮았다.

## 판단

현재 기준 모델에는 밝기, 이동, 크기 변경, 좌우 반전, mosaic가 이미 포함되어 있다. 추가 변환이
학습 이미지의 다양성을 늘렸지만, 현재 데이터에서는 탐지 정확도 향상으로 이어지지 않았다.

세 증강 모델은 Validation에서 기준 모델을 넘지 못했으므로 Test 평가에서 제외한다. 기준 모델과
서비스용 confidence threshold `0.30`을 유지한다. 다음 모델 개선 실험에서는 외부 데이터의 라벨
품질과 데이터 분할을 먼저 정비한다.

기계가 읽을 수 있는 요약은 `yolo11n-augmentation-experiment.json`에 보관한다. 전체 학습 결과와
threshold sweep은 Git에서 제외한 `runs/augmentation/`과 `runs/evaluation/`에 보관한다.
