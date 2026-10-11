# YOLO11n dataset-v2 재평가

## 결론

수정한 Validation 라벨에서도 YOLO11n의 최적 confidence threshold는 `0.20`이다. Test F1은
0.628로 거의 같았고 Count MAE는 4.106에서 3.745로 줄었다. 모델 가중치는 바꾸지 않았으므로
이 차이는 모델 개선 효과가 아니라 평가 라벨과 제외 대상이 바뀐 결과다.

## 평가 조건

- 가중치: `yolo11n-acne-seed42/weights/best.pt`
- 가중치 SHA-256: `3870c1a2ccf3a0498e03cdb35b26d4e0671fde70a3c496264a0988d5cf75168b`
- 입력 크기: 640×640
- 실행 장치: CPU(Apple M3)
- IoU threshold: 0.50
- Validation: dataset-v2 54장, 라벨 677개
- Test: dataset-v2 47장, 라벨 611개
- threshold 선택 규칙: Validation F1 최대, 동률이면 Count MAE 최소와 Recall 최대 순서

## Validation threshold 선택 결과

| 데이터 | Threshold | Precision | Recall | F1 | Count MAE |
|---|---:|---:|---:|---:|---:|
| 수정 전 56장 | 0.20 | 0.602 | 0.596 | 0.599 | 3.661 |
| dataset-v2 54장 | 0.20 | 0.648 | 0.578 | 0.611 | 3.370 |

수정 후에도 0.20에서 F1이 가장 높았다. Precision은 0.046 높아졌고 Recall은 0.018 낮아졌다.
F1은 0.012 높아졌으며 Count MAE는 0.290 줄었다.

## Test 결과

| 데이터 | Precision | Recall | F1 | Count MAE | mAP50-95 | mAP50 | mAP75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 수정 전 47장 | 0.662 | 0.597 | 0.628 | 4.106 | 0.297 | 0.651 | 0.220 |
| dataset-v2 47장 | 0.677 | 0.586 | 0.628 | 3.745 | 0.293 | 0.649 | 0.212 |

Test에서는 Precision이 0.015 높아지고 Recall이 0.011 낮아졌다. F1은 사실상 유지됐고 Count
MAE는 0.362 줄었다. mAP50-95는 0.004 낮아졌다. 라벨 수와 위치가 바뀌었으므로 이 수치를
수정 전 결과와 직접적인 모델 성능 향상으로 해석하지 않는다.

## 결과 파일

- `threshold-sweep-yolo11n-v2-val.json`: Validation threshold 전체 결과
- `threshold-selected-yolo11n-v2-test.json`: 선택한 threshold의 Test 결과

다음 단계에서는 dataset-v2 기준 FP/FN을 다시 분류한다. 특히 새로 추가한 라벨에서 계속 발생하는
FN과 기존 라벨 누락이 해소되면서 사라진 FP를 구분해 Train 데이터 점검과 증강 조건을 정한다.
