# YOLO11n baseline 실험

## 결론

새 YOLO11n baseline은 기존 모델보다 COCO mAP와 mAP@0.5가 높다. confidence 0.4에서는 Precision이 높아졌지만 Recall과 F1은 낮아졌다. 다음 실험은 재학습보다 confidence threshold 조정이 우선이다.

## 학습 조건

- Ultralytics 8.4.157과 YOLO11n COCO 사전학습 가중치를 사용했다.
- train 823장과 validation 56장을 사용했다.
- 입력 크기는 640×640, batch size는 8, seed는 42이다.
- Apple M3 MPS에서 35 epoch를 완료한 뒤 사용자 요청으로 중단했다.
- validation 최고 성능은 35 epoch의 mAP50-95 0.286이다.
- MPS 일부 연산은 완전한 결정성을 지원하지 않으므로 같은 seed에서도 미세한 차이가 날 수 있다.

## Clean test 비교

중복 이미지를 제외한 동일한 test 이미지 47장을 사용했다.

| 지표 | acne-CV 기존 모델 | YOLO11n baseline |
|---|---:|---:|
| Precision | 0.759 | 0.854 |
| Recall | 0.370 | 0.340 |
| F1 | 0.498 | 0.486 |
| COCO mAP | 0.196 | 0.297 |
| mAP@0.5 | 0.475 | 0.651 |
| mAP@0.75 | 0.121 | 0.220 |

threshold 기반 지표에는 confidence 0.4와 IoU 0.5를 적용했다. mAP 향상은 모델의 전체 confidence 순위와 위치 정확도가 좋아졌음을 뜻한다. 고정 threshold에서는 검출 수가 줄어 Recall이 낮으므로 threshold sweep으로 Precision과 Recall의 균형점을 찾아야 한다.

전체 결과는 `docs/experiments/yolo11n-baseline-clean.json`에 기록했다. 모델 가중치는 Git에서 제외하며 SHA-256은 `3870c1a2ccf3a0498e03cdb35b26d4e0671fde70a3c496264a0988d5cf75168b`이다.
