# YOLO11n dataset-v2 기준 모델

## 결론

수정한 `dataset-v2`로 YOLO11n을 다시 학습한 결과, 이전 가중치보다 Validation과 Test의 모든
핵심 지표가 좋아졌다. 새 모델 `yolo11n-dataset-v2-control-seed42`를 데이터 증강 실험의 기준
모델로 사용한다. 서비스용 confidence threshold는 Validation에서 선택한 `0.30`으로 고정한다.

## 학습 조건

- 데이터: `acne-dataset-v2`의 Train 823장, Validation 54장, Test 47장
- 초기 가중치: `yolo11n.pt`
- 실행 장치: NVIDIA RTX A5000
- 학습 횟수: 50 epoch
- 입력 크기: 640×640
- 배치 크기: 8
- seed: 42
- patience: 15
- Ultralytics: 8.4.157
- PyTorch: 2.10.0+cu130

학습에는 Ultralytics 기본 증강 설정을 사용했다. 주요 값은 좌우 반전 `0.5`, 이동 `0.1`, 크기
변경 `0.5`, mosaic `1.0`이다. 회전, 상하 반전, mixup은 적용하지 않았다. 전체 설정은
`yolo11n-dataset-v2-control.json`에 기록했다.

가중치 파일은 Git에 넣지 않는다. 파일 위치와 무결성 확인값은 다음과 같다.

- 로컬 위치: `runs/baselines/yolo11n-dataset-v2-control-seed42/weights/best.pt`
- SHA-256: `0b486ebc4cd983ae23380529cf72790914838a4bdf298cf54732b9762d12beef`

## Validation 결과

Validation 54장에서 F1을 최대로 만드는 threshold를 선택했다. 동률이면 Count MAE가 작은 값과
Recall이 큰 값을 차례로 사용했다.

| 모델 | Threshold | Precision | Recall | F1 | Count MAE | mAP50-95 | mAP50 | mAP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 이전 가중치 | 0.20 | 0.648 | 0.578 | 0.611 | 3.370 | 0.284 | 0.642 | 0.221 |
| dataset-v2 재학습 | 0.30 | 0.667 | 0.617 | 0.641 | 3.296 | 0.301 | 0.678 | 0.228 |

재학습 모델은 True Positive가 27개 늘었다. False Positive는 3개, False Negative는 27개
줄었다. F1은 0.030 높아졌고 mAP50-95는 0.017 높아졌다.

## Test 결과

Test에는 Validation에서 선택한 threshold `0.30`을 한 번 적용했다. Test에서 threshold를 다시
선택하지 않았다.

| 모델 | Threshold | Precision | Recall | F1 | Count MAE | mAP50-95 | mAP50 | mAP75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 이전 가중치 | 0.20 | 0.677 | 0.586 | 0.628 | 3.745 | 0.293 | 0.649 | 0.212 |
| dataset-v2 재학습 | 0.30 | 0.703 | 0.620 | 0.659 | 3.574 | 0.314 | 0.689 | 0.225 |

재학습 모델은 True Positive가 21개 늘었다. False Positive는 11개, False Negative는 21개
줄었다. F1은 0.031 높아졌고 mAP50-95는 0.021 높아졌다. Validation과 Test에서 같은 방향의
개선이 나타났으므로 이 모델을 새 기준 모델로 확정한다.

## 해석과 다음 실험

후반 10 epoch의 Validation mAP50-95는 0.279부터 0.301 사이에서 오르내렸다. 최고값은 마지막
50 epoch에서 나왔지만 직전 최고값보다 0.0004 높았다. 학습 횟수만 즉시 늘리기보다 새 모델의
False Positive와 False Negative를 먼저 분류한다. 그 결과를 근거로 밝기·대비, 소폭 회전,
약한 흐림·노이즈를 한 조건씩 비교한다.

서버에서 측정된 추론 시간은 기록하지 않는다. 당시 평가 코드는 CUDA 작업을 완료하기 전에 시간을
읽을 수 있었다. 정확도 지표에는 영향을 주지 않으며, CUDA 동기화를 추가한 뒤 속도를 다시 측정한다.
