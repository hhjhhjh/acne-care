# 이전 모델 기준 성능

## 결론

`acne-CV` 모델은 새 모델의 비교 기준으로 사용할 수 있다. confidence threshold 0.4에서 Precision은 0.759이지만 Recall은 0.370이다. 탐지한 대상은 비교적 정확하지만 실제 여드름을 많이 놓치므로, 새 baseline의 우선 개선 목표는 Recall과 mAP이다.

## 평가 대상

평가 모델은 `acne-CV`에서 가져온 KerasCV YOLOv8 XS 가중치이다. 파일의 SHA-256은 `0aa53871402aa540ebebd6368151eb2d21cf4fa9fa1a41b742ac3facc18f0f35`이다.

데이터는 Osman Kagan Kurnaz의 `Acne Dataset in YOLOv8 Format`을 사용했다. train과 test에 걸친 완전 중복 이미지 한 장을 제외하여 test 이미지 47장을 평가했다. 제외 근거와 경로는 `data/splits/acne-yolov8-v1-exclusions.txt`가 관리한다.

## 실행 환경

- Python 3.11.14를 사용했다.
- TensorFlow 2.16.2를 사용했다.
- Keras 3.15.1을 사용했다.
- KerasCV 0.9.0을 사용했다.
- 입력 이미지는 640×640으로 변환했다.
- threshold 기반 지표에는 confidence 0.4와 IoU 0.5를 적용했다.

이 호환 환경은 이전 가중치를 평가하기 위한 전용 환경이다. KerasCV 저장소가 보관 상태이므로 새 모델 학습의 기본 프레임워크로 채택하지 않는다.

## 측정 결과

| 지표 | 값 |
|---|---:|
| True Positive | 217 |
| False Positive | 69 |
| False Negative | 369 |
| Precision | 0.759 |
| Recall | 0.370 |
| F1 | 0.498 |
| COCO mAP | 0.196 |
| mAP@0.5 | 0.475 |
| mAP@0.75 | 0.121 |
| 이미지당 추론 시간 | 0.182초 |

추론 시간은 Apple Silicon CPU 환경의 모델 실행 시간만 합산한 값이다. 모델 최초 구성, 가중치 로드와 평가 지표 집계 시간은 포함하지 않았다.

## 해석과 다음 비교 기준

False Negative가 369개로 True Positive보다 많다. 따라서 새 baseline은 같은 47장과 같은 제외 목록을 사용하고 Precision, Recall, F1, COCO mAP를 모두 비교한다. threshold 하나의 결과만 보고 모델을 선택하지 않고 confidence별 Precision-Recall 변화도 확인한다.

참고로 중복 이미지를 포함한 원래 test 48장의 COCO mAP는 0.198, mAP@0.5는 0.477이었다. 중복을 제외한 값과 차이는 작지만, 학습 이미지가 test에 포함된 상태는 평가 계약에 맞지 않으므로 47장 결과만 정식 비교값으로 사용한다.

이 데이터에는 사전 augmentation 이미지가 포함되어 있으며 완전 중복 이외의 유사 이미지와 동일 인물 중복은 아직 검사하지 못했다. 현재 수치는 모델 간 비교에는 사용할 수 있지만 외부 촬영 환경에서의 성능을 뜻하지 않는다.

기계가 읽을 수 있는 전체 결과는 `docs/experiments/legacy-baseline-clean.json`에 보존한다. 평가는 `ml/evaluation/evaluate_legacy.py`로 다시 실행할 수 있다.
