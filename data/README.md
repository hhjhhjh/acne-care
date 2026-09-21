# 데이터 관리

초기 기준 데이터는 Osman Kagan Kurnaz가 공개한 [Acne Dataset in YOLOv8 Format](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format)이다.

Kaggle 데이터 카드에는 이미지 927개, YOLOv8 라벨, train·validation·test 분할, 640×640 stretch resize와 사전 augmentation이 적용되었다고 적혀 있다. 라이선스 표시는 Apache 2.0이다.

## 로컬 저장 위치

다운로드한 원본은 `data/raw/acne-dataset-in-yolov8-format/`에 둔다. `data/raw/`는 Git에서 제외되어 있다.

## 학습 전 검사

- 이미지와 라벨 파일의 대응을 확인한다.
- 클래스 수와 클래스 이름을 확인한다.
- bounding box 좌표 범위를 확인한다.
- train, validation과 test 사이의 동일 이미지 및 유사 이미지를 검사한다.
- 사전 augmentation 이미지가 여러 분할에 섞였는지 확인한다.
- 사람별로 여러 이미지가 있다면 같은 사람이 여러 분할에 들어갔는지 확인한다.

검사를 마치기 전에는 제공된 test 분할의 성능을 최종 성능으로 보고하지 않는다.
