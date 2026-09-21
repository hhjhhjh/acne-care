# 데이터 관리

초기 기준 데이터는 Osman Kagan Kurnaz가 공개한 [Acne Dataset in YOLOv8 Format](https://www.kaggle.com/datasets/osmankagankurnaz/acne-dataset-in-yolov8-format)이다.

로컬에서 확인한 데이터는 이미지와 YOLO 라벨이 각각 927개이다. 분할은 train 823개, validation 56개, test 48개이며 클래스는 `Acne` 한 개이다.

## 출처와 라이선스

Kaggle 데이터 카드에는 640×640 stretch resize와 사전 augmentation이 적용되었다고 적혀 있으며 라이선스는 Apache 2.0으로 표시되어 있다. 다운로드 파일의 `data.yaml`은 Roboflow의 `skin-detection-uvj1f` 버전 8을 출처로 적고 라이선스를 CC BY 4.0으로 표시한다.

두 라이선스 표시가 일치하지 않는다. 출처를 추가로 확인하기 전까지 데이터 파일을 재배포하지 않고, 보고서와 결과물에는 Kaggle 작성자와 `data.yaml`에 기록된 Roboflow 프로젝트를 함께 표시한다.

## 로컬 저장 위치

다운로드한 원본은 `data/raw/acne-dataset-in-yolov8-format/`에 둔다. `data/raw/`는 Git에서 제외되어 있다.

## 학습 전 검사

이미지와 라벨의 대응, 클래스 값과 bounding box 범위를 검사했다. 파일 내용이 완전히 같은 이미지 한 쌍이 train과 test에 걸쳐 발견되었다.

- `train/images/acne-299_jpeg.rf.193780200dffa83e6069b20d7eb65086.jpg`
- `test/images/acne-112_jpeg.rf.a03c796422719e964f46be1c118861bc.jpg`

두 이미지는 같지만 라벨 수는 각각 11개와 21개로 서로 다르다. 따라서 test 쪽 이미지를 [제외 목록](splits/acne-yolov8-v1-exclusions.txt)에 기록했고, 모델 비교에는 나머지 test 이미지 47개를 사용한다.

현재 검사는 파일이 완전히 같은 경우만 확인했다. 변형된 유사 이미지와 동일 인물의 분할 간 중복 여부는 아직 확인하지 못했다. 이 검사가 끝나기 전에는 47개 비교 집합의 성능을 외부 일반화 성능으로 해석하지 않는다.

검사를 마치기 전에는 제공된 test 분할의 성능을 최종 성능으로 보고하지 않는다.
