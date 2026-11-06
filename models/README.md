# 모델 파일 관리

모델 가중치는 Git에 직접 커밋하지 않는다. 웹 서비스의 기본 모델은
[`detector-baseline.yaml`](detector-baseline.yaml)에서 관리한다. 파일 위치, SHA-256과 탐지 설정을
이 파일에서 확인한다. 학습 조건과 평가 결과는 [기준 모델 실험](../docs/experiments/yolo11n-dataset-v2-control.md)을 참고한다.

웹앱을 실행하려면 별도로 보관한 학습 결과 `best.pt`를 모델 설정의 `weights_path`에 복사한다.
macOS에서는 `shasum -a 256 <가중치 파일 경로>`로 확인값을 비교할 수 있다. 초기 사전학습
가중치 `yolo11n.pt`나 아래의 이전 프로젝트 가중치는 현재 서비스 모델을 대신하지 않는다.

이전 `acne-CV` 가중치의 출처, 해시와 모델 구조는
[`model-manifest.yaml`](../references/acne-cv/model-manifest.yaml)에서 별도로 관리한다.

## 기존 가중치 가져오기

두 저장소가 모두 `Desktop/github` 아래에 있으면 다음 명령을 실행한다.

```bash
./scripts/import_legacy_model.sh
```

명령은 원본 파일의 크기와 SHA-256을 확인한 뒤 다음 경로에 복사한다.

```text
models/legacy/acne-cv/yolo_acne_detection.weights.h5
```

같은 파일이 이미 있으면 복사를 건너뛴다. 다른 파일이 있으면 덮어쓰지 않고 오류를 낸다.
