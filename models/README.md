# 모델 파일 관리

모델 가중치는 Git에 직접 커밋하지 않는다. 파일의 출처, 해시와 모델 구조는 `references/acne-cv/model-manifest.yaml`에서 관리한다.

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
