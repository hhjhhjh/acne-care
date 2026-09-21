# acne-CV 참고 자료

이 디렉터리는 이전 프로젝트에서 가져온 사실과 설정을 보존한다. 여기 있는 설정은 검증 전 자료이며 새 애플리케이션이 직접 읽는 제품 설정으로 사용하지 않는다.

## 파일

- `model-manifest.yaml`은 기존 KerasCV YOLOv8 모델을 다시 구성하는 데 필요한 정보를 기록한다.
- `face-region-landmarks.yaml`은 기존 얼굴 부위별 MediaPipe Face Mesh 인덱스를 보존한다.

원본 모델 코드는 `/Users/hwangjeonghyun/Desktop/github/acne-CV/app/model/model.py`와 `/Users/hwangjeonghyun/Desktop/github/acne-CV/PythonProject1/app_db/model/model.py`에서 확인했다. 코드 전체를 복사하지 않은 이유와 재사용 판정은 `docs/legacy-acne-cv-audit.md`에 적었다.
