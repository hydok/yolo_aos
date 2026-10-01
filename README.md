# yolo_aos
YOLO 11n 모델을 활용한 OnDevice AI 사물 인식 (Android)

카메라 프리뷰 위에 실시간으로 사물을 검출해 박스와 라벨(`이름 점수%`)을 그리는 앱입니다. 서버 없이 기기 안에서 LiteRT(구 TensorFlow Lite)로 추론합니다.

<p align="center">
  <img src="screenshot.gif" width="320" alt="실기기 실행 화면: 노트북, 컵, 병, 키보드, 마우스 인식" />
</p>

## 주요 기능

- CameraX 후면 카메라 프리뷰 (전체 화면, 권한 요청 포함)
- YOLO11n(COCO 80종) 온디바이스 실시간 사물 검출
- 프리뷰 위 박스 + 라벨 오버레이, 프레임 간 위치 보간으로 부드러운 박스 이동
- 라벨·입력 크기·클래스 수를 모델 파일에서 자동으로 읽음 → 모델 파일만 바꾸면 커스텀 모델 사용 가능
- GPU 델리게이트 우선 시도, 실패 시 CPU(4스레드)로 자동 전환

## 기술 스택

| 구분 | 내용 |
| --- | --- |
| 언어 / UI | Kotlin 2.2, Jetpack Compose (Material3) |
| 카메라 | CameraX 1.6.2 (`LifecycleCameraController`, `PreviewView`) |
| 추론 | LiteRT 1.4.2 (`litert`, `litert-gpu`) |
| 모델 | Ultralytics YOLO11n → `.tflite` (float32, 640×640) |
| 빌드 | AGP 9.4.1, minSdk 26, targetSdk 37 |

## 프로젝트 구조

```
app/src/main/
├── assets/
│   └── yolo11n.tflite          # 변환된 모델 (메타데이터 포함)
├── java/hydok/yolo/
│   ├── MainActivity.kt         # 권한, 카메라 프리뷰, 검출 결과 오버레이
│   ├── YoloDetector.kt         # 모델 로드, 전처리, 추론, NMS
│   └── ui/theme/               # Compose 테마
└── AndroidManifest.xml         # CAMERA 권한
```

## 동작 방식

```
카메라 프레임 (ImageAnalysis, RGBA)
  → 프리뷰에 보이는 영역만 크롭 → 회전 보정 → 640×640 레터박스 (남는 곳은 회색)
  → NCHW float32 (0~1) 변환
  → LiteRT 추론
  → 신뢰도 0.4 이상 후보 추림 → 클래스별 NMS (IoU 0.5)
  → 레터박스 좌표를 프리뷰 기준(0~1)으로 환산
  → Compose Canvas에 박스·라벨 그리기 (매 화면 프레임마다 30%씩 새 위치로 보간)
```

- 분석은 별도 단일 스레드에서 돌고, 처리 중 들어온 프레임은 버립니다(최신 프레임만 처리).
- 모델 입력은 비율을 유지하는 레터박스 방식입니다. 늘려서 맞추면 사물이 찌그러져 정확도가 떨어집니다.

## 모델 규격

| 항목 | 값 |
| --- | --- |
| 입력 | `[1, 3, 640, 640]` float32, RGB, 0~1, **NCHW** |
| 출력 | `[1, 84, 8400]` float32 — 앞 4개: `cx, cy, w, h`(0~1 정규화), 뒤 80개: 클래스 점수 |
| NMS | 미포함 (앱에서 처리) |
| 메타데이터 | 모델 파일 끝에 zip으로 `metadata.json` 포함 (`names`, `imgsz` 등) |

일반적인 TFLite 예제(NHWC)와 채널 순서가 다르니 주의하세요.

## 빌드 및 실행

1. Android Studio로 프로젝트 열기
2. 폰을 USB(또는 무선 디버깅)로 연결
3. Run(▶) 실행 → 카메라 권한 허용

명령줄로 설치하려면:

```bash
./gradlew :app:installDebug
```

Logcat에서 `YoloDetector`로 필터하면 GPU/CPU 중 어느 쪽으로 추론하는지 보입니다 (`Running on GPU` / `Running on CPU`).

## 모델 다시 만들기

Python 3.12 환경에서 진행했습니다. (3.14는 변환 도구 호환 문제가 있음)

```bash
python3.12 -m venv venv
```

```bash
./venv/bin/pip install ultralytics
```

```bash
./venv/bin/yolo export model=yolo11n.pt format=tflite imgsz=640
```

생성된 `yolo11n.tflite`를 `app/src/main/assets/`에 넣으면 됩니다. (`format=tflite`는 현재 `litert` 포맷으로 자동 전환되어 내보내집니다.)

### 커스텀 모델 사용

직접 학습한 YOLO 검출 모델도 같은 방식으로 내보내 `app/src/main/assets/yolo11n.tflite`를 교체하면 됩니다. 라벨, 입력 크기, 클래스 수는 모델에서 읽으므로 코드 수정이 필요 없습니다. 파일 이름을 바꾸려면 `YoloDetector.kt`의 `MODEL` 상수를 수정하세요.

## 조정 가능한 값

| 값 | 위치 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `CONFIDENCE_THRESHOLD` | `YoloDetector.kt` | 0.4 | 낮추면 더 많이 잡고 오검출도 늘어남 |
| `IOU_THRESHOLD` | `YoloDetector.kt` | 0.5 | 겹친 박스를 하나로 합치는 기준 |
| `SMOOTHING` | `MainActivity.kt` | 0.3 | 높이면 박스가 빨리 따라붙고, 낮추면 더 부드러움 |

## 참고 사항

- **LiteRT 버전**: 최신 2.x(2.1.6, 2.2.0)는 `litert`와 `litert-api`의 네임스페이스 충돌로 이 AGP 버전에서 빌드가 실패해 1.4.2를 사용합니다.
- **GPU**: 에뮬레이터에서는 GPU 초기화가 실패해 CPU로 동작합니다. 실기기는 기기별로 다를 수 있으며, 실패하면 자동으로 CPU를 씁니다.
- **APK 크기**: 모델(10MB)과 LiteRT·GPU 네이티브 라이브러리 포함으로 디버그 APK가 약 57MB입니다.
- **라이선스**: YOLO11 모델은 Ultralytics의 AGPL-3.0 라이선스를 따릅니다. 배포 시 확인이 필요합니다.
