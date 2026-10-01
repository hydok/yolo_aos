# 라이선스 안내

이 앱에 포함된 모델과 라이브러리의 라이선스, 그리고 앱을 배포할 때 지켜야 할 조건을 정리한 문서입니다.

> 이 문서는 개발 참고용 정리이며 법률 자문이 아닙니다. 상업적으로 배포하기 전에는 각 원문 라이선스를 직접 확인하세요. (확인 기준일: 2026-10-01)

## 요약

- 이 앱에는 **AGPL-3.0** 모델인 YOLO11n이 들어 있습니다. 그래서 앱을 배포하려면 **앱 전체 소스를 AGPL-3.0으로 공개**하거나, Ultralytics **Enterprise 라이선스**를 구매해야 합니다.
- 그 밖의 구성요소(MediaPipe, LiteRT, AndroidX 등)는 대부분 **Apache-2.0**이라 상업적 이용이 자유롭습니다. 라이선스 고지만 지키면 됩니다.
- 현재 이 저장소에는 프로젝트 자체의 `LICENSE` 파일이 없습니다. 공개 저장소로 유지하려면 AGPL-3.0 `LICENSE` 파일을 추가하는 것을 권장합니다.

## 구성요소별 라이선스

### AI 모델

| 모델 | 파일 | 라이선스 | 출처 |
| --- | --- | --- | --- |
| YOLO11n (Ultralytics) | `yolo11n.tflite` | **AGPL-3.0** | [Ultralytics License](https://www.ultralytics.com/license) |
| Selfie Segmentation | `selfie_segmenter.tflite` | Apache-2.0 | 모델 카드 |
| BlazeFace + Face Mesh V2 + Blendshape V2 | `face_landmarker.task` | Apache-2.0 | 모델 카드 3종 |
| Hand Tracking | `gesture_recognizer.task` (손 관절 부분) | Apache-2.0 | 모델 카드 |
| Hand Gesture Classifier | `gesture_recognizer.task` (제스처 분류 부분) | 모델 카드에 라이선스 표기 없음 ※ | 모델 카드 |
| BlazePose GHUM 3D | `pose_landmarker_lite.task` | Apache-2.0 | 모델 카드 |
| EfficientDet-Lite0 | `efficientdet_lite0.tflite` | 공식 문서에 라이선스 표기 없음 ※ | MediaPipe 문서 |
| EfficientNet-Lite0 | `efficientnet_lite0.tflite` | 공식 문서에 라이선스 표기 없음 ※ | MediaPipe 문서 |

※ 표시는 직접 확인한 문서에 라이선스가 적혀 있지 않은 항목입니다. MediaPipe가 Apache-2.0으로 함께 배포하는 모델이지만, 상업적으로 배포하기 전에는 MediaPipe 저장소나 구글에 확인하세요.

MediaPipe 모델 카드는 [MediaPipe 솔루션 가이드](https://ai.google.dev/edge/mediapipe/solutions/guide)의 각 기능 페이지에서 링크를 따라가면 볼 수 있습니다.

### 라이브러리

| 라이브러리 | 용도 | 라이선스 |
| --- | --- | --- |
| MediaPipe Tasks Vision | MediaPipe 추론 | Apache-2.0 |
| LiteRT (`litert`, `litert-gpu`) | YOLO 추론 | Apache-2.0 |
| AndroidX (CameraX, Compose, Activity, Core, Lifecycle) | 카메라, UI | Apache-2.0 |
| Kotlin 표준 라이브러리 | 언어 | Apache-2.0 |
| Guava, Flogger | MediaPipe 의존성 | Apache-2.0 |
| Protocol Buffers (`protobuf-javalite`) | MediaPipe 의존성 | BSD-3-Clause |

### 학습 데이터 (참고)

모델이 학습에 쓴 데이터셋의 라이선스는 모델 라이선스와 별개입니다. 앱이 데이터셋 자체를 포함하지는 않습니다.

| 데이터셋 | 사용 모델 | 비고 |
| --- | --- | --- |
| COCO | YOLO11n, EfficientDet-Lite0 | 주석은 CC BY 4.0 |
| ImageNet | EfficientNet-Lite0 | 데이터셋 이용 약관은 비상업 연구용 |

## AGPL-3.0 (YOLO11n) 조건

Ultralytics는 사전학습 모델뿐 아니라 **직접 학습한 모델과 변환한 모델**(`.tflite` 포함)도 AGPL-3.0 대상이라고 밝히고 있습니다.

앱을 배포(APK, 플레이스토어 등)할 때 지켜야 할 것:

1. **소스 공개**: YOLO만이 아니라 앱 전체 소스를 AGPL-3.0으로 공개
2. **라이선스 고지**: 앱을 받는 사람에게 AGPL-3.0 전문과 소스를 받을 방법을 안내
3. **같은 라이선스 유지**: 수정한 버전도 AGPL-3.0으로 배포, 추가 제한 금지
4. **네트워크 서비스**: YOLO를 서버에서 돌려 서비스만 제공해도 소스 공개 대상

Enterprise 라이선스가 필요한 경우:

- 소스를 공개하지 않는 상업 앱
- 사내 전용 도구, 비공개 SaaS
- 커스텀 학습 모델을 상업 제품에 쓸 때

## Apache-2.0 (MediaPipe 등) 조건

상업적 이용, 수정, 재배포가 모두 가능합니다. 지켜야 할 것:

1. 배포물에 Apache-2.0 라이선스 전문 포함
2. 원본에 `NOTICE` 파일이 있으면 그 내용 유지
3. 원본을 수정했다면 수정했다는 사실 표시

앱에서는 보통 설정 화면에 "오픈소스 라이선스" 메뉴를 두어 고지합니다. Google의 `oss-licenses-plugin`을 쓰면 의존성 라이선스 목록을 자동으로 만들어 줍니다.

## 사용 시나리오별 정리

| 시나리오 | 가능 여부 | 필요한 조치 |
| --- | --- | --- |
| 개인 학습, 포트폴리오 (GitHub 공개) | 가능 | AGPL-3.0 `LICENSE` 파일 추가 권장 |
| 오픈소스 앱으로 플레이스토어 배포 | 가능 | 앱 전체 소스 AGPL-3.0 공개, 앱 내 라이선스 고지 |
| 비공개 소스 상업 앱 (YOLO 포함) | 조건부 | Ultralytics Enterprise 라이선스 구매 |
| 비공개 소스 상업 앱 (YOLO 제거, MediaPipe만) | 가능 | 앱 내 Apache-2.0 고지, ※ 표시 모델 라이선스 확인 |

YOLO 없이 비공개 상업 앱으로 내려면 `yolo11n.tflite`, `YoloDetector.kt`, 그리고 첫 화면의 YOLO 선택지를 빼면 됩니다. 이 경우 사물 검출은 MediaPipe의 EfficientDet-Lite0로 대신할 수 있습니다.

## 참고 링크

- [Ultralytics License](https://www.ultralytics.com/license)
- [GNU AGPL-3.0 전문](https://www.gnu.org/licenses/agpl-3.0.html)
- [Apache License 2.0 전문](https://www.apache.org/licenses/LICENSE-2.0)
- [MediaPipe 솔루션 가이드](https://ai.google.dev/edge/mediapipe/solutions/guide)
