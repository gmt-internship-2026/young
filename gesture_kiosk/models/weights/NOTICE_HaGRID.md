# 고지문 — HaGRID 제스처 모델 (저작자 표시 의무 이행)

> ⚠ **2026-07-10 라이선스 C안**: 기본 제스처 엔진이 MediaPipe(Apache-2.0)로 교체되어
> **기본 구성에서는 HaGRID 모델을 사용하지 않는다** — 이 경우 본 고지문은 불필요하다.
> 아래 내용은 비교 시험용 구 엔진(`gesture_engine: onnx`)을 켜서 HaGRID 모델을
> 실제로 사용·배포하는 경우에만 적용된다. 또한 이 모델은 AGPL-3.0(YOLOv10/ultralytics)
> 계열 학습·변환 가중치라 **비공개 상업 납품에는 사용 금지** (README 라이선스 절).

본 소프트웨어의 손동작 인식 기능이 아래 공개 모델을 사용하는 경우,
이 고지문은 HaGRID 라이선스 제3조(저작자 표시)에 따라 제품 문서·정보 화면에
포함되어야 한다.

| 항목 | 내용 |
|---|---|
| 사용 자료 | HaGRIDv2 사전학습 제스처 검출 모델 (YOLOv10n_gestures) |
| 제작자 | HaGRID 프로젝트 (A. Kapitanov, K. Kvanchiani, A. Nagaev, R. Kraynov, A. Makhliarchuk 외) |
| 출처 | https://github.com/hukenovs/hagrid |
| 라이선스 | 저장소 동봉 자체 공개 라이선스 (Creative Commons BY-SA 4.0을 재작업한 것, CC 라이선스 아님) — https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf |
| 변경 사항 | 원본 PyTorch 가중치(.pt)를 기능 변경 없이 ONNX 형식으로 변환함 (2026-07-11, scripts/export_onnx.py) |

본 소프트웨어의 나머지 구성요소: rtmlib/RTMPose(Apache-2.0), ONNX Runtime(MIT),
EasyOCR(Apache-2.0) — 각 라이선스 사본은 해당 패키지 배포물에 포함되어 있다.
