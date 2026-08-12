# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 제스처를 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 + Python 3.11.5 — CPU 단독** (정부 민원발급기):
  2026-07-29 포즈 스택 제거로 GPU·CUDA 불필요 (구 GPU 자동 감지 통합판은 07-24~07-29)
- 동작 체계(2026-07-23 — 「제스처 정의 보고서」 손 모양 기준, 회사 확정):
  **손 모양이 계층을, 이동 방향이 기능을 정한다** — 한 손가락=탐색, 주먹=명령
- 모델: **MediaPipe HandLandmarker(Apache-2.0) 단일** — 내장 TFLite(XNNPACK) CPU 추론.
  손 21점(화면+월드 3D)으로 모양 판별·궤적·사용자 선별·거리 자까지 전부 판정
  (2026-07-29 포즈(rtmlib/ONNX Runtime) 제거 — 손 모양 판별은 자체 기하 규칙, 별도 CNN 없음)
- 연동: **파이프(stdio)** — 이벤트를 stdout에 한 줄씩 print, 델파이7 UI가 파이프로
  수신 (2026-07-23 회사 확정 — 네트워크(UDP·웹소켓) 전면 철회)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — CPU 전용, 내부망은 설치가이드.md B절
py main.py                          :: 실행 — 카메라·판정 계기판 창을 켠 채 시작(기본, 2026-08-11)
py main.py --no-cam                 :: 창 없이 이벤트만(stdout, 델파이 연동 동일)
:: 실행 중에는 콘솔에 cam on / cam off 로 창을 켜고 끌 수 있다
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 델파이7 UI 연동(수신부 완성 코드 포함): **[docs/델파이7_연동가이드.md](docs/델파이7_연동가이드.md)**

## 인식 동작 (이벤트 10종 — 2026-07-29 개편 + 07-31 손바닥·탭 추가)

| 이벤트 | 손 모양 | 동작 | 키오스크 명령 |
|---|---|---|---|
| left / right / select | **한 손가락** (종류 무관) | 좌 / 우 / 위 쓸기 | 포커스 1칸 이동 |
| back | **주먹** | 왼쪽 쓸기 | 이전 화면 |
| home | **주먹** | 위 쓸기 | 처음 화면 |
| confirm | **주먹** | 오른쪽 쓸기 | 현재 항목 실행 |
| temp_left / temp_right / temp_top | **손바닥**(전부 폄) | 좌 / 우 / 위 쓸기 | 회사 정의 예정 (2026-07-31 추가) |
| click | **한 손가락** | 제자리 검지 까딱 2회 | 클릭 (회사 정의 예정) |

- **핵심 규칙**: 손 모양이 계층을(탐색/명령), 이동 방향이 기능을 정한다 —
  반복 횟수·화면 좌표는 쓰지 않는다. 탐색(한 손가락)은 아무리 반복해도 화면이
  안 바뀌고, 화면을 바꾸는 동작은 주먹을 쥐어야만 실행된다 (오발 안전 구조)
- 손 모양은 **래치 상태기**(07-28)로 확정한다 — 저속에서 연속 판별로 고정,
  빠른 이동 중(모션 블러) 판별 동결, 반대 모양 연속 확인 시에만 전환.
  방향은 **첫 선 고정**(07-28) — 원점을 떠나는 첫 이동 벡터가 방향을 정한다
- 아래 방향은 정의 없음(두 모양 공통 — 07-29 bottom 제거) — 무시. 위 방향
  (select·home)은 팔 들어올리기(예비 동작) 오발을 휴식 존 게이트가 막는다
- 사용자 손 선별(hand_select): **단일 손 추적**(07-31 라벨 제거) — 움직여서
  지시한 손 하나를 공간 연속성으로 고정하고, 교체는 그 손을 내린 뒤에만.
  앞단에 **머리 앵커 게이트**(포즈 기반 — 마스크·모자 무관)가 가장 가까운
  사람의 팔 도달 반경 밖 손을 차단한다. **★한 명 사용 가정** (docs/TODO.md №1-2)
- 스펙 변천: 주먹→펴기(07-15 제거) → 손등/팔등(07-15 2차 제거) → 고개 꾸벅(07-16
  제거) → 쓸기 일원화(07-16) → **현행: 손 모양 기준(07-23 — 보고서 개정 반영)**

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 손 랜드마크(MediaPipe HandLandmarker — 주 추론 모델)
  → 사용자 손 선별(hand_select: 머리 앵커 게이트 + 단일 손 추적, 손 실측 거리 자)
  → 손 모양(hand_shape: 주먹/한 손가락/손바닥) → 동작 판정(gesture_filter:
     모양 래치 + 첫 선 궤적 + 탭) → 이벤트 print(stdio)
  ※머리 앵커(포즈)는 별도 스레드에서 초당 10회 — 손 루프 비차단
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ main.py                  # 공식 진입점 — 델파이가 직접 실행 (2026-08-03)
├─ install.bat              # 설치 — wheelhouse 있으면 오프라인, 없으면 받아서 번들 생성
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # hand_landmarker.task · pose_landmarker_lite.task (배포 zip 포함)
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/hand_tracker.py     # 손 랜드마크 (MediaPipe) — 유일한 추론 모델
│   ├─ inference/head_detector.py    # 머리 앵커 관측 (MediaPipe 포즈 — 마스크 무관)
│   ├─ postprocess/hand_select.py    # 앵커 게이트 + 단일 손 추적 + 손 실측 거리 자
│   ├─ postprocess/hand_shape.py     # 손 모양 — 주먹/한 손가락/손바닥 (21점 기하 규칙)
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 모양 래치 + 첫 선 궤적 + 탭 클릭
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   └─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (stdio/console)
├─ scripts/                 # pipe_listen · download_weights · benchmark · smoke_test · eval_accuracy
├─ tests/                   # 단위 테스트 176건 (카메라·모델 없이 실행 가능)
├─ docs/코드설명서.md       # 코드 지도 — 어디서 무엇을 하는지 (2026-08-03)
└─ docs/TODO.md             # 작업 분해 및 회사 확인 필요 항목
```

★ 표시는 **회사 키오스크 프로그램과의 연동 접점** (기획서 1.2, 9장 №7·№8).

## 실행 모드

| 명령 | 용도 |
|---|---|
| `py main.py` | 엔진 — 카메라·계기판 창을 켠 채 시작 (공식 실행 — 시스템 파이썬) |
| 실행 중 `cam on` / `cam off` (+Enter) | 카메라·계기판 창 켜기/끄기 — 재실행 불필요 |
| `py main.py --no-cam` | 창 없이 이벤트만(stdout) |
| `py scripts\pipe_listen.py` | 델파이 대역 — 파이프 수신 규격 자가 검증 |
| `py scriptsenchmark.py` | 추론 단독 FPS 측정 (기획서 6.1 — KPI 30 FPS) |
| `py -m unittest discover tests -v` | 판정·손 추적·손모양·시나리오 단위 테스트 |

## 회사 프로그램(UI) 연동 계약

1. 델파이가 엔진을 자식 프로세스로 실행 → stdout 파이프에서 줄 단위 수신 —
   규격·수신 코드는 **[docs/델파이7_연동가이드.md](docs/델파이7_연동가이드.md)**
2. 이벤트 **10종**: `left` `right` `select` `back` `home` `confirm` +
   `temp_left` `temp_right` `temp_top` `click` (07-29 개편 + 07-31 추가 — UI 분기문 갱신 필요)
3. 새 수신 규격 확정 시 `event_sender.py`에 Sender 1개 추가 — 파이프라인 수정 불필요

## 개인정보·라이선스 주의

- 엔진은 프레임·인식값을 저장하지 않고 로그는 마스킹한다 (설치가이드.md F절)
- **라이선스 (2026-07-29 기준)**: 스택 전체가 상업 사용 가능 + 코드 공개(카피레프트) 의무 없음 —
  MediaPipe(Apache-2.0) 단독. 손 모양 판별은 자체 기하 규칙이라 추가 의존성 0.
  Apache의 라이선스 문서 동봉(배포물 내 고지)은 통상 절차 — 제품 화면 표시 의무는 없다.
  구 HaGRID YOLOv10 ONNX 엔진(AGPL 리스크)·rtmlib/ONNX Runtime 포즈 스택은
  코드·가중치 모두 제거 완료 (기획서 9장 №9 — MediaPipe 하나로 해소)

## 참고 링크

- MediaPipe (Apache-2.0): https://github.com/google-ai-edge/mediapipe
