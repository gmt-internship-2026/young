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
venv_win\Scripts\python.exe main.py   :: 실행 — 이벤트가 stdout에 한 줄씩 (델파이 연동 동일)
run_debug.bat                        :: + 로컬 디버그 창 (카메라·판정 계기판)
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 델파이7 UI 연동(수신부 완성 코드 포함): **[docs/델파이7_연동가이드.md](docs/델파이7_연동가이드.md)**

## 인식 동작 (2026-07-29 개편 스펙 — 상하 포커스 제거, 위=select, ok→confirm)

| 이벤트 | 손 모양 | 이동 방향 | 키오스크 명령 |
|---|---|---|---|
| left / right / select | **한 손가락** (종류 무관) | 좌 / 우 / 위 | 포커스 1칸 이동 |
| back | **주먹** | 왼쪽 | 이전 화면 |
| home | **주먹** | 위 | 처음 화면 |
| confirm | **주먹** | 오른쪽 | 현재 항목 실행 |

- **핵심 규칙**: 손 모양이 계층을(탐색/명령), 이동 방향이 기능을 정한다 —
  반복 횟수·화면 좌표는 쓰지 않는다. 탐색(한 손가락)은 아무리 반복해도 화면이
  안 바뀌고, 화면을 바꾸는 동작은 주먹을 쥐어야만 실행된다 (오발 안전 구조)
- 손 모양은 **래치 상태기**(07-28)로 확정한다 — 저속에서 연속 판별로 고정,
  빠른 이동 중(모션 블러) 판별 동결, 반대 모양 연속 확인 시에만 전환.
  방향은 **첫 선 고정**(07-28) — 원점을 떠나는 첫 이동 벡터가 방향을 정한다
- 아래 방향은 정의 없음(두 모양 공통 — 07-29 bottom 제거) — 무시. 위 방향
  (select·home)은 팔 들어올리기(예비 동작) 오발을 휴식 존 게이트가 막는다
- 사용자 손 선별(hand_select, 07-29 포즈 잠금 대체): 쪽별 가장 큰 손 + 연속성
  우선 — 옆 사람 손의 순간 난입을 막는다. **★한 명 사용 가정** (상시 다중 인원
  구도면 포즈 잠금 구판(2ea58a5 이전) 검토 — docs/TODO.md №1-2)
- 스펙 변천: 주먹→펴기(07-15 제거) → 손등/팔등(07-15 2차 제거) → 고개 꾸벅(07-16
  제거) → 쓸기 일원화(07-16) → **현행: 손 모양 기준(07-23 — 보고서 개정 반영)**

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 손 랜드마크(MediaPipe HandLandmarker — 유일한 모델)
  → 사용자 손 선별(hand_select: 크기+연속성, 손 실측 거리 자) → 손 모양(hand_shape)
  → 동작 판정(gesture_filter: 손 모양 래치 + 첫 선 궤적 4방향) → 이벤트 print(stdio)
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ main.py                  # 공식 진입점 — 델파이가 직접 실행 (2026-08-03)
├─ install.bat / run_debug.bat / make_offline_bundle.bat  # 설치·현장 진단·번들 제작 (설치가이드.md)
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # hand_landmarker.task (8MB — download_weights.py가 받는다)
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/hand_tracker.py     # 손 랜드마크 (MediaPipe) — 유일한 추론 모델
│   ├─ postprocess/hand_select.py    # 사용자 손 선별(크기+연속성) + 손 실측 거리 자
│   ├─ postprocess/hand_shape.py     # 손 모양 판별 — 주먹/한 손가락 (손 21점 기하 규칙)
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 손 모양 래치 + 첫 선 궤적 4방향
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   └─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (stdio/console)
├─ scripts/                 # pipe_listen · download_weights · benchmark · smoke_test · eval_accuracy
├─ tests/                   # 단위 테스트 126건 (카메라·모델 없이 실행 가능)
└─ docs/TODO.md             # 작업 분해 및 회사 확인 필요 항목
```

★ 표시는 **회사 키오스크 프로그램과의 연동 접점** (기획서 1.2, 9장 №7·№8).

## 실행 모드

| 명령 | 용도 |
|---|---|
| `venv_win\Scripts\python.exe main.py` | 엔진 — 이벤트가 stdout에 한 줄씩 (공식 실행) |
| 실행 중 `cam on` / `cam off` (+Enter) | 카메라·계기판 창 켜기/끄기 — 재실행 불필요 |
| `main.py --debug` (= `run_debug.bat`) | 창을 켠 채 시작 |
| `python scripts/pipe_listen.py` | 델파이 대역 — 파이프 수신 규격 자가 검증 |
| `python scripts/benchmark.py` | 추론 단독 FPS 측정 (기획서 6.1 — KPI 30 FPS) |
| `python -m unittest discover tests -v` | 판정·잠금·손모양·시나리오 단위 테스트 |

## 회사 프로그램(UI) 연동 계약

1. 델파이가 엔진을 자식 프로세스로 실행 → stdout 파이프에서 줄 단위 수신 —
   규격·수신 코드는 **[docs/델파이7_연동가이드.md](docs/델파이7_연동가이드.md)**
2. 이벤트 6종: `left` `right` `select` `back` `home` `confirm` (2026-07-29 개편)
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
