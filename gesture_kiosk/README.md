# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 제스처를 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 + Python 3.11.5 — GPU 자동 감지 통합판** (정부 민원발급기):
  NVIDIA GPU가 있으면 CUDA로, 없으면 CPU로 실행 (config `device/pose_mode: auto`).
  2026-07-24 구 feat/think_win_cpu·feat/think_win_gpu 브랜치를 이 브랜치(feat/think_win)로 통합
- 동작 체계(2026-07-23 — 「제스처 정의 보고서」 손 모양 기준, 회사 확정):
  **손 모양이 계층을, 이동 방향이 기능을 정한다** — 한 손가락=탐색, 주먹=명령
- 모델: **RTMPose wholebody 포즈(Apache-2.0) 단일** — 손 모양·궤적·사용자 잠금이
  전부 키포인트 하나로 판정된다 (손 모양도 손 21점 기하 규칙 — 별도 CNN 없음)
- 연동: **파이프(stdio)** — 이벤트를 stdout에 한 줄씩 print, 델파이7 UI가 파이프로
  수신 (2026-07-23 회사 확정 — 네트워크(UDP·웹소켓) 전면 철회)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — GPU 유무 자동 감지, 내부망은 설치가이드.md B절
run.bat            :: 실행 — 이벤트가 콘솔(stdout)에 GESTURE| 한 줄씩
run.bat --debug    :: + 로컬 디버그 창 (카메라·판정 계기판)
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 델파이 UI 연동·데모: **[delphi_ui/빌드_실행_안내.md](delphi_ui/빌드_실행_안내.md)**

## 인식 동작 (2026-07-23 확정 스펙)

| 이벤트 | 손 모양 | 이동 방향 | 키오스크 명령 |
|---|---|---|---|
| left / right / top / bottom | **한 손가락** (종류 무관) | 좌 / 우 / 위 / 아래 | 포커스 1칸 이동 |
| back | **주먹** | 왼쪽 | 이전 화면 |
| home | **주먹** | 위 | 처음 화면 |
| ok | **주먹** | 오른쪽 | 현재 항목 실행 |

- **핵심 규칙**: 손 모양이 계층을(탐색/명령), 이동 방향이 기능을 정한다 —
  반복 횟수·화면 좌표는 쓰지 않는다. 탐색(한 손가락)은 아무리 반복해도 화면이
  안 바뀌고, 화면을 바꾸는 동작은 주먹을 쥐어야만 실행된다 (오발 안전 구조)
- 손 모양 판별은 프레임별로 흔들릴 수 있어 **궤적 창 안 다수결**로 확정한다 —
  블러로 판별이 끊겨도(기권) 이벤트가 유실되지 않는다
- 주먹+아래는 정의 없음 — 무시. 위 방향(top·home)은 팔 들어올리기(예비 동작)
  오발을 휴식 존 게이트가 막는다
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손만 인식 — **다른 사람 손 무시**
  (IoU 동일인 매칭으로 대기줄에서 잠금 안 뺏김, 도달 거리 게이트로 옆 사람 손 차단)
- 스펙 변천: 주먹→펴기(07-15 제거) → 손등/팔등(07-15 2차 제거) → 고개 꾸벅(07-16
  제거) → 쓸기 일원화(07-16) → **현행: 손 모양 기준(07-23 — 보고서 개정 반영)**

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 사람 포즈(rtmlib RTMPose wholebody — 유일한 모델)
  → 사용자 잠금(person_lock: 얼굴 선명도×크기 + IoU 추적) → 손 모양·손 중심(hand_shape)
  → 동작 판정(gesture_filter: 궤적 4방향 + 손 모양 다수결) → 이벤트 print(stdio)
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ install.bat / run.bat / make_offline_bundle.bat  # 윈도우 이식·실행 (설치가이드.md)
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # (비어 있음 — 포즈 모델은 ~/.cache/rtmlib 자동 캐시)
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/pose_estimator.py   # 사람 포즈 (rtmlib RTMPose) — 유일한 추론 모델
│   ├─ postprocess/person_lock.py    # 사용자 잠금(IoU 추적) + 손 신호(모양·중심)·어깨너비
│   ├─ postprocess/hand_shape.py     # 손 모양 판별 — 주먹/한 손가락 (손 21점 기하 규칙)
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 궤적 4방향 + 손 모양 다수결
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   └─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (stdio/console)
├─ scripts/                 # run_demo · pipe_listen · download_weights · benchmark · smoke_test
├─ tests/                   # 단위 테스트 121건 (카메라·모델 없이 실행 가능)
├─ delphi_ui/               # ★ 델파이7 데모 UI — 파이프 수신 + 포커스 이동 (우리가 작성)
└─ docs/TODO.md             # 작업 분해 및 회사 확인 필요 항목
```

★ 표시는 **회사 키오스크 프로그램과의 연동 접점** (기획서 1.2, 9장 №7·№8).

## 실행 모드

| 명령 | 용도 |
|---|---|
| `run.bat` / `python scripts/run_demo.py` | 엔진 — 이벤트가 stdout에 `GESTURE\|...` 한 줄씩 |
| `run.bat --debug` | + 로컬 디버그 창 (카메라·판정 계기판 오버레이) |
| `python scripts/pipe_listen.py` | 델파이 대역 — 파이프 수신 규격 자가 검증 |
| `python scripts/benchmark.py` | 추론 단독 FPS 측정 (기획서 6.1 — KPI 30 FPS) |
| `python -m unittest discover tests -v` | 판정·잠금·손모양·시나리오 단위 테스트 |

## 회사 프로그램(UI) 연동 계약

1. 델파이가 엔진을 자식 프로세스로 실행 → stdout 파이프에서 줄 단위 수신 —
   규격·수신 코드는 **[docs/델파이7_연동가이드.md](docs/델파이7_연동가이드.md)**
2. 이벤트 7종: `left` `right` `top` `bottom` `back` `home` `ok` (config classes와 동일)
3. 새 수신 규격 확정 시 `event_sender.py`에 Sender 1개 추가 — 파이프라인 수정 불필요

## 개인정보·라이선스 주의

- 엔진은 프레임·인식값을 저장하지 않고 로그는 마스킹한다 (설치가이드.md F절)
- **라이선스 (2026-07-23 기준)**: 스택 전체가 상업 사용 가능 + 코드 공개(카피레프트) 의무 없음 —
  rtmlib/RTMPose(Apache-2.0) · ONNX Runtime(MIT). 손 모양 판별은 자체 기하 규칙이라
  추가 의존성 0 (새 스펙 전환에도 검토 대상 불변).
  Apache/MIT의 라이선스 문서 동봉(배포물 내 고지)은 통상 절차 — 제품 화면 표시 의무는 없다.
  구 HaGRID YOLOv10 ONNX 엔진(AGPL 리스크)은 코드·가중치 모두 제거 완료 (기획서 9장 №9 해소)

## 참고 링크

- rtmlib (RTMPose, Apache-2.0): https://github.com/Tau-J/rtmlib
