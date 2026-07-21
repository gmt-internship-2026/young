# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 제스처를 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. (2026-07-16 주민등록증 OCR 제거 — 제스처 집중) **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 CPU(GPU 불필요) + Python 3.11.5** — CPU 추론판 (정부 민원발급기)
- GPU 있는 PC용 고성능판: **feat/think_win_gpu 브랜치** (같은 코드 — 설치 스택·성능 기준만 다름)
- 동작 체계(2026-07-16 토크백 정렬): **좌/우 쓸기=이동 · 위/아래 1회=확인/선택 ·
  위/아래 2연속=처음/이전(안전장치)** — 손·손가락이 없어도 팔(팔꿈치 폴백)로 조작
- 모델: **RTMPose 포즈(Apache-2.0) 단일** — 쓸기·사용자 잠금이 전부 키포인트
  하나로 판정된다 (학습 0회 스택)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — 내부망은 설치가이드.md B절
run.bat            :: 실행 — 브라우저 http://localhost:5000
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 개발 맥에서는 `venv` 활성화 후 `python scripts/run_demo.py` (torch 백엔드 그대로).

## 인식 동작 (2026-07-15 개편 스펙 — 범용 설계)

| action | 동작 | 판정 방식 | 키오스크 명령 |
|---|---|---|---|
| move_left / move_right | 팔을 **좌/우로 쓸기** | 포즈 손목 궤적 — 즉시 확정 | 포커스 1칸 이동 |
| read_focus | **위로 쓸기 1회** | 판정 창(1.2초) 경과 후 확정 | 확인 — 현재 항목 다시 읽기 |
| select | **아래로 쓸기 1회** | 〃 (보이스오버 더블탭 대응) | 선택·실행 |
| go_home | **위로 2연속** 쓸기 | 판정 창 안에 같은 방향 2회 | 처음 화면 (안전장치) |
| go_back | **아래로 2연속** 쓸기 | 〃 | 이전 화면 (안전장치) |

- **토크백/보이스오버 문법 정렬(2026-07-16)**: 좌/우 쓸기=포커스 순회, 아래 1회=선택
  (더블탭 대응), 위 1회=현재 항목 확인 — 스크린리더 사용자의 기존 습관과 일치.
  홈·이전은 화면을 이탈하는 파괴적 동작이라 **2연속 쓸기**를 요구한다 (오발 안전장치)
- **범용 설계**: 쓸기는 손이 아니라 **손목 키포인트(포즈)** 궤적이라 손·손가락이
  없어도 동작하고 — 손목 키포인트가 신뢰도 미달(절단 등)이면 **팔꿈치로 자동 폴백**
  (elbow_gain 보정, 화면 추적점에 "(E)" 표시). ⚠ 고개 꾸벅 제거로 양팔이 모두 없는
  사용자의 선택 수단은 UI 자동 순회(스캐닝) 협의 필요 — №1.

- 상하 포커스 이동 없음 — **줄 끝에서 다음 줄 첫 칸 랩(토크백식 선형 순회)은 UI 책임**
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손목·팔만 인식 — **다른 사람 손 무시**
- 스펙 변천: 주먹→펴기 체계(07-15 제거) → 손등/팔등 보이기(07-15 2차 제거) →
  고개 꾸벅 2회(07-16 제거) → **현행: 쓸기 일원화 토크백 정렬**. help_call은 계약 제외 (№1)

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 사람 포즈(rtmlib RTMPose — 유일한 모델)
  → 사용자 잠금(person_lock: 얼굴 선명도×크기) → 손목 좌/우 보정·목 길이 비율
  → 동작 판정(gesture_filter: 손목 쓸기 궤적 — 1회/2연속 분기) → 이벤트 전송(event_sender) + 음성 안내(announce)
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
│   ├─ postprocess/person_lock.py    # 사용자 잠금 + 쓸기 추적점(손목→팔꿈치 폴백)·어깨너비
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 쓸기 궤적(1회/2연속 분기)
│   ├─ announce/announcer.py         # 토크백 TTS (pyttsx3 — SAPI/nsss)
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test
├─ tests/                   # 단위 테스트 46건 (카메라·모델 없이 실행 가능)
├─ demo_ui/index.html       # ★ 예시 민원발급기 화면 (회사 UI 수령 시 교체)
└─ docs/TODO.md             # 작업 분해 및 회사 확인 필요 항목
```

★ 표시는 **회사 키오스크 프로그램을 받으면 교체/제거되는 부분** (기획서 1.2, 9장 №7·№8).

## 실행 모드

| 명령 | 용도 |
|---|---|
| `run.bat` / `python scripts/run_demo.py` | 파이프라인 + 예시 UI (시연용) |
| `run.bat --headless` | 파이프라인만 — 이벤트는 `event_output` 설정대로 전송 |
| `python scripts/benchmark.py` | 추론 단독 FPS 측정 (기획서 6.1 — KPI 30 FPS) |
| `python -m unittest discover tests -v` | 판정·잠금·쓸기 시나리오 단위 테스트 (104건) |

## 회사 프로그램(UI) 연동 계약

1. 이벤트(엔진→UI): `event_output.mode`(console/udp) 또는 `/data` 폴링 —
   JSON `{"class_name": "move_right", "conf": 0.87, "ts_sec": ..., "hand_side": "right"}`
2. 음성 안내(UI→엔진): `POST /announce {"text": "발급하기 버튼"}` — 포커스 항목을 TTS로
3. 새 수신 규격 확정 시 `event_sender.py`에 Sender 1개 추가 — 파이프라인 수정 불필요
5. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

## 개인정보·라이선스 주의

- **개인정보**: 2026-07-16 OCR 제거로 엔진은 개인정보를 다루지 않는다 — 입력은 포즈
  좌표뿐, 카메라 프레임 미저장 (구 №11 쟁점 소멸)
- **라이선스 (2026-07-15 2차 기준)**: 스택 전체가 상업 사용 가능 + 코드 공개(카피레프트) 의무 없음 —
  rtmlib/RTMPose(Apache-2.0) · ONNX Runtime(MIT) ·
  pyttsx3(MPL-2.0 — 무수정 사용이라 공개 의무 없음). 추론 모델이 포즈 하나뿐이라
  검토 대상 자체가 최소화됐다 (MediaPipe·자체 학습 CNN도 2차에서 제거).
  Apache/MIT의 라이선스 문서 동봉(배포물 내 고지)은 통상 절차 — 제품 화면 표시 의무는 없다.
  구 HaGRID YOLOv10 ONNX 엔진(AGPL 리스크)은 코드·가중치 모두 제거 완료 (기획서 9장 №9 해소)

## 참고 링크

- rtmlib (RTMPose, Apache-2.0): https://github.com/Tau-J/rtmlib
