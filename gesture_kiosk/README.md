# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 손 제스처를 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 CPU(GPU 불필요) + Python 3.11.5** — CPU 추론판 (정부 민원발급기)
- GPU 있는 PC용 고성능판: **feat/think_win_gpu 브랜치** (같은 코드 — 설치 스택·성능 기준만 다름)
- 실행 환경: **윈도우 + NVIDIA GPU + Python 3.11.5** (2026-07-10 타깃 변경 — 정부 민원발급기)
- 동작 체계(2026-07-20 개편 — 이동은 팔, 화면 전환은 손가락으로 역할 분리): **팔 쓸기
  (좌/우=이동) + 손가락 1개 신호(위/아래 이동=화면 전환, 제자리 유지=선택)** —
  무손·무지 사용자 접근성 요건은 계획에서 빠졌다(회사 확인 필요, docs/TODO.md №1).
  좌/우 쓸기는 여전히 손목→팔꿈치 폴백으로 손이 없어도 동작한다
- 모델: **RTMPose 포즈(Apache-2.0)** — 쓸기·사용자 잠금 담당 + **MediaPipe HandLandmarker
  (Apache-2.0)** — 선택(손가락 인식) 담당. 잠긴 사용자 bbox 크롭만 추론해 CPU FPS·
  다른 사람 손 오인식을 방지 (학습 0회 스택 유지)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — 내부망은 설치가이드.md B절
run.bat            :: 실행 — 브라우저 http://localhost:5000
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 개발 맥에서는 `venv` 활성화 후 `python scripts/run_demo.py` (torch 백엔드 그대로).

## 인식 동작 (2026-07-20 개편 스펙 — 이동은 팔, 화면 전환은 손 위치, 선택만 손가락 개수)

| action | 동작 | 판정 방식 | 키오스크 명령 |
|---|---|---|---|
| move_left / move_right | 팔을 **좌/우로 쓸기** | 포즈 손목 궤적 (window 내 이동량·주축 우세) | 포커스 1칸 이동 |
| go_home | **손을 위로 이동** (손 모양 무관) | 잠긴 사용자 bbox 크롭 → MediaPipe 손 랜드마크(손목) 궤적 | 처음 화면으로 |
| go_back | **손을 아래로 이동** (손 모양 무관) | 〃 | 이전 화면 |
| select | **손가락 1개(엄지 제외)를 편 채로 제자리에서 0.6초 이상 유지** | 〃(위치는 그대로, 유지 시간만 확인) | 선택·확인 |

- 좌/우 쓸기는 손이 아니라 **손목 키포인트(포즈)** 궤적이라 손·손가락이 없어도 동작하고 —
  손목 키포인트가 신뢰도 미달(절단 등)이면 **팔꿈치로 자동 폴백**(elbow_gain 보정,
  화면 추적점에 "(E)" 표시)
- go_home/go_back은 **손 위치 이동만** 보고 손가락 개수는 안 본다 — 손을 어떤 모양으로
  하든 위/아래로만 움직이면 된다. **손가락 1개**는 select 전용 신호다(2026-07-20 재확정
  — 처음엔 select와 같은 "손가락 1개 신호"로 통일했으나, 화면 전환까지 손가락 모양을
  요구할 필요는 없다고 판단해 손 위치만으로 완화). 엄지는 손 방향(좌/우 손·거울 반전)에
  따라 판정 축이 달라져 select 집계에서 제외
- **2026-07-16부로 무손·무지 사용자 접근성 요건은 계획에서 빠졌다(회사 확인 필요,
  docs/TODO.md №1)** — 이전(2026-07-15)에는 팔이 전혀 없어도 고개 꾸벅으로 선택
  가능했으나 UX 부담으로 기각됐다
- 상하 포커스 이동 없음 — **줄 끝에서 다음 줄 첫 칸 랩(토크백식 선형 순회)은 UI 책임**
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손목·팔·손만 인식 — **다른 사람 손 무시**
- 구 동작(주먹→펴기·OK핀치·양손바닥 10초·손등 보이기·고개 꾸벅)과 레거시 토글은
  2026-07-15·2026-07-16에 순차 제거 — 직원 호출(help_call)은 트리거가 사라져
  이벤트 계약에서도 제외 (회사 협의 №1)

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 사람 포즈(rtmlib RTMPose) → 사용자 잠금(person_lock:
  얼굴 선명도×크기) → 손목 좌/우 보정 → 잠긴 사용자 bbox 크롭 → 손 랜드마크
  (MediaPipe HandLandmarker) → 동작 판정(gesture_filter: 손목 쓸기 궤적 + 손가락
  1개 인식) → 이벤트 전송(event_sender) + 음성 안내(announce)
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ install.bat / run.bat / make_offline_bundle.bat  # 윈도우 이식·실행 (설치가이드.md)
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # hand_landmarker.task(손 모델, 저장소 포함) — 포즈 모델은
│                           #   ~/.cache/rtmlib 자동 캐시라 여기 없음
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/pose_estimator.py   # 사람 포즈 (rtmlib RTMPose) — 쓸기·사용자 잠금
│   ├─ inference/hand_estimator.py   # 손 랜드마크 (MediaPipe HandLandmarker) — 선택 판정
│   ├─ postprocess/person_lock.py    # 사용자 잠금 + 쓸기 추적점(손목→팔꿈치 폴백)
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 손목 쓸기 궤적 + 손가락 1개 인식(선택)
│   ├─ announce/announcer.py         # 토크백 TTS (pyttsx3 — SAPI/nsss)
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test
├─ tests/                   # 단위 테스트 35건 (카메라·모델 없이 실행 가능)
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
| `python -m unittest discover tests -v` | 판정·잠금 단위 테스트 (35건) |

## 회사 프로그램(UI) 연동 계약

1. 이벤트(엔진→UI): `event_output.mode`(console/udp) 또는 `/data` 폴링 —
   JSON `{"class_name": "move_right", "conf": 0.87, "ts_sec": ..., "hand_side": "right"}`
2. 음성 안내(UI→엔진): `POST /announce {"text": "발급하기 버튼"}` — 포커스 항목을 TTS로
3. 새 수신 규격 확정 시 `event_sender.py`에 Sender 1개 추가 — 파이프라인 수정 불필요
4. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

## 개인정보·라이선스 주의

- 주민등록증 인식(OCR) 기능은 **2026-07-16 전면 제거됐다** — 계획 변경으로 더는
  개인정보(주민등록번호) 처리 자체가 없다. 관련 법적 근거 검토 항목(docs/TODO.md
  №11)도 해당 없음으로 종료.
- **라이선스**: 스택 전체가 상업 사용 가능 + 코드 공개(카피레프트) 의무 없음 —
  rtmlib/RTMPose(Apache-2.0) · MediaPipe(Apache-2.0) · ONNX Runtime(MIT) ·
  pyttsx3(MPL-2.0 — 무수정 사용이라 공개 의무 없음).
  Apache/MIT의 라이선스 문서 동봉(배포물 내 고지)은 통상 절차 — 제품 화면 표시 의무는 없다.
  구 HaGRID YOLOv10 ONNX 엔진(AGPL 리스크)은 코드·가중치 모두 제거 완료 (기획서 9장 №9 해소)

## 참고 링크

- rtmlib (RTMPose, Apache-2.0): https://github.com/Tau-J/rtmlib
- MediaPipe: https://github.com/google-ai-edge/mediapipe
