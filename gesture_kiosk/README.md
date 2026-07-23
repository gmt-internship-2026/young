# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 손 제스처를 실시간 인식해
키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 CPU(GPU 불필요) + Python 3.11.5** — CPU 추론판 (정부 민원발급기)
- GPU 있는 PC용 고성능판: **feat/think_win_gpu 브랜치** (같은 코드 — 설치 스택·성능 기준만 다름)
- 실행 환경: **윈도우 + NVIDIA GPU + Python 3.11.5** (2026-07-10 타깃 변경 — 정부 민원발급기)
- 동작 체계(2026-07-23 전면 개편 — 손 모양+이동으로 통합): **검지 1개(point, "가리키기")를
  펴고 좌/우/상/하로 이동 = 포커스 이동 4방향** + **주먹(fist)을 내밀고 좌/우/상으로
  이동 = 이전/확인/홈**(아래는 미사용) — 무손·무지 사용자 접근성 요건은 계획에서
  빠졌다(회사 확인 필요, docs/TODO.md №1)
- 모델: **RTMPose 포즈(Apache-2.0)** — 사용자 잠금(얼굴) 전용 + **MediaPipe HandLandmarker
  (Apache-2.0)** — 손 모양·손 위치 이동 판정 전담. 잠긴 사용자 bbox 크롭만 추론해 CPU FPS·
  다른 사람 손 오인식을 방지 (학습 0회 스택 유지)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — 내부망은 설치가이드.md B절
run.bat            :: 실행 — 브라우저 http://localhost:5000
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 개발 맥에서는 `venv` 활성화 후 `python scripts/run_demo.py` (torch 백엔드 그대로).

## 인식 동작 (2026-07-23 전면 개편 스펙 — 손 모양(point/fist) + 이동 방향으로 확정)

| action | 동작 | 판정 방식 | 키오스크 명령 |
|---|---|---|---|
| move_left / move_right / move_up / move_down | **검지 1개만 편 채(point)** 손을 좌/우/상/하로 이동 | MediaPipe 손 랜드마크(손목) 궤적 (window 내 이동량·주축 우세) | 포커스 이동 4방향 |
| select | **주먹(fist)을 낸 채** 손을 우측으로 이동 | 〃 | 선택·확인 |
| go_back | **주먹을 낸 채** 손을 좌측으로 이동 | 〃 | 이전 화면 |
| go_home | **주먹을 낸 채** 손을 상단으로 이동 | 〃 | 처음 화면으로 |

- 손 모양(point=검지 1개, fist=전부 접음)과 이동 궤적이 **같은 MediaPipe 손 랜드마크
  프레임**에서 나와 항상 동기화된다 — 옛 체계(팔 쓸기는 RTMPose 포즈 손목, 화면 전환은
  MediaPipe 손 위치로 출처가 달라 손 모양을 이동 판정에 반영할 수 없었다)의 한계를 해소
- **fist+아래**는 매핑이 없다(미사용) — 회사 스펙에 정의되지 않음(2026-07-23 사용자 확정).
  방향 자체는 감지되지만 이벤트가 나가지 않는다
- point/fist가 아닌 손가락 개수(2개 이상 등)는 어느 쪽도 추적하지 않는다 — 이동 중이
  아닌 것으로 본다
- **2026-07-16부로 무손·무지 사용자 접근성 요건은 계획에서 빠졌다(회사 확인 필요,
  docs/TODO.md №1)** — 이번 개편도 이 전제 위에서 진행했다(손이 있어야 손 모양을
  만들 수 있으므로 옛 팔꿈치 폴백 같은 무손 대체 경로는 없다)
- 상하 포커스 이동 없음 — **줄 끝에서 다음 줄 첫 칸 랩(토크백식 선형 순회)은 UI 책임**
- 잠긴 사용자(초점 맞은 얼굴 기준)의 bbox로 크롭한 손만 인식 — **다른 사람 손 무시**
- 구 동작(주먹→펴기·OK핀치·양손바닥 10초·손등 보이기·고개 꾸벅, 팔 쓸기+손가락 정지
  유지)과 레거시 토글은 2026-07-15~2026-07-23에 순차 제거 — 직원 호출(help_call)은
  트리거가 사라져 이벤트 계약에서도 제외 (회사 협의 №1)

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 사람 포즈(rtmlib RTMPose) → 사용자 잠금(person_lock:
  얼굴 선명도×크기) → 잠긴 사용자 bbox 크롭 → 손 랜드마크(MediaPipe HandLandmarker)
  → 동작 판정(gesture_filter: 손 모양(point/fist) + 손 위치 이동 방향을 함께 봐서
  4방향 이동·선택·이전·홈 확정) → 이벤트 전송(event_sender) + 음성 안내(announce)
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
│   ├─ inference/pose_estimator.py   # 사람 포즈 (rtmlib RTMPose) — 사용자 잠금(얼굴) 전용
│   ├─ inference/hand_estimator.py   # 손 랜드마크 (MediaPipe HandLandmarker) — 손 모양·위치
│   ├─ postprocess/person_lock.py    # 사용자 잠금(얼굴 선명도×크기) + bbox 산출
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 손 모양(point/fist) + 이동 방향
│   ├─ announce/announcer.py         # 토크백 TTS (pyttsx3 — SAPI/nsss)
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp/pipe — 실연동은 pipe)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test
├─ tests/                   # 단위 테스트 79건 (카메라·모델 없이 실행 가능)
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
| `python -m unittest discover tests -v` | 판정·잠금·이벤트 전송 단위 테스트 (79건) |

## 회사 프로그램(델파이7) 연동 계약

1. **실연동: 네임드 파이프**(2026-07-23 팀 확정) — 델파이7이 파이프 **서버**, 이 엔진이
   **클라이언트**로 접속(`event_output.mode: pipe`, `event_sender.py`의
   `PipeEventSender`). JSON이 아니라 **평문 명령어 7개 고정**을 개행 구분으로 전송한다:
   `up`/`down`/`left`/`right`/`home`/`back`/`ok` — 델파이 쪽은 파이프에 들어온 한 줄을
   그대로 인식해 실행(팀장 확인, 별도 파싱 불필요)
2. `event_output.pipe.name`은 아직 placeholder(`\\.\pipe\GestureKiosk`) — 델파이7
   개발자가 확정한 실제 파이프 이름으로 교체 후 `mode: pipe`로 전환할 것
3. 개발·디버깅용으로 `console`/`udp`도 남아있다(`/data` 폴링으로 JSON 확인 가능)
4. 새 수신 규격이 필요해지면 `event_sender.py`에 Sender 1개만 추가 — 파이프라인 수정 불필요
5. 음성 안내(UI→엔진): `POST /announce {"text": "발급하기 버튼"}` — 포커스 항목을 TTS로
6. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

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
