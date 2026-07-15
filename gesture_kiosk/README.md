# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 손 제스처와 주민등록증을 실시간
인식해 키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 CPU(GPU 불필요) + Python 3.11.5** — CPU 추론판 (정부 민원발급기)
- GPU 있는 PC용 고성능판: **feat/think_win_gpu 브랜치** (같은 코드 — 설치 스택·성능 기준만 다름)
- 동작 체계(2026-07-15 개편): **팔 쓸기(좌/우=이동·아래=이전·위=처음) + 손등/팔등 보이기(선택)**
  — 장애인·비장애인 범용 설계: 손가락·손이 없어도 팔만으로 모든 조작이 가능하다
- 모델: MediaPipe 손 랜드마크(Apache-2.0, 손등/손바닥) + RTMPose 포즈(Apache-2.0, 쓸기·사용자 잠금)
  + **자체 학습 팔등 분류 CNN(제3자 가중치 없음)** — 카피레프트·고지 의무 없는 상용 안전 스택
- 파인튜닝 계열 학습은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 팔등 분류기만
  이 폴더의 scripts/collect_arm_side.py·train_arm_side.py로 만든다 (자체 데이터 필수)

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
| move_left / move_right | 팔을 **좌/우로 쓸기** | 포즈 손목 궤적 (window 내 이동량·주축 우세) | 포커스 1칸 이동 |
| go_back | 팔을 **아래로 쓸기** | 〃 | 이전 화면 |
| go_home | 팔을 **위로 쓸기** | 〃 | 처음 화면으로 |
| select | **손등/팔등 보이기** 유지 | 손: 랜드마크 외적 부호 / 팔: 자체 CNN | 선택·확인 |
| fill_id_fields | 주민등록증 제시 | OCR 모드에서 EasyOCR 판독 | 이름·주민번호 자동 입력 |

- **범용 설계 근거**: 쓸기는 손이 아니라 **손목 키포인트(포즈)** 궤적이라 손·손가락이
  없어도 동작한다. 선택도 손이 검출되면 손등, 안 되면 전완(팔등) 분류로 이중화 —
  손등/팔등 판정 모두 손가락 유무와 무관하다 (MCP·전완만 사용)
- 상하 포커스 이동 없음 — **줄 끝에서 다음 줄 첫 칸 랩(토크백식 선형 순회)은 UI 책임**
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손목·팔만 인식 — **다른 사람 손 무시**
- 구 동작(주먹→펴기·OK핀치·양손바닥 10초)과 레거시 토글은 2026-07-15 제거 —
  직원 호출(help_call)은 트리거가 사라져 이벤트 계약에서도 제외 (회사 협의 №1)

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 손등/손바닥 검출(MediaPipe 랜드마크) + 사람 포즈(rtmlib RTMPose)
  → 사용자 잠금(person_lock: 얼굴 선명도×크기) → 손 귀속(왼/오른) + 팔등 분류(arm_side CNN)
  → 동작 판정(gesture_filter: 손목 쓸기 궤적·손등 유지) → 이벤트 전송(event_sender) + 음성 안내(announce)
주민등록증 OCR(easyocr)은 별도 워커 스레드 — UI가 /ocr/start로 요청할 때만
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ install.bat / run.bat / make_offline_bundle.bat  # 윈도우 이식·실행 (설치가이드.md)
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # hand_landmarker.task(손) + arm_side_cnn.onnx(팔등 — 자체 학습)
├─ data/raw/arm_side/       # 팔등 분류 학습 데이터 (collect_arm_side.py — 인물별 폴더)
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/detector_mediapipe.py  # 손등/손바닥 검출 (MediaPipe 랜드마크 외적 부호)
│   ├─ inference/arm_side_classifier.py # 팔등(전완 등쪽) 분류 — 자체 학습 CNN (ONNX)
│   ├─ inference/detector.py         # 공통 Detection 구조 + 엔진 팩토리 + ORT 헬퍼
│   ├─ inference/pose_estimator.py   # 사람 포즈 (rtmlib RTMPose) — 손목·팔꿈치, 얼굴 키포인트
│   ├─ postprocess/person_lock.py    # 사용자 잠금 + 손 귀속 (거울 좌/우 보정)
│   ├─ postprocess/gesture_filter.py # 동작 판정 — 손목 쓸기 궤적 + 손등/팔등 유지
│   ├─ ocr/idcard_reader.py          # 주민등록증 이름·주민번호 (마스킹 로그)
│   ├─ announce/announcer.py         # 토크백 TTS (pyttsx3 — SAPI/nsss)
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce·/ocr 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test
│                           #   · collect_arm_side / train_arm_side (팔등 분류기 제작)
├─ tests/                   # 단위 테스트 64건 (카메라·모델 없이 실행 가능)
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
| `python -m unittest discover tests -v` | 판정·잠금·OCR 파싱·단위 테스트 (64건) |

## 회사 프로그램(UI) 연동 계약

1. 이벤트(엔진→UI): `event_output.mode`(console/udp) 또는 `/data` 폴링 —
   JSON `{"class_name": "move_right", "conf": 0.87, "ts_sec": ..., "hand_side": "right"}`
2. 음성 안내(UI→엔진): `POST /announce {"text": "발급하기 버튼"}` — 포커스 항목을 TTS로
3. 주민등록증(UI→엔진): `POST /ocr/start` → 성공 시 `fill_id_fields` 이벤트(data에 이름·주민번호)
4. 새 수신 규격 확정 시 `event_sender.py`에 Sender 1개 추가 — 파이프라인 수정 불필요
5. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

## 개인정보·라이선스 주의

- **주민등록번호 처리 법적 근거(개인정보보호법 제24조의2) — 회사 확인 필수** (docs/TODO.md №11).
  엔진은 프레임·인식값을 저장하지 않고 로그는 마스킹한다 (설치가이드.md F절)
- **라이선스 (2026-07-15 기준)**: 스택 전체가 상업 사용 가능 + 코드 공개(카피레프트) 의무 없음 —
  MediaPipe(Apache-2.0) · rtmlib/RTMPose(Apache-2.0) · ONNX Runtime(MIT) ·
  EasyOCR(Apache-2.0) · pyttsx3(MPL-2.0 — 무수정 사용이라 공개 의무 없음).
  **팔등 분류기는 자체 데이터·무(無) 사전학습으로 직접 학습한 자산이라 제3자 의무가 아예 없다.**
  Apache/MIT의 라이선스 문서 동봉(배포물 내 고지)은 통상 절차 — 제품 화면 표시 의무는 없다.
  구 HaGRID YOLOv10 ONNX 엔진(AGPL 리스크·비교 시험용)은 신규 스펙을 판정할 수 없어
  2026-07-15 코드·가중치 모두 제거 완료 (기획서 9장 №9 잔여 확인 항목 해소)

## 참고 링크

- MediaPipe Hand Landmarker (Apache-2.0): https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- rtmlib (RTMPose, Apache-2.0): https://github.com/Tau-J/rtmlib
- EasyOCR: https://github.com/JaidedAI/EasyOCR
