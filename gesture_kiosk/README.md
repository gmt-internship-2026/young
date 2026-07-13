# gesture_kiosk — 제스처 인식 배리어프리 민원발급기 (추론)

(주)광명테크 인턴 프로젝트. USB 카메라 1대로 손 제스처와 주민등록증을 실시간
인식해 키오스크 프로그램으로 이벤트를 전달한다. **기획서(기획서.docx)의
2.3 디렉터리 구조와 4장 코딩 컨벤션을 따른다.**

- 실행 환경: **윈도우 + NVIDIA GPU + Python 3.11.5** (2026-07-10 타깃 변경 — 정부 민원발급기)
- GPU 없는 PC용 CPU 추론판: **feat/think_win_cpu 브랜치** (같은 코드 — 설치 스택·성능 기준만 다름)
- 모델: MediaPipe 손 랜드마크 제스처(Apache-2.0) + RTMPose 포즈(사용자 잠금) —
  **학습 0회, 카피레프트 없는 상용 안전 스택** (2026-07-10 C안: AGPL 계열 HaGRID YOLOv10 제거)
- 학습(파인튜닝)은 별도 `training/` 폴더 담당 (feat/study 브랜치) — 이 폴더는 추론 전용

## 빠른 시작 (윈도우)

```bat
install.bat        :: 설치 (인터넷) — 내부망은 설치가이드.md B절
run.bat            :: 실행 — 브라우저 http://localhost:5000
```

> 상세 절차·내부망(오프라인) 반입·문제 해결: **[설치가이드.md](설치가이드.md)**
> 개발 맥에서는 `venv` 활성화 후 `python scripts/run_demo.py` (torch 백엔드 그대로).

## 인식 동작 (2026-07-10 확정 스펙)

| action | 동작 | 판정 방식 | 키오스크 명령 |
|---|---|---|---|
| move_left | **왼손** 주먹 쥐었다 펴기 | fist N프레임 → open_within_sec 안 펴짐 | 포커스 왼쪽 1칸 |
| move_right | **오른손** 주먹 쥐었다 펴기 | 〃 | 포커스 오른쪽 1칸 |
| select | OK 사인 | N프레임 정적 유지 | 선택·확인 (통일) |
| go_home | 양 손바닥 | 10초 이상 유지 | 처음 화면으로 |
| fill_id_fields | 주민등록증 제시 | OCR 모드에서 EasyOCR 판독 | 이름·주민번호 자동 입력 |

- 상하 이동 없음 — **줄 끝에서 다음 줄 첫 칸 랩(토크백식 선형 순회)은 UI 책임**
- 왼/오른손 구분은 **손 모양 자체(MediaPipe handedness)** 로 판정 (2026-07-10) —
  포즈 손목 키포인트가 없어도 동작하므로 **한쪽 팔이 없는 사용자도 인식된다**
- 잠긴 사용자(초점 맞은 얼굴 기준)의 손만 인식 — **다른 사람 손 무시**
  (손목 근접 검사, 손목 키포인트 소실 시 잠긴 사람 박스 근접 검사로 폴백)
- 레거시 동작(point/palm 스와이프/thumbs_up 등 기획서 5.1 초안)은
  `gestures.legacy.enabled`로 병행 유지 중 — 회사 협의(№1) 후 정리
- "양 손바닥 유지"를 직원 호출로 쓰려면 `gestures.two_palm.action: help_call`

## 처리 흐름

```
카메라(스레드) → 거울 반전 → 제스처 검출(MediaPipe 손 랜드마크 + 기하 판정) + 사람 포즈(rtmlib RTMPose)
  → 사용자 잠금(person_lock: 얼굴 선명도×크기) → 손 귀속(왼/오른)
  → 동작 판정(gesture_filter FSM) → 이벤트 전송(event_sender) + 음성 안내(announce)
주민등록증 OCR(easyocr)은 별도 워커 스레드 — UI가 /ocr/start로 요청할 때만
```

## 폴더 구조 (기획서 2.3 + 신규 모듈)

```
gesture_kiosk/
├─ install.bat / run.bat / make_offline_bundle.bat  # 윈도우 이식·실행 (설치가이드.md)
├─ configs/config.yaml      # 모든 설정값의 단일 출처 — 튜닝은 여기서만
├─ models/weights/          # hand_landmarker.task(제스처) + 구 ONNX(납품 금지) + trt_cache(★PC 전용)
├─ src/
│   ├─ capture/camera_stream.py      # USB 카메라 캡처 스레드 (윈도우 MSMF 기본)
│   ├─ inference/detector_mediapipe.py  # 제스처 검출 기본 엔진 (MediaPipe — Apache-2.0)
│   ├─ inference/detector.py         # 구 ONNX 엔진(HaGRID YOLOv10 — 납품 금지) + 엔진 팩토리
│   ├─ inference/pose_estimator.py   # 사람 포즈 (rtmlib RTMPose — Apache-2.0)
│   ├─ postprocess/person_lock.py    # 사용자 잠금 + 손 귀속 (거울 좌/우 보정)
│   ├─ postprocess/gesture_filter.py # 동작 판정 FSM — 신규 스펙 + 레거시 토글
│   ├─ ocr/idcard_reader.py          # 주민등록증 이름·주민번호 (마스킹 로그)
│   ├─ announce/announcer.py         # 토크백 TTS (pyttsx3 — SAPI/nsss)
│   ├─ pipeline/realtime_loop.py     # 실시간 루프 조립 (멀티스레딩)
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce·/ocr 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test · export_onnx(개발용)
├─ tests/                   # 단위 테스트 47건 (카메라·모델 없이 실행 가능)
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
| `python -m unittest discover tests -v` | 판정·잠금·OCR 파싱·좌표 변환 단위 테스트 (47건) |

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
- **라이선스 C안 적용(2026-07-10)**: 제스처 엔진을 HaGRID YOLOv10 → **MediaPipe(Apache-2.0)로 교체**.
  YOLOv10은 AGPL-3.0(ultralytics 계열) 학습·변환 가중치라 ultralytics의 공식 해석상
  비공개 상업 사용 시 전체 코드 공개 또는 유료 라이선스가 요구되어 제외했다.
  현재 스택 전체: MediaPipe(Apache-2.0) · rtmlib/RTMPose(Apache-2.0) · ONNX Runtime(MIT) ·
  EasyOCR(Apache-2.0) · pyttsx3(MPL-2.0 — 무수정 사용이라 공개 의무 없음) —
  **카피레프트·저작자 표시 의무 없음**. ⚠ 구 ONNX 경로(`gesture_engine: onnx`)와
  models/weights/YOLOv10n_gestures.onnx는 비교 시험용 잔존 — **납품 빌드에서 제거할 것**
  (쓸 경우 NOTICE_HaGRID.md 고지 + AGPL 리스크). 최종 법무 확인은 기획서 9장 №9

## 참고 링크

- MediaPipe Hand Landmarker (Apache-2.0): https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- rtmlib (RTMPose, Apache-2.0): https://github.com/Tau-J/rtmlib
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- (기록) HaGRID — 구 제스처 모델, C안에서 제외: https://github.com/hukenovs/hagrid
