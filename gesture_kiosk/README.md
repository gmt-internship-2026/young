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

## 인식 동작 (2026-07-23 전면 개편 스펙 — 손 모양(point/fist) + 이동 방향으로 확정.
이벤트 이름은 「제스처 정의 보고서」 회사 확정 7종 고정 명칭 — 델파이 파싱 코드와
문자열이 정확히 일치해야 하므로 임의 변경 금지)

| action | 동작 | 판정 방식 | 키오스크 명령 |
|---|---|---|---|
| left / right / top / bottom | **검지 1개만 편 채(point)** 손을 좌/우/상/하로 이동 | MediaPipe 손 랜드마크(손목) 궤적 (window 내 이동량·주축 우세) | 포커스 이동 4방향 |
| ok | **주먹(fist)을 낸 채** 손을 우측으로 이동 | 〃 | 선택·확인 |
| back | **주먹을 낸 채** 손을 좌측으로 이동 | 〃 | 이전 화면 |
| home | **주먹을 낸 채** 손을 상단으로 이동 | 〃 | 처음 화면으로 |

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
- 엔진은 방향 판정만 하고, 그리드 열 수 기준 줄 단위 이동 및 줄 끝→다음 줄 첫 칸 랩
  (토크백식 선형 순회)은 **UI 책임** — `demo_ui/index.html`의 `focusCols()`가 화면별
  열 수(홈 3열 그리드, 그 외 1줄)만큼 top/bottom 시 포커스를 이동시킨다
- 방향(left/right/top/bottom) 판정은 기본이 임계값(min_dist_*_shoulder/axis_dominance)
  비교다 — 2026-07-24 GMtech_project(feat/think_win_cpu) 이식: 임계값 단위를 화면 비율에서
  **어깨너비 배수**로 교체해 카메라 거리와 무관하게 같은 동작이 같은 결과를 내고,
  전체 창 이동량이 부족해도 최근 짧은 구간의 단호한 움직임(플릭)이면 확정하는 경로 B,
  들어올리기 예비 동작 오발 방지(들어올리기 게이트), 짧은 신호 소실을 견디는 유예도
  함께 들어왔다(자세한 설계는 `src/postprocess/gesture_filter.py` 모듈 docstring 참고).
  같은 팀원이 델파이7 실기로 재확인한 결과 point 모양의 상/하 이벤트명도 up/down이
  아니라 top/bottom이 맞아 함께 교체했다(주먹 쪽 up→home 매핑은 원래 다른 이름이라 무관).
  이 임계값 방식으로도 오판정이 계속되면 학습된 분류기로 대체할 수 있다 —
  `collect_direction.bat`으로 궤적 데이터를 모으고 `train_direction.bat`으로 학습한 뒤
  `configs/config.yaml`의 `gestures.hand_move.classifier_weights_path` 주석을 해제하면
  적용된다(hand_shape의 `classifier_weights_path`와 같은 패턴 —
  `src/postprocess/direction_classifier.py`. 2026-07-24 시점 실기 정확도는 아직 임계값
  방식보다 낫다고 확인되지 않았다 — 데이터를 더 모아 재검증 필요)
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
│   ├─ pipeline/event_sender.py      # ★ 회사 프로그램 연동 접점 (console/udp/stdio — 실연동은 stdio)
│   └─ pipeline/demo_server.py       # ★ 예시 UI 서버 + /announce 계약
├─ scripts/                 # run_demo · download_weights · benchmark · smoke_test ·
│                           #   collect_hand_shape_data · train_hand_shape_classifier ·
│                           #   collect_direction_data · train_direction_classifier
├─ collect_hand_shape.bat / train_hand_shape.bat  # 손 모양 학습 분류기 수집·학습
├─ collect_direction.bat / train_direction.bat    # 방향(좌/우/상/하) 학습 분류기 수집·학습
│                                                 # (2026-07-24 도입 — 기본은 꺼짐, 아래 참고)
├─ delphi_ui/               # 델파이7 수신 데모(참고용) — GMtech_project에서 이식
├─ tests/                   # 단위 테스트 106건 (카메라·모델 없이 실행 가능)
├─ demo_ui/index.html       # ★ 예시 민원발급기 화면 (회사 UI 수령 시 교체)
├─ docs/TODO.md             # 작업 분해 및 회사 확인 필요 항목 (자세한 기술 기록)
└─ docs/작업일지.md          # 날짜별 작업 요약 (사람이 훑어보기 좋은 버전)
```

★ 표시는 **회사 키오스크 프로그램을 받으면 교체/제거되는 부분** (기획서 1.2, 9장 №7·№8).

## 실행 모드

| 명령 | 용도 |
|---|---|
| `run.bat` / `python scripts/run_demo.py` | 파이프라인 + 예시 UI (시연용) |
| `run.bat --headless` | 파이프라인만 — 이벤트는 `event_output` 설정대로 전송 |
| `python scripts/benchmark.py` | 추론 단독 FPS 측정 (기획서 6.1 — KPI 30 FPS) |
| `python -m unittest discover tests -v` | 판정·잠금·이벤트 전송 단위 테스트 (92건) |

## 회사 프로그램(델파이7) 연동 계약

1. **실연동: 파이프(stdio)**(2026-07-23 회사 확정 — "엔진은 이벤트를 print만 하면
   된다") — 델파이7이 이 엔진을 **자식 프로세스로 직접 실행**하고, 엔진이 stdout에
   찍는 텍스트 한 줄을 익명 파이프로 읽는다(`event_output.mode: stdio`,
   `event_sender.py`의 `StdioEventSender`). 네트워크(UDP·웹소켓)는 전면 철회됐다.
2. 전송 규격: `GESTURE|이벤트|손|신뢰도|시각\r\n` 한 줄(ASCII). 이벤트명은 7개 고정
   — `left`/`right`/`top`/`bottom`/`back`/`home`/`ok`(2026-07-24 up/down→top/bottom
   교체 — GMtech_project 팀원이 델파이7 실기로 재확인한 실제 프로토콜). "손" 필드는
   이 엔진이 손 좌/우
   정체성을 구분하지 않아 항상 빈 문자열이다(현재 델파이 파싱도 이 필드는 안 씀).
   델파이 쪽 수신 예제(CreateProcess + ReadFile, Delphi 소스 포함)는 자매 코드베이스
   `GMtech_project/gesture_kiosk`의 `delphi_ui/`·`docs/델파이7_연동가이드.md` 참고.
3. **로그는 stderr·파일로만 나가야 한다** — stdout은 이벤트 전용 채널. `logger.py`의
   `StreamHandler()`는 기본이 stderr라 별도 조치 없이 이미 안전하다.
4. Delphi가 실행할 명령은 `cmd /c run.bat --headless`(반드시 `--headless`) — 인자
   없이 `run.bat`만 실행하면 브라우저 데모 서버(uvicorn)까지 같이 뜬다(불필요).
5. 개발·디버깅용으로 `console`/`udp`도 남아있다(`/data` 폴링으로 JSON 확인 가능)
6. 새 수신 규격이 필요해지면 `event_sender.py`에 Sender 1개만 추가 — 파이프라인 수정 불필요
7. 음성 안내(UI→엔진): `POST /announce {"text": "발급하기 버튼"}` — 포커스 항목을 TTS로
   (참고: GMtech_project는 2026-07-22 TTS를 엔진에서 전면 제거하고 UI가 담당하도록
   바꿨다 — 이 저장소는 아직 엔진 TTS를 유지 중, 필요시 재검토)
8. 연동 완료 후 `demo_server.py`·`demo_ui/`는 제거

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
