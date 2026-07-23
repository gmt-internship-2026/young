# TODO — 작업 분해 및 진행 상황

작성: 2026-07-08 · 개정: 2026-07-16 (선택 동작 재확정 — 손가락 인식 / 주민등록증 인식
기능 제거 / TTS 상황별 안내 보강 / CPU 성능 실측·쓸기 임계값 조정).
기획서(기획서.docx) 주차 계획·9장 체크리스트와 연동. **기획서 2.2/3.1/5.1은
2026-07-10(타깃)·2026-07-15(동작 체계)·2026-07-16(선택 재확정·OCR 제거·TTS 보강) 변경을
반영해 개정 필요.**

## ✅ 완료 (2026-07-23 2차 — 델파이7 실연동: 네임드 파이프 프로토콜)

- [x] **전송 방식 팀 확정 — 네임드 파이프.** 델파이7이 파이프 **서버**로 열고, 이 엔진이
      **클라이언트**로 접속(WebSocket 검토안과 같은 방향 — Delphi가 서버, 엔진이 클라이언트,
      GMtech_project 참고). JSON 페이로드가 아니라 **평문 명령어 7개 고정**을 개행 구분
      한 줄로 전송한다: `up`/`down`/`left`/`right`/`home`/`back`/`ok` — 델파이 쪽은 파이프에
      들어온 문자열을 그대로 인식해 실행(팀장 확인, 별도 파싱 코드 불필요)
- [x] `src/pipeline/event_sender.py`에 `PipeEventSender` 추가 — `PIPE_COMMAND_BY_CLASS_NAME`
      (우리 7개 제스처 클래스 = 프로토콜 7개 토큰과 1:1 대응). 파이썬 표준 `open()`으로
      윈도우 네임드 파이프 경로(`\\.\pipe\이름`)를 여는 방식이라 pywin32 등 추가 의존성
      없음. 연결 끊김 시 `reconnect_backoff_sec` 간격으로만 재접속 시도 —
      `WebSocketEventSender`(GMtech_project)와 같은 패턴을 그대로 이식
- [x] `configs/config.yaml`/`config_mac.yaml`에 `event_output.pipe`(name·
      reconnect_backoff_sec) 신설. `event_output.mode`는 여전히 `console` 기본값 —
      `pipe.name`이 placeholder(`\\.\pipe\GestureKiosk`)라 실제 이름 확정 전까지는
      전환하지 않는다(WebSocket 때와 같은 원칙)
- [x] `tests/test_event_sender.py` 신규(이 저장소에 원래 없던 파일 — GMtech_project
      패턴 이식) — open_fn·clock 주입으로 연결/재접속 backoff/전송 실패 경로 검증,
      7개 클래스 전부가 서로 다른 토큰에 매핑되는지 확인. 전체 79건 통과
- [ ] **★TODO(팀 확인 필요)** — `PipeEventSender`가 파이프를 `"r+b"`(duplex 가정)로
      여는데, 델파이 쪽 `CreateNamedPipe`가 인바운드 전용(`PIPE_ACCESS_INBOUND`)이면
      `"wb"`로 바꿔야 열린다. 델파이7 개발자에게 파이프 접근 모드 확인 필요
- [ ] **실기 확인 필요** — 실제 델파이7 파이프 서버 기동 후 연결·명령어 수신·재접속
      시나리오(델파이 재시작 등) 검증. 파이프 이름 확정되는 대로 `mode: pipe`로 전환

## ✅ 완료 (2026-07-23 — 동작 체계 전면 개편: 손 모양+이동 통합)

- [x] **동작 스펙 변경(사용자 확정, 회사 정식 확인은 아직 — №1과 함께 처리)** —
      point(검지 1개, "가리키기") + 좌/우/상/하 이동 = move_left/move_right/move_up/
      move_down(포커스 이동 4방향, 신규 확장). fist(주먹) + 좌/우/상 이동 = go_back/
      select/go_home(이전/확인/홈). fist+아래는 미정의(사용자 확정 — 매핑 없음, 방향은
      감지되나 이벤트 없음). 옛 체계(팔 쓸기=좌/우 이동, 손 위치 이동=화면 전환, 손가락
      1개 정지 유지=선택)는 전면 폐기
- [x] **손 모양·이동 판정을 한 곳(MediaPipe 손 랜드마크)으로 통합** — 옛 팔 쓸기는
      RTMPose 포즈 손목 궤적, 화면 전환은 MediaPipe 손 위치로 출처가 달라 손 모양을
      이동 판정에 반영할 수 없었다. 이제 손 모양(gestures.shapes)과 이동(gestures.
      hand_move)이 같은 프레임에서 나와 항상 동기화된다
- [x] `src/postprocess/gesture_filter.py` 전면 재작성 — `_SwipeTracker`는 그대로 재사용
      (판정 로직 검증됨), point/fist 전용 트래커 2개로 분리(모양 전환 시 서로 리셋해
      다른 모양의 이동량이 섞이지 않게 함). `_FingerSelectTracker`(정지 유지 판정)는
      더 이상 필요 없어 제거
- [x] `src/postprocess/person_lock.py`에서 `user_swipe_points()`·`user_side_points()`·
      `KPT_LEFT/RIGHT_WRIST`·`KPT_LEFT/RIGHT_ELBOW` 제거 — 포즈 손목/팔꿈치 궤적 기반
      쓸기 추적점 공급을 아무도 안 쓰게 됐다. 이 클래스는 이제 얼굴 기반 사용자 잠금·
      bbox 산출만 담당(포즈 모델은 여전히 필요 — 잠금용 얼굴 검출)
- [x] `configs/config.yaml`/`config_mac.yaml` — `gestures.swipe`·`gestures.hand_swipe`·
      `gestures.select` 3섹션을 `gestures.hand_move`(이동 판정 1개, 4방향 공용)·
      `gestures.shapes`(손 모양 판정)로 통폐합. `classes`에 move_up/move_down 신설,
      `announce.event_templates`에 문구 추가
- [x] `GestureEvent.hand_side`(옛 쓸기 팔 좌/우) → `GestureEvent.shape`(point/fist)로
      필드명·의미 변경 — `event_sender.py`·`demo_server.py`·`demo_ui/index.html` 함께 수정
- [x] 단위 테스트 재작성 — `test_gesture_filter.py`(PointMoveTest·FistMoveTest·
      ShapeSwitchTest 등 신규), `test_person_lock.py`(SwipePointTest 삭제, 쓸기 관련
      단언 제거)
- [ ] **★잠정치 주의** — `hand_move.min_dist_x_ratio`(0.18)는 이번 개편으로 처음
      "손 위치만으로 좌/우"를 판정하게 되며 새로 정한 값(옛 팔 쓸기 임계 0.25는 팔
      전체 스케일이라 못 씀, 옛 hand_swipe의 0.4는 좌/우를 애초에 비활성화하려던
      값이라 역시 못 씀) — 실기 재튜닝 전 가정값. 계기판 POINT/FIST x 진행도로
      확인할 것
- [ ] **실기 확인 필요** — fist(주먹) 상태에서 손가락 인식(count_extended_fingers)이
      point(검지 1개)만큼 안정적으로 0으로 잡히는지(주먹도 오검출 가능성 있음),
      실제 사용자가 "내밀고 이동"하는 동작에서 hand_move 임계값들이 적절한지

## ✅ 완료 (2026-07-16 — CPU 추론 성능 최적화: 잠금 중 검출 스킵)

- **문제 실측(이 개발 PC — AMD Ryzen 5 3550H 4C/8T, 배포 기준과 동일한 윈도우+
  Python 3.11.5)**: `scripts/benchmark.py`로 잰 포즈 추론 단독 FPS가 **평균
  4~7 FPS**(목표 30 FPS의 15~25%, 프레임당 137~325ms) — §5·§6 KPI(30 FPS) 관련
  №5·№6 회사 협의 시 이 실측치를 공유할 것
- **원인 프로파일링**: rtmlib `Body`(lightweight)의 검출기(YOLOX-tiny, 416×416
  고정)만 단독으로 **~89ms/프레임**(사람이 0명이어도 고정 비용) — 이게 병목의
  대부분. onnxruntime 스레드는 이미 자동(전체 코어)이라 더 짤 여지 없고
  oneDNN/OpenVINO 가속 프로바이더도 미설치
- **기각한 대안(실측 후 폐기)**: ① `RTMO`(rtmlib 내장 단일모델, 검출+포즈 통합) —
  기대와 달리 **3.06 FPS로 더 느렸음**. ② `det_input_size` 축소(416→320/256) —
  YOLOX-tiny ONNX가 고정 입력 shape라 런타임에서 못 바꿈(즉시 shape 에러).
  ③ `model.input_size_px` 설정을 실제로 연결 — 애초에 죽은 설정값이었고
  (`preprocessor.py`/`pose_estimator.py` 어디서도 안 읽음), rtmlib가 항상 고정
  크기로 리사이즈해 캡처 해상도를 낮춰도 속도 차이가 거의 없었음. **README·
  설치가이드가 안내해 온 "30 FPS 미달 시 input_size_px 640→480" 1차 대응법은
  실제로는 통하지 않는다** — 문서 정정 필요(별도 항목)
- **채택한 해결책**: `RTMPose`에 **이미 아는 bbox**를 직접 넣으면(검출 생략)
  **~22ms/프레임**(46 FPS 페이스) — 검출기가 압도적 병목이었으므로, 사용자를
  이미 잠근 뒤에는 매 프레임 재검출하지 않고 이전 위치로 포즈만 재추정,
  `redetect_interval_frames`(기본 10)마다 한 번만 전체 검출로 재동기화하는
  패스트패스를 넣음:
  - `src/inference/pose_estimator.py` — `infer_at_bbox(frame, bbox)`(검출 생략)·
    `pad_bbox(bbox, pad_ratio, w, h)`(순수 함수, 프레임 경계 clamp) 신규.
    `infer()`/`infer_at_bbox()`가 키포인트→`PersonPose` 변환 로직을 공유하도록
    `_persons_from_keypoints()`로 분리
  - `src/pipeline/realtime_loop.py` — `should_refresh(is_active,
    frames_since_refresh, interval)`(순수 함수)로 이번 프레임에 풀 검출을 쓸지
    결정. `person_lock.py`는 변경 없음(1개짜리 후보 리스트도 기존
    `_follow_locked()`가 그대로 처리)
  - `configs/config.yaml`/`config_mac.yaml`의 `model`에 `redetect_interval_frames:
    10`·`bbox_pad_ratio: 0.3` 추가
  - `tests/test_pose_estimator.py`(`pad_bbox`)·`tests/test_realtime_loop.py`
    (`should_refresh`) 신규
- **결과 실측(이 CPU, 잠금 상태 시뮬레이션 — 더미 프레임 60장)**: **30.86 FPS**
  (32.4ms/프레임 평균) — 목표 30 FPS 달성. 개선 전(4~7 FPS) 대비 4~7배
- **한계**: 재동기화 사이(`redetect_interval_frames`=10프레임) 사람이 패딩
  범위(`bbox_pad_ratio`=0.3) 밖으로 빠르게 움직이면 놓칠 수 있음 — 실제 카메라·
  사람으로는 검증 못 했음(이 환경엔 카메라 없음). 아래 실기 검증 절 참고
- [ ] README·설치가이드의 "input_size_px 640→480" 1차 대응법 문구를 이번 실측
      결과로 정정(죽은 설정 + 효과 없음 — 대신 잠금 후 자동 최적화된다고 안내)

### (같은 날 2차) 손가락 인식(hand_estimator)도 새 병목으로 드러나 같이 최적화

- 위 포즈 최적화 직후 전체 루프(포즈+손 인식)를 실측하니 **12.85 FPS로 오히려
  낮았음** — 손 인식(MediaPipe) 단독이 ~32ms/프레임이라, 병목이 검출기에서
  손 인식으로 옮겨간 것뿐이었다
- 포즈와 같은 철학 적용 — `should_refresh()`를 그대로 재사용(이름을
  `should_redetect`→`should_refresh`로 일반화): 손가락도 매 프레임이 아니라
  `hand_model.infer_interval_frames`(기본 3)마다만 재인식, 그 사이는
  `last_finger_count` 캐시를 재사용한다. **주의**: 건너뛴 프레임에 `None`을
  넣으면 `_FingerSelectTracker`의 `hold_sec`(0.6초) 유지 타이머가 매번
  리셋되므로, 반드시 마지막 값을 그대로 재사용해야 한다(`realtime_loop.py`에
  주석으로 명시). 사용자 잠금 해제 시에는 캐시를 버림(이전 사용자의 손가락
  상태가 다음 사용자에게 새는 것 방지)
- **결과 실측(포즈+손 인식 모두 최적화 적용, 더미 프레임 90장)**: **23.05 FPS**
  (43.4ms/프레임) — 손만 최적화하기 전(12.85 FPS) 대비 약 1.8배. 30 FPS까지는
  `hand_model.infer_interval_frames`를 더 늘려야 하지만(~16 이상 필요), 그러면
  hold_sec 0.6초 안에 손가락 샘플이 너무 적게 잡혀 정상적인 선택 동작을 놓칠
  위험이 커진다고 판단해 **정확성과 성능의 균형점으로 3을 채택** — 순수 FPS
  숫자보다 "선택이 실제로 잘 인식되는지"가 우선
- [ ] `infer_interval_frames`(현재 3) 실기에서 더 키워도 select가 안정적으로
      잡히는지 확인 — 여유가 있으면 성능을 더 끌어올릴 수 있음
- [x] **(별도 완화 조치)** `gestures.swipe`가 30 FPS 가정으로 튜닝돼 있었음(`window_sec`
      0.6초·`min_track_frames` 4 → 4/0.6≈6.7 FPS 필요, 개선 전 4~7 FPS로는 시간창
      안에 최소 프레임이 잘 안 채워져 **쓸기가 거의 확정되지 않는 증상**으로 이어짐
      (사용자 리포트: "좌우 스윕이 잘 안 먹힘"). `window_sec`→1.0초, `min_track_frames`
      →3, `cooldown_sec`→1.2초(윈도우보다 길게 유지하는 기존 원칙 유지)로 조정.
      이제 잠금 후 30 FPS 근접이라 되돌려도 되지만, 재동기화 프레임(10프레임마다)
      순간 지연이 섞이므로 여유값 그대로 유지 — 실기 확인 후 재검토

## ✅ 완료 (2026-07-16 — TTS 상황별 안내 보강)

- [x] **사용자 인식/해제 음성 안내 신설** — `person_lock` 잠금·해제 전환 시
      `Announcer.on_user_lock_change()`로 1회 안내(`config announce.status_templates`).
      기존엔 화면 배지(`is_user_locked`)만 갱신하고 소리가 전혀 없었다 — 화면을
      못 보는 사용자가 카메라에 인식됐는지 알 수 있는 유일한 수단이라 추가.
      `person_lock`의 `lock_frame_count`(3프레임)·`release_sec`(2초)가 이미
      디바운스 역할을 해 별도 쿨다운 없이 전환 시점에만 1회 안내됨
- [x] **TTS 보이스 자동 선택** — `_init_tts_engine()`이 pyttsx3 기본 보이스를
      그대로 쓰던 것을, `config announce.voice_keyword`와 이름/id가 일치하는
      보이스를 찾아 명시 지정하도록 변경(`select_voice`). 실제 배포 PC에 여러
      보이스가 깔려 있어도 한국어로 읽히게 보장 — 못 찾으면 경고 로그 후 기본
      보이스로 폴백(치명 오류 아님). 윈도우는 `"Korean"`, 맥은 `"ko"`(nsss 보이스는
      이름 대신 id에 로캘 코드가 들어가는 경우가 많음) 키워드로 분리
- [x] 순수 함수(`select_voice`·`status_announcement`)로 분리해 pyttsx3·오디오
      없이 단위 테스트 — `tests/test_announcer.py` 11건 신규. 전체 46건 통과
- [x] **실기 확인 완료** — 이 개발 PC가 실제 윈도우 + Python 3.11.5라 바로 검증됨.
      설치된 SAPI 보이스: `Microsoft Heami Desktop - Korean`(한국어)·
      `Microsoft Zira Desktop - English`(영어) 2개 — `voice_keyword: "Korean"`이
      Heami를 정확히 선택함을 `engine.getProperty("voice")`로 실측 확인

## ✅ 완료 (2026-07-16 — 선택 동작 재확정: 손가락 인식)

- [x] **선택(OK) = 손가락 1개(엄지 제외) 인식으로 교체** — 무손·무지 사용자 접근성
      요건이 계획에서 빠지면서(회사 확인 필요, №1 갱신), 꾸벅임은 UX 부담이 크다는
      판단으로 기각. `_NodTracker`·목 길이 신호(person_lock.user_neck_ratio)는 전면
      제거하고 `_FingerSelectTracker`(hold_sec 유지 판정)로 대체
- [x] 손 랜드마크 엔진으로 MediaPipe HandLandmarker(Tasks API, Apache-2.0) 재도입 —
      rtmlib Hand/Wholebody(mmpose 2단계: 검출+포즈)보다 CPU 실시간 손 추적에 강점.
      모델(hand_landmarker.task)은 2026-07-15 제거 때 남아있던 파일을 재사용해
      재다운로드 불필요. person_lock이 잠근 사용자의 bbox로 크롭한 영역만 추론해
      CPU 30 FPS 예산·다른 사람 손 오인식을 방지
- [x] 단위 테스트 갱신 — 꾸벅 관련 테스트(NodSelectTest·UserNeckRatioTest) 삭제,
      손가락 판정 테스트(FingerSelectTest·test_hand_estimator.py) 추가. 전체 47건 통과

## ✅ 완료 (2026-07-16 — 주민등록증 인식(OCR) 기능 전면 제거)

- [x] **계획 변경으로 OCR(EasyOCR) 기능 전체 삭제** — `src/ocr/`, `tests/test_idcard_parse.py`,
      `training/idcard/` 디렉터리 삭제. 파이프라인의 OCR 워커 스레드(`_start_ocr_worker`)·
      `PipelineState` OCR 상태·`demo_server.py`의 `/ocr/start`·`/ocr/stop` 엔드포인트·
      `fill_id_fields` 이벤트(config `classes`·`announce.event_templates`)를 전면 제거
- [x] `requirements.txt`에서 `torch`·`torchvision`·`easyocr` 삭제 (EasyOCR 전용 의존성 —
      설치 용량 ~2GB 절감). `pose_estimator.ensure_cuda_dlls()`의 `import torch`는
      원래 `try/except ImportError`로 감싸져 있어 영향 없음(GPU 가속 시도만 건너뜀)
- [x] **(부수 발견) opencv 패키지 충돌 실측** — rtmlib는 `opencv-python`, mediapipe는
      `opencv-contrib-python`을 각각 요구해 둘 다 설치된다. 같은 `cv2` 경로를 공유해
      한쪽만 별도로 설치/제거하면 파일이 뒤섞여 깨진다(`cv2.dnn` 소실 실측) — 두 패키지를
      항상 같은 버전으로 함께 설치하도록 `requirements.txt`에 명시
- [x] `demo_ui/index.html`의 예시 흐름을 3단계(증명서 선택→본인확인→발급완료)에서
      2단계(증명서 선택→발급완료)로 축소 — 본인확인 화면·OCR 배지·토스트 알림 제거.
      겸사겸사 이전 세션에서 놓친 "고개 꾸벅 2회" 잔여 문구도 "손가락 1개 들기"로 정정
- [x] `install.bat`·`make_offline_bundle.bat`의 EasyOCR 한국어 모델 다운로드/복사 단계 삭제
- [x] 단위 테스트 47건 → OCR 12건 제거 → **35건** 통과

## ✅ 완료 (2026-07-15 — 동작 체계 범용 설계 개편, 같은 날 2차 확정)

- [x] 쓸기 동작 도입: 팔 좌/우 쓸기=포커스 이동 · 아래=이전 화면(**go_back 이벤트 신설**) ·
      위=처음으로 — 포즈 손목 궤적 판정이라 손·손가락이 없는 사용자도 동일 조작
- [x] **(2차) 선택(OK) = 고개 꾸벅 2회 확정** — "끄덕임=예"라 직관적이고 팔이 전혀 없어도
      가능. 신호 = 목 길이 비율(코~어깨 중점, 어깨 너비 정규화), 숙였다 제때(0.8초) 복귀 ×2.
      내려다보기(지갑·신분증)는 복귀 시간 초과로 무효 — 오탐 방지 설계
      **(2026-07-16 대체됨 — 무손·무지 접근성 요건 제외로 손가락 1개 인식으로 교체, 위 참고)**
- [x] (2차) 1차 안이던 손등/팔등 보이기 경로 전면 제거 — MediaPipe 손 검출·팔등 CNN·
      수집/학습 스크립트 삭제 → **포즈(RTMPose) 단일 엔진** (requirements에서 mediapipe 제거)
      **(2026-07-16 부분 되돌림 — 선택이 손가락 인식으로 바뀌며 MediaPipe 재도입,
      단 팔등 CNN·수집 스크립트는 그대로 제거 상태 유지)**
- [x] 구 동작 전면 제거: 주먹→펴기·OK핀치·양손바닥 10초·레거시(5.1 초안) 토글
- [x] 구 HaGRID ONNX 제스처 엔진 제거 — 신규 스펙 판정 불가로 사장 + AGPL 리스크 해소
      (제거 산출물 백업: ~/Documents/backup/gesture_kiosk_2026-07-15_before_gesture_rework/,
       _before_nod_select/)
- [x] 단위 테스트 43건 재작성·통과 (쓸기·꾸벅 판정·목 길이 신호·잠금)
- [x] **(2026-07-16) 쓸기 팔꿈치 폴백** — 손 절단 사용자는 손목 키포인트 신뢰도가
      낮을 수 있어(사용자 지적), 손목 미달 시 팔꿈치 궤적 + elbow_gain 배율로 판정.
      추적점 전환 시 궤적 리셋, 화면 라벨 "(E)" 표시. 테스트 48건

## ✅ 완료 (2026-07-08 — 1주차)

- [x] 기획서 2.3 디렉터리 구조 + 4장 코딩 컨벤션 적용한 저장소 골격
- [x] 캡스톤 코드 구조 이식 — 단일 USB 카메라·제스처용으로 정리
- [x] 실시간 파이프라인: 캡처 스레드 → 전처리 → YOLO 추론 → 판정 → 이벤트
- [x] 무학습 모델 경로 확보: HaGRIDv2 사전학습 YOLOv10n (다운로드 스크립트 포함)
- [x] 예시 UI · 이벤트 전송 접점(event_sender) · 단위 테스트

## ✅ 완료 (2026-07-10 — 전면 리팩토링, feat/think_win_gpu)

- [x] 타깃 확정: **윈도우 + NVIDIA + Python 3.11.5** (정부 민원발급기)
- [x] 신규 동작 체계: 주먹→펴기 이동(왼손=왼쪽·오른손=오른쪽) / OK=선택·확인 통일 /
      양 손바닥 10초=처음으로. 레거시(5.1 초안)는 config 토글로 병행 유지
- [x] 사용자 잠금: YOLO11n-pose + 얼굴 선명도(라플라시안)×크기 점수 —
      잠긴 사용자의 손목 근처 손만 인식 (다른 사람 손 차단, 거울 좌/우 보정)
- [x] 주민등록증 OCR(EasyOCR): 이름·주민번호 자동 입력(fill_id_fields) —
      프레임·원문 미저장, 로그 마스킹
      **(2026-07-16 전면 제거됨 — 계획 변경, 위 참고)**
- [x] 토크백: 이벤트 자동 안내 + UI 요청(POST /announce) TTS(pyttsx3)
- [x] 이식 체계: requirements 버전 고정 · install.bat / run.bat ·
      내부망용 make_offline_bundle.bat · 설치가이드.md · smoke_test
- [x] 예시 UI를 민원발급기 흐름(증명서 선택→본인확인→발급)으로 교체
      **(2026-07-16 본인확인 단계 제거됨 — 증명서 선택→발급 2단계로 축소, 위 참고)**
- [x] 단위 테스트 43건 통과 (FSM·사용자 잠금·OCR 파싱)

## 🔴 회사 확인 필요 — 멋대로 진행 금지 (기획서 9장 연동)

- [ ] **№1 제스처 스펙 확정** — 2026-07-23 기준 스펙(point+이동 4방향 / fist+이동 3방향
      확인·이전·홈)으로 재구현(위 "동작 체계 전면 개편" 절 참고) — 사용자 확정으로 진행
      중이나 **회사에 정식 확인 필요**. 무손·무지 사용자 접근성 요건이 계획에서 빠졌다는
      전제도 함께 진행 중이라 재확인 필요 (빠지지 않는다면 대체 판정 재검토).
      직원 호출(help_call)은 트리거(양 손바닥)가 사라져 이벤트 계약에서 제외됨 —
      필요 시 대체 트리거(예: 위로 쓸기 길게) 회사 협의 필요
- [ ] **№7 회사 키오스크 프로그램(UI) 파일 수령** — 수령 후 demo_ui 교체 작업 시작
- [ ] **№7 이벤트 전달 규격 합의** — udp JSON 예시 구현 상태. 소켓/시리얼 등 확정 시 Sender 추가.
      **2026-07-15 계약 변경: classes에 go_back 신설·help_call 제외 — UI 측 반영 확인 필요**
- [ ] **№8 통합 범위 확정** — UI 통합까지인지 이벤트 출력까지인지
- [ ] **№5·№6 KPI 측정 기준 합의** — 정확도 85% 산식 / 30 FPS(엔드투엔드 vs 추론 단독)
- [ ] **№9 라이선스 검토** — 2026-07-16: 선택이 손가락 인식으로 바뀌며 MediaPipe
      (Apache-2.0)가 다시 추가돼 추론 모델이 rtmlib RTMPose + MediaPipe HandLandmarker
      2개. 둘 다 Apache-2.0이라 2026-07-10 확인된 "카피레프트·저작자표시 의무 없음"
      결론은 유지되지만, MediaPipe 추가분은 원문 재확인 필요(자체 CNN·HaGRID는 여전히 제거 상태).
      잔여: 배포물에 Apache/MIT 라이선스 문서(rtmlib + mediapipe) 동봉 절차 최종 빌드 때 확인
- [ ] **№10 코드 저장 위치·백업 정책** — 외부 Git 허용 여부
- [x] ~~**№11 (신규) 주민등록번호 처리 법적 근거**~~ — **해당 없음 (2026-07-16)**:
      주민등록증 인식(OCR) 기능을 계획에서 완전히 뺐다 — 더는 개인정보(주민등록번호)를
      처리하지 않아 이 항목의 법적 근거 확인 자체가 불필요해짐
- [ ] **№12 (신규) 설치 대상 PC 사양·망 환경** — GPU 모델(RTX 50시리즈?),
      인터넷/내부망 여부 → 내부망이면 오프라인 번들 반입 절차(설치가이드 B절) 협의

## 🟡 윈도우 실기기 검증 (설치 PC 확보 시 — 맥 개발기에서 불가한 항목)

- [ ] install.bat / run.bat / make_offline_bundle.bat 실기 동작 확인 (작성만 됨 — 미검증)
- [ ] onnxruntime 1.23.2 + rtmlib + mediapipe 설치 확인 (smoke_test)
- [x] ~~CPU 추론 benchmark — 30 FPS 충족 확인~~ — **완료**: 이 개발 PC에서
      개선 전 4~7 FPS(미달) → 검출 스킵 패스트패스 적용 후 더미 프레임 기준
      30.86 FPS(위 "CPU 추론 성능 최적화" 절 참고). 다른 CPU 사양 배포 PC에서는
      재측정 필요 — 이 결과가 모든 대상 PC를 대표하진 않음
- [ ] **(2026-07-16 신규)** 검출 스킵 패스트패스 실기 확인 — 이번 최적화는 더미
      프레임(고정 bbox 반복)으로만 속도를 쟀다. 실제 사람이 움직일 때
      `bbox_pad_ratio`(0.3)가 충분한지, `redetect_interval_frames`(10)마다
      순간적인 지연/끊김이 체감되는지, 빠르게 움직이면 추적을 놓치는지 확인
      후 필요시 값 조정
- [ ] 오토포커스 카메라로 person_lock 튜닝 (sharpness_weight·wrist_match_ratio)
- [ ] 한국어 TTS 보이스(Heami 등) 설치 확인 + 안내 문구 낭독 품질
- [ ] 두 사람 동시 프레임 진입 시 잠금 유지 확인 (다른 사람 손 차단)
- [ ] **(2026-07-16 신규)** 손가락 1개 인식 select 실기 확인 — 거리·조명·손 각도별 오탐/미탐.
      `hand_model.infer_interval_frames`(3)로 스로틀링 중이라 실제 손가락을 짧게
      들었다 내렸을 때도 놓치지 않고 잡히는지 특히 확인(위 "손가락 인식 최적화" 절 참고)
- [x] ~~TTS 한국어 보이스 자동 선택 실기 확인~~ — **완료** (위 2026-07-16 TTS
      절 참고, 이 개발 PC에서 Heami 보이스 선택 실측 확인됨)
- [ ] **(2026-07-16 신규)** 사용자 인식/해제 음성 안내 실기 확인 — 카메라 앞에
      서고 벗어날 때 "사용자가 인식되었습니다"/"인식이 해제되었습니다"가 매끄럽게
      들리는지(연속 프레임 반복 안내 없이 전환 시 1회만) — 이 항목은 카메라·
      person_lock 파이프라인 전체 구동이 필요해 별도로 남겨둔다

## 🟢 이후 (조건부 — 학습이 필요해질 때, feat/study)

- [ ] 제스처 정확도 미달 시: 데이터 수집·라벨링 → training/으로 파인튜닝 (기획서 5.4 인물 단위 분할)
- [ ] 학습 가중치 명명 규칙: {모델}_{데이터셋}_{지표}_{날짜}.pt (기획서 4.8)

## 메모

- 학습은 5080 / 맥북 M5 Pro 등 아무 환경 — 결과 .pt만 추론(윈도우)으로 이식 (2026-07-10)
- 예시 UI의 증명서 목록은 데모용 더미 — 실제 화면·업무 흐름은 회사 프로그램 담당
