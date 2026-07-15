# CLAUDE.md — gesture_kiosk 코딩 컨벤션

기준 문서는 **기획서 4장(개발 표준)** 과 **PEP 8**이다. 이 파일은 그 요약본 —
충돌하면 기획서가 우선한다. 모든 브랜치(판)에 동일하게 적용된다.
(형식 참고: [andrej-karpathy-skills/CLAUDE.md](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md))

## 1. 원칙

- **설정값은 `configs/config.yaml` 한 곳에서만 읽는다.** 코드에 숫자 하드코딩 금지.
- **하나의 개념에는 하나의 이름만 쓴다.** 영상 한 장은 어디서나 `frame`이다 (3장 용어 사전).
- **한 함수는 한 가지 일만 한다.** 30줄이 넘으면 분리를 검토한다.
- **`print()` 금지.** `src/utils/logger.py`의 `get_logger()`를 쓴다.
- **고치라는 곳만 고친다.** 요청과 무관한 리팩토링·스타일 변경을 섞지 않는다
  (브랜치가 8개라 diff가 오염되면 판별 비교가 무너진다).
- **실측과 이유는 주석으로 남긴다.** 날짜와 함께 — 6장 참고.
- 무거운 의존(rtmlib·easyocr 등)은 **함수/생성자 안에서 지연 임포트**한다 —
  해당 기능을 끈 환경에서 임포트 비용·설치 부담을 지지 않기 위해서다.

## 2. 네이밍 (기획서 4.2)

| 대상 | 규칙 | 예시 |
|---|---|---|
| 변수 · 함수 | snake_case | `frame_count`, `load_model()` |
| 클래스 | PascalCase | `GestureDetector`, `CameraStream` |
| 상수 | UPPER_SNAKE_CASE | `CONF_THRESHOLD`, `TARGET_FPS` |
| 모듈(파일) | snake_case.py | `gesture_filter.py` |
| 패키지(폴더) | snake_case | `postprocess/` |
| 내부 전용 멤버 | 밑줄 1개 접두어 | `_frame_buffer` |
| 불리언 | `is_` / `has_` 접두어 | `is_running`, `has_detection` |
| 컬렉션 | 복수형 | `detections`, `class_names` |

## 3. 도메인 용어 사전 (기획서 4.3) — 표준 표기만 쓴다

| 개념 | 표준 | 금지(혼용) |
|---|---|---|
| 영상 한 장 | `frame` | img, image, pic |
| 제스처 | `gesture` | motion, action, hand_sign |
| 신뢰도 | `conf` | confidence_score, prob, score |
| 경계 상자 | `bbox` | box, rect, bounding_box |
| 클래스 번호 / 이름 | `class_id` / `class_name` | cls, label_id / label_name |
| 임계값 | `…_threshold` | thres, th, limit |
| 모델 실행 | `infer` | predict, detect, run_model |
| 전 / 후처리 | `preprocess` / `postprocess` | pre_process, post_proc |
| 가중치 파일 | `weights` | ckpt, model_file |
| TensorRT 산출물 | `engine` | trt_model, rt_file |

## 4. 함수 이름 (기획서 4.4) — 「동사_목적어」, 동사는 이 목록에서

`load_`/`save_`(파일·모델) · `init_`(초기화) · `get_`/`set_`(조회/설정) ·
`capture_`(프레임 획득) · `preprocess_`/`postprocess_`(가공) · `infer`/`run_`(추론/루프) ·
`filter_`(거르기) · `draw_`(디버그 시각화) · `measure_`/`eval_`(측정/평가) · `is_`/`has_`(참/거짓)

## 5. 단위 접미사 (기획서 4.5) — 숫자 변수는 단위를 이름에 붙인다

`_ms`/`_sec`(시간) · `_px`(픽셀) · `_fps`(초당 프레임) · `_idx`/`_count`(인덱스/개수) ·
`_path`/`_dir`(경로) · `_ratio`(0.0~1.0 비율)

예: `elapsed_ms`, `frame_width_px`, `cooldown_sec`, `overlap_ratio` — 초↔밀리초 착오를 원천 차단.

## 6. 주석 · 독스트링

- **한국어로 쓴다.** 모듈 첫머리 독스트링에 목적과 큰 변경 이력(날짜)을 남긴다.
- 주석은 코드가 보여주지 못하는 **"왜"** 만 적는다 — 제약, 트레이드오프, 실측 근거.
  다음 줄이 무엇을 하는지 번역하는 주석은 쓰지 않는다.
- 실측 기반 결정은 날짜와 함께: 예) `max_infer_fps: 30 — 60이면 캡처가 30→15 FPS로
  반토막난다 (2026-07-10 실측)`.
- 회사 확인이 필요한 항목은 `docs/TODO.md`의 №번호를 주석에 단다.

## 7. 포매터 · 에디터 설정

- PEP 8 기본: **들여쓰기 4칸(스페이스)**, 한 줄 **100자 안팎**, 파일 인코딩 **UTF-8**.
- 자동 포매터(black 등)는 **아직 도입하지 않았다.** 도입할 때는 브랜치 간 diff 오염을
  막기 위해 **전 브랜치에 같은 커밋으로 동시 적용**한다 — 그 전까지 수동 준수.
- VS Code 사용 시 권장 설정(.vscode/settings.json — 커밋하지 않음):

```json
{
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "editor.rulers": [100],
  "files.encoding": "utf8",
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true
}
```

## 8. 테스트 · 커밋

- `tests/`는 **카메라·모델 없이** 돌아가야 한다: `python -m unittest discover tests`
  (판정·잠금·OCR 파싱 로직만 검증 — 현재 65건).
- 동작 검증 순서: 단위 테스트 → `scripts/smoke_test.py` → 실기 `scripts/benchmark.py`.
- 커밋 메시지: `feat: YYYY-MM-DD 요약` 형식, 한국어. 무엇을 왜 바꿨는지 본문에 남긴다.
- **브랜치 간 merge 금지.** 다른 판의 수정을 가져올 때는
  `git checkout <브랜치> -- <파일>` 후 자체 커밋한다 (판별 독립 관리).
