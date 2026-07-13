# VS Code에서 gesture_kiosk 구동 가이드 (맥)

> 작성 2026-07-10. 대상: 맥 + VS Code. 처음 환경을 만드는 법은 [CODE_SETUP.md](CODE_SETUP.md),
> 맥에서 겪은 문제와 해결 기록은 [MAC_RUN_LOG.md](MAC_RUN_LOG.md) 참고.
> 이 문서는 **이미 만들어진 환경(venv·모델·설정)을 VS Code에서 실행하는 법**만 다룬다.

## 0. 한눈 요약

```
VS Code로 gesture_kiosk 폴더 열기 → F5 → "데모 실행 (맥 — localhost:5001)" → 브라우저 접속
```

나머지는 전부 자동이다. 아래는 각 단계의 확인 포인트와 문제 해결.

## 1. 폴더 열기

- VS Code 메뉴 → File → Open Folder → `~/Desktop/GMtech_project/GMtech_project/gesture_kiosk`
- **상위 폴더(GMtech_project)가 아니라 gesture_kiosk 를 열어야 한다** —
  `.vscode/` 설정(인터프리터·실행 구성)이 이 폴더 기준으로 저장돼 있다.
- 터미널에서 여는 법: `"/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" ~/Desktop/GMtech_project/GMtech_project/gesture_kiosk`

## 2. 인터프리터 확인 (자동)

`.vscode/settings.json` 이 파이썬 인터프리터를 `./venv`(**Python 3.10.20**)로 자동 지정한다. 확인만 하면 된다:

- VS Code 우측 하단 상태바에 `3.10.20 ('venv')` 표시가 있는지 확인
- 다르게 나오면: `Cmd+Shift+P` → "Python: Select Interpreter" → `./venv/bin/python` 선택

## 3. 실행 — F5 한 번

`Run and Debug`(왼쪽 ▶🐞 아이콘) 또는 **F5** → 상단 드롭다운에서 구성 선택:

| 구성 이름 | 하는 일 |
|---|---|
| **데모 실행 (맥 — localhost:5001)** | 카메라+추론+판정 파이프라인 + 예시 키오스크 UI 서버 |
| 단위 테스트 (판정 로직) | 카메라·모델·LLM 없이 판정 로직 24건 검증 |

실행 전 확인 2가지:

1. **다른 데모가 떠 있으면 먼저 종료** — 포트 5001은 하나만 쓸 수 있다.
   Terminal(`run_demo_mac.command`)로 띄운 게 있으면 그 창에서 `Ctrl+C`,
   또는 아무 터미널에서 `pkill -f run_demo.py`
2. **카메라 권한** — VS Code에서 처음 실행하면 macOS가
   "Visual Studio Code가 카메라에 접근하려고 합니다" 팝업을 띄운다 → **허용**.
   (팝업을 놓쳤으면 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 Visual Studio Code 켜기)

실행되면 VS Code 내장 터미널에 로그가 흐른다. 정상 기동 로그 순서:

```
[capture]     카메라 캡처 스레드 시작 (device_id=0)
[inference]   모델 로딩 완료: ... (backend=torch, device=mps)
[postprocess] ollama 웜업 완료: qwen3:4b        ← LLM 판정 준비 (아래 5번 참고)
[postprocess] LLM 판정 백엔드: ollama:qwen3:4b
[pipeline]    실시간 파이프라인 시작
INFO: Uvicorn running on http://0.0.0.0:5001
```

## 4. 브라우저 확인

- **http://localhost:5001** 접속 → FINOK 무인민원발급기 화면
- 음성안내가 기본 켬이지만, 크롬 정책상 **페이지를 한 번 클릭해야 소리가 나기 시작**할 수 있다
- 우하단 PIP(카메라 화면)에서 확인할 것:
  - 초록 박스 = 손 검출, **하늘색 원+십자 = 오토포커스 잠금 영역** (원 밖의 손은 무시)
  - 추론 FPS ≈ 60 (상한 고정), 카메라 FPS ≈ 30
- 카메라 없이 UI만 볼 때는 키보드로 흐름 점검: 방향키=포커스, `[` `]`=페이지,
  `{` `}`=2장, `s`=선택(OK), `c`=확인(따봉), `x`=취소, `m`=단축키, `h`=직원호출

## 5. LLM 판정(Ollama) — 꺼져 있어도 데모는 돈다

맥 설정(`configs/config_mac.yaml`)은 판정 백엔드가 `ollama`(로컬 LLM, qwen3:4b)다.

- Ollama는 `brew services start ollama` 로 백그라운드 서비스 등록돼 있어 보통 자동으로 떠 있다
- 확인: 터미널에서 `curl -s localhost:11434/api/version` 이 응답하면 정상
- **Ollama가 꺼져 있어도 데모는 정상 구동된다** — LLM 호출이 실패하면 경고 로그 후
  자동으로 규칙(rule) 판정으로 폴백된다. 로그에 `LLM 판정 실패 … 규칙 폴백` 이 보이면 이 상태다
- 아예 규칙 판정만 쓰려면: `configs/config_mac.yaml` → `judge.backend: rule`

## 6. 종료·재시작

- 종료: **Shift+F5** (디버그 중지) 또는 내장 터미널에서 `Ctrl+C`
- 코드(파이썬)를 고쳤으면 재시작해야 반영된다. **UI(demo_ui/index.html)는 재시작 없이
  브라우저 새로고침만으로 반영**된다 (서버가 요청마다 파일을 다시 읽음)
- 설정(`configs/*.yaml`) 변경도 재시작 필요

## 7. 문제 해결

| 증상 | 원인·해결 |
|---|---|
| `address already in use` (5001) | 이전 데모가 살아 있음 → `pkill -f run_demo.py` 후 재실행 |
| `카메라(device_id=0)를 열 수 없습니다` | VS Code 카메라 권한 없음 → 시스템 설정 → 카메라에서 Visual Studio Code 허용 후 재실행. (권한 팝업이 안 떴던 이력은 MAC_RUN_LOG §3-1) |
| 포트 5000으로 접속했는데 이상한 응답 | 5000은 맥 AirPlay가 점유 — 맥은 **5001** 이다 (MAC_RUN_LOG §3-2) |
| `ModuleNotFoundError` | 인터프리터가 venv가 아님 → 2번 단계 다시 |
| 로그에 `ollama 웜업 실패` | Ollama 서버 꺼짐 → `brew services start ollama`. 그동안은 규칙 폴백으로 동작 |
| 첫 LLM 판정만 느림/폴백 | 모델 콜드 스타트 — 웜업이 처리하지만, Ollama 를 방금 켰다면 한 번은 느릴 수 있음 |
| 소리가 안 남 | 브라우저 페이지를 한 번 클릭(크롬 자동재생 정책), 음성안내 버튼이 켬(파란색)인지 확인 |
| 제스처 인식이 남의 손에 반응 | 오토포커스 확인 — PIP의 하늘색 원 밖이면 무시가 정상. 반경 조정은 `detect.focus_lock` |

## 8. 자주 쓰는 터미널 명령 (VS Code 내장 터미널)

```bash
./venv/bin/python -m unittest discover tests -v      # 판정 로직 테스트 24건
python scripts/run_demo.py --config configs/config_mac.yaml   # F5와 동일 (venv 활성 시)
python scripts/run_demo.py --config configs/config_mac.yaml --headless  # UI 없이 파이프라인만
curl -s localhost:5001/data | python3 -m json.tool   # FPS·이벤트 상태 JSON
grep gesture_event logs/*.log | tail                 # 최근 판정 이벤트 로그
```

> 용어: **F5(디버그 실행)** — `.vscode/launch.json`에 적힌 구성대로 프로그램을 실행하고
> 중단점(breakpoint)을 걸 수 있는 모드. 코드 줄번호 왼쪽을 클릭해 빨간 점을 찍으면
> 그 줄에서 실행이 멈춰 변수 값을 들여다볼 수 있다 — 판정 로직 디버깅에 유용하다.
