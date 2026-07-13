# 학습일지 — 맥 내장 카메라로 gesture_kiosk 구동 (2026-07-10)

> 목적: 배포 실기 없이 **맥북 내장 카메라**만으로 전체 파이프라인(캡처 → 추론 → 판정 → 이벤트)이
> 동작하는지 검증한다. 셋업 절차 자체는 [CODE_SETUP.md](CODE_SETUP.md)를 따르되,
> 이 문서는 **실제 수행 기록 + 맥에서만 발생한 문제 2건과 해결법**을 남긴다.

## 1. 환경

| 항목 | 값 |
|---|---|
| 기기 | MacBook Pro (Apple Silicon) — 내장 카메라 사용 |
| OS | macOS (Darwin 25.5.0) |
| Python | 3.12 (venv) |
| 주요 패키지 | torch 2.13.0(arm64/MPS), ultralytics 8.4.91, opencv-python 5.0.0.93, fastapi 0.139.0 |
| 모델 | HaGRIDv2 사전학습 YOLOv10n_gestures.pt (22MB, 학습 0회) |
| 추론 디바이스 | **MPS (맥 GPU)** — 코드가 CUDA → MPS → CPU 순 자동 감지 |

## 2. 수행 절차 (실제 실행한 명령 그대로)

```bash
# 0) 작업 전 백업 (~/Documents/backup/gesture_kiosk_20260710_mac_run)
rsync -a --exclude '__pycache__' gesture_kiosk/ ~/Documents/backup/gesture_kiosk_20260710_mac_run/

cd gesture_kiosk

# 1) 가상환경 + 라이브러리 (CODE_SETUP.md 4~5단계와 동일)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt        # 약 2GB, 수 분 소요

# 2) 단위 테스트 — 카메라·모델 없이 판정 로직 검증
./venv/bin/python -m unittest discover tests -v   # → 11개 전부 OK

# 3) MPS(맥 GPU) 가용성 확인
./venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"  # → True

# 4) 가중치는 이미 존재 (없으면: ./venv/bin/python scripts/download_weights.py)

# 5) 데모 실행 — 맥은 반드시 아래 스크립트로! (§3-1 카메라 권한 문제 때문)
open scripts/run_demo_mac.command
# 브라우저: http://localhost:5001   ← 5000이 아님! (§3-2)
```

## 3. 맥에서만 발생한 문제 2건과 해결

### 3-1. 카메라 권한(TCC)이 팝업 없이 자동 거부됨

**증상**: `RuntimeError: 카메라(device_id=0)를 열 수 없습니다`
그 위에 `OpenCV: not authorized to capture video (status 0), requesting...` 로그.

**원인**: macOS는 카메라 접근 권한을 **프로세스를 소유한 GUI 앱** 단위로 판정한다(TCC).
Claude·IDE 등 GUI 앱의 하위 셸에서 파이썬을 실행하면 권한 요청이 그 앱 이름으로
올라가는데, 앱에 따라 팝업 없이 자동 거부되고 **시스템 설정 → 개인정보 보호 및 보안 →
카메라 목록에 아예 나타나지도 않는다** (신형 설정 앱에는 수동 추가 버튼도 없음).

**해결**: **Terminal.app에서 실행**하면 권한이 Terminal에 귀속되어 정상적으로 팝업이 뜬다.
이를 위해 `scripts/run_demo_mac.command`를 만들었다 — 더블클릭(또는 `open`)하면
Terminal이 열리며 실행되고, 권한 허용 전 실패는 3초 간격으로 자동 재시도한다.

> 용어: **TCC**(Transparency, Consent, and Control) — macOS가 카메라·마이크 등
> 민감 자원 접근을 앱 단위로 허용/거부하는 개인정보 보호 체계.

### 3-2. 포트 5000을 맥 AirPlay 수신기가 선점

**증상**: `lsof -nP -i :5000` → `ControlCe`(ControlCenter)가 이미 LISTEN 중.
데모 서버가 5000에 못 뜨고, 더 헷갈리는 건 **AirPlay가 5000에서 응답하기 때문에
curl이 성공한 것처럼 보인다**는 점 (서버가 떠 있다고 착각하기 쉬움).

**원인**: macOS Monterey부터 [시스템 설정 → 일반 → AirDrop 및 Handoff → AirPlay 수신 모드]가
켜져 있으면 ControlCenter가 포트 5000(+7000)을 상시 점유한다.

**해결**: 시스템 설정을 건드리지 않고 **맥 전용 설정 파일** `configs/config_mac.yaml`을
만들어 `demo_ui.port: 5001`로 변경했다 (원본 `config.yaml`은 배포용 5000 그대로 보존).
`run_demo_mac.command`가 이 설정으로 실행한다.

## 4. 검증 결과 (2026-07-10)

| 검증 항목 | 결과 |
|---|---|
| 단위 테스트 (판정 로직 11건) | ✅ 전부 통과 |
| 카메라 캡처 FPS | ✅ **30.0** (내장 카메라 최대치) |
| 추론 FPS (YOLOv10n @ MPS) | ✅ **약 185~209** — KPI 30 FPS 대비 6배 이상 여유 |
| MJPEG 스트림 (`/video_feed`) | ✅ 실제 카메라 화면 + FPS 오버레이 수신 확인 |
| 제스처 이벤트 | ✅ 손바닥 제시 → `palm_stop` conf 0.72 / 0.61 두 건 감지 |

- 확인 방법: `curl http://localhost:5001/data` → `capture_fps`·`infer_fps`·`events` JSON
- 주의: 맥에서의 FPS는 **개발 환경 참고치**일 뿐, KPI 판정은 배포 장비(윈도우 + NVIDIA)에서 측정한다

## 5. 이후 맥에서 실행하는 법 (요약)

```bash
open gesture_kiosk/scripts/run_demo_mac.command   # 또는 Finder에서 더블클릭
# → 브라우저에서 http://localhost:5001
# 종료: Terminal 창에서 Ctrl+C
```

venv·가중치가 이미 있으므로 위 한 줄이면 된다. 새 맥이라면 §2를 처음부터.

## 6. 추가 기록 (2026-07-10 오후) — Python 3.10 재구성 + VS Code 실행 환경

당시 검토하던 배포 후보 장비(Ubuntu 22.04)의 **Python 3.10과 버전을 일치**시키기 위해
venv를 3.12 → 3.10으로 재구성했다. 절차는 §2와 동일하고 인터프리터만 다르다:

```bash
brew install python@3.10                  # 3.10.20 설치
rm -rf venv
/opt/homebrew/bin/python3.10 -m venv venv
./venv/bin/pip install -r requirements.txt
```

- 패키지 차이: **numpy 2.5 → 2.2.6** (numpy 2.3부터 Python 3.11 이상 요구 → pip이 자동 하향).
  torch 2.13(MPS)·ultralytics 8.4.91·opencv 5.0은 동일
- 재검증 결과: 단위 테스트 11건 OK, 캡처 **30.0 FPS**, 추론 **174.6 FPS** (MPS) — 3.12와 동등

**VS Code 실행 구성** (`.vscode/` — CODE_SETUP.md 3~4단계 자동화):

| 파일 | 역할 |
|---|---|
| `.vscode/settings.json` | 인터프리터를 `./venv`(3.10)로 자동 지정 — "Select Interpreter" 수동 단계 불필요 |
| `.vscode/launch.json` | **F5 → "데모 실행 (맥 — localhost:5001)"** 선택으로 즉시 구동. 단위 테스트 구성도 포함 |

VS Code에서 처음 실행하면 카메라 권한 팝업이 "Visual Studio Code" 이름으로 뜬다 → 허용.
포트 5001은 한 프로세스만 쓸 수 있으므로, **Terminal(.command)로 띄운 데모가 돌고 있다면
Ctrl+C로 끄고 F5를 누를 것** (안 끄면 `address already in use`).

## 7. 참고 자료

- [CODE_SETUP.md](CODE_SETUP.md) — 개발 PC(맥/윈도우/리눅스) 공통 셋업 가이드
- HaGRID 모델·데이터셋: https://github.com/hukenovs/hagrid (CC BY-SA 4.0 변형 — 상용 탑재 전 회사 검토)
- Apple TCC 개요: https://support.apple.com/ko-kr/guide/security/secb0e2b247d/web
- AirPlay 수신기의 포트 5000 점유: macOS Monterey 릴리즈 노트 및 개발자 커뮤니티 다수 보고
