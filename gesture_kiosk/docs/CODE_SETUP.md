# 개발 PC 셋업 가이드 (VS Code — 빈 환경 기준)

> 이 문서는 **개발용 보조 PC**(맥/윈도우/x86 리눅스)에서 코드를 고치고 빠르게
> 확인하고 싶을 때 쓰는 선택 사항이다. 배포(윈도우 + NVIDIA) 설치는 [설치가이드.md](../설치가이드.md)를 볼 것.

작성: 2026-07-08. 맥(Apple Silicon)·윈도우·리눅스(우분투 x86) 공통, OS별 차이는 각 단계에 표기.

## 1. 프로그램 설치 (PC에 딱 3개)

| 프로그램 | 확인 방법 | 설치 |
|---|---|---|
| Python 3.10 이상 | 터미널에서 `python3 --version` | 맥: 기본 내장(3.9 이하면 [python.org](https://www.python.org/downloads/)) / 윈도우: python.org에서 설치(설치 시 "Add to PATH" 체크) / 리눅스: `sudo apt install python3 python3-venv python3-pip` |
| Git | `git --version` | 맥: 기본 내장 / 윈도우: [git-scm.com](https://git-scm.com) / 리눅스: `sudo apt install git gh` |
| VS Code | — | [code.visualstudio.com](https://code.visualstudio.com) (리눅스: .deb 다운로드 또는 `sudo snap install code --classic`) |

## 2. VS Code 확장 설치 (2개면 충분)

VS Code 왼쪽 확장(Extensions) 탭에서 검색해 설치:

1. **Python** (Microsoft) — 실행·디버깅
2. **Pylance** (Microsoft) — 자동완성·타입 검사 (보통 Python 설치 시 같이 깔림)

## 3. 프로젝트 열기

```bash
# 이미 클론돼 있으면 생략
gh repo clone <GitHub계정>/GMtech_project   # Private 저장소 — gh auth login 선행

# VS Code로 gesture_kiosk 폴더를 연다 (상위 폴더 말고 gesture_kiosk를!)
code GMtech_project/gesture_kiosk
```

## 4. 가상환경 만들기 — 라이브러리를 프로젝트 안에 격리

VS Code 내장 터미널(``Ctrl+` ``)에서:

```bash
python3 -m venv venv
source venv/bin/activate        # 맥·리눅스 공통 / 윈도우: venv\Scripts\activate
```

그다음 VS Code에 이 가상환경을 알려 준다:
`Cmd+Shift+P`(윈도우 `Ctrl+Shift+P`) → **Python: Select Interpreter** → `./venv` 선택.
이후 터미널 프롬프트 앞에 `(venv)`가 붙어 있는지 항상 확인.

## 5. 라이브러리 설치 — 이 한 줄이 "빈깡통"을 채운다

```bash
pip install -r requirements.txt
```

- onnxruntime(실행기)·rtmlib(포즈)·opencv(카메라)·fastapi(서버)·easyocr 등이 한 번에 설치된다 (수 분 소요)
- PyTorch는 EasyOCR용으로 함께 설치한다. **Apple Silicon 맥은 이 기본 설치만으로
  MPS(맥 GPU 가속)까지 지원** — 코드가 CUDA → MPS → CPU 순으로 자동 감지한다
- 윈도우 + NVIDIA GPU라면 GPU 가속용 torch를 별도 설치:
  `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- 리눅스 x86 PC는 기본 pip torch에 CUDA가 이미 포함 — NVIDIA 드라이버만 있으면 끝
  (`nvidia-smi`가 정상 출력되는지, `python3 -c "import torch; print(torch.cuda.is_available())"`가 True인지 확인)

## 6. 동작 확인 (순서대로)

```bash
# ① 단위 테스트 — 카메라·모델 없이 판정 로직 검증 (즉시 통과해야 정상)
python -m unittest discover tests -v

# ② 모델 가중치 다운로드 (22MB)
python scripts/download_weights.py

# ③ 실시간 데모 — 개발 PC에서는 내장/USB 카메라가 USB 웹캠의 대역을 한다
#    (실전 구성은 어디까지나 윈도우 + NVIDIA + USB 웹캠. 여기서는 코드 확인용일 뿐)
python scripts/run_demo.py
#   브라우저에서 http://localhost:5000 접속
#   macOS가 카메라 권한을 물으면 "허용"
```

③에서 팔을 좌/우로 쓸면 → 화면에 `move_left`/`move_right` 이벤트가 뜨면 전체 파이프라인 정상.

## 문제 해결

| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: yaml` 등 | 가상환경 미활성 — 터미널에 `(venv)` 있는지 확인 후 `source venv/bin/activate` |
| VS Code가 임포트에 빨간 줄 | 인터프리터가 venv가 아님 — 4단계의 Select Interpreter 다시 |
| 카메라가 안 열림 | 맥: 시스템 설정 → 개인정보 보호 → 카메라에서 터미널/VS Code 허용. 리눅스: `sudo usermod -aG video $USER` 후 재로그인. 외장 캠이면 `config.yaml`의 `camera.device_id`를 1, 2로 변경 |
| 맥에서 MPS 관련 연산 오류 | 터미널에서 `export PYTORCH_ENABLE_MPS_FALLBACK=1` 후 재실행 (미지원 연산만 CPU로 우회) |
| 추론이 너무 느림 | `python3 -c "import torch; print(torch.backends.mps.is_available())"` 가 True인지 확인 |

## 개발 시 지켜야 할 것 (기획서 4장)

- 튜닝값 수정은 코드가 아니라 **configs/config.yaml에서만**
- 명명 규칙: snake_case, 용어 사전(frame/gesture/conf/bbox), 단위 접미사(_px/_sec/_ratio)
- 커밋 메시지: `type: 요약` (feat / fix / docs / refactor / test)
- 로직을 고쳤으면 커밋 전에 ① 단위 테스트부터 재실행
