# training — 학습(파인튜닝) 작업 공간

**조건부 가동** (2026-07-10 결정): 추론(gesture_kiosk)은 사전학습 모델로 무학습 동작이
기본이다. 아래 상황이 **실측으로 확인될 때만** 이 폴더로 학습을 시작한다.

1. 신규 제스처 스펙(주먹→펴기 등)의 인식 정확도가 KPI(85%) 미달

(주민등록증 인식 학습 항목은 2026-07-16 OCR 기능 전면 제거로 삭제됨 — idcard/ 폴더도 함께 삭제)

## 환경 — OS 무관, 결과만 이식

학습은 **리눅스 / 맥 / 윈도우 어디서든** 가능하다 (RTX 5080, 맥북 M5 Pro 등).
결과물(.pt 가중치)은 OS·장비와 무관한 텐서 파일이라, 어디서 학습했든
`export/to_inference.py`로 추론(윈도우) 쪽에 그대로 이식된다.

| OS | 셋업 | 학습 디바이스 |
|---|---|---|
| 윈도우 (RTX) | `setup_windows.bat` | CUDA (cu128 — RTX 50시리즈 포함) |
| 맥 (Apple Silicon) | `bash setup_mac.sh` | MPS (CUDA 전용 커널 코드는 불가 — train.py는 자동 감지) |
| 리눅스 (RTX/서버) | `bash setup_linux.sh` | CUDA |

> Jetson Orin Nano는 **학습에 부적합** (추론 특화 보드·공유 메모리 8GB) — 학습 장비에서 제외.

## 브랜치 운용 (feat/study/*)

`feat/study/windows` `feat/study/mac` `feat/study/linux` 세 브랜치는 **같은 내용에서
시작**한다. 각 OS에서 작업할 때 해당 브랜치를 쓰고, 공통 코드 수정은 한 브랜치에서
커밋 후 나머지에 머지한다 (내용 분기 최소화 원칙).

## 폴더 구조

```
training/
├─ setup_windows.bat / setup_mac.sh / setup_linux.sh   # OS별 환경 구성 (코드는 공통)
├─ requirements.txt        # 학습 공통 의존성 (torch는 OS별 스크립트가 설치)
├─ gesture/                # 제스처 파인튜닝
│   ├─ collect_frames.py   #    카메라로 학습 프레임 수집 (라벨링 전 단계)
│   ├─ dataset_template.yaml  # YOLO 데이터셋 양식 — 라벨링 후 경로 채움
│   └─ train.py            #    파인튜닝 (device 자동: cuda→mps→cpu)
└─ export/
    └─ to_inference.py     # 학습 결과 → gesture_kiosk/models/weights (기획서 4.8 명명규칙)
```

## 표준 흐름

```bash
# 1) 환경 (OS에 맞는 스크립트 1회)
setup_windows.bat            # 또는 bash setup_mac.sh / bash setup_linux.sh

# 2) 데이터 수집 → 라벨링 (라벨링 도구는 회사 협의 — labelImg/Roboflow 등)
python gesture/collect_frames.py --out ../gesture_kiosk/data/raw/session1

# 3) 파인튜닝 (기존 HaGRID 가중치에서 이어서)
python gesture/train.py --data gesture/dataset_v1.yaml --epochs 50

# 4) 추론 쪽으로 이식 (명명규칙 적용 + 검증 안내 출력)
python export/to_inference.py --weights runs/pose/train/weights/best.pt --dataset v1 --metric map85
```

## 주의

- 데이터 수집·라벨링 시 **인물 단위 분할**(기획서 5.4) — 같은 사람이 train/val에 겹치면 안 됨
- HaGRID(CC BY-SA 4.0 변형)·ultralytics(AGPL-3.0) 라이선스 — 파인튜닝 결과물에도 승계됨 (№9)
