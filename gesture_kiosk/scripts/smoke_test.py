"""설치 검증(스모크 테스트) — 카메라 없이 설치가 제대로 됐는지 확인한다.

install.bat 마지막 단계에서 자동 실행된다. 확인 항목:
1. 파이썬 버전 (배포 기준 3.11.5 — 시험 장비의 다른 버전은 경고만)
2. 핵심 패키지 임포트 (onnxruntime·rtmlib·cv2·mediapipe)
3. 실행 장치 확인 (통합판 — GPU 스택이면 CUDA 인식, CPU 스택이면 CPU 정상)
4. 더미 프레임 추론 — 포즈(검출·잠금·어깨 자) + 손(MediaPipe — 손 모양·손 중심,
   2026-07-28 교체)

사용법 (프로젝트 루트에서):
    python scripts/smoke_test.py
종료 코드 0 = 통과. 실패 항목은 [FAIL]로 표시된다.
"""
import os
import platform
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")

is_all_passed = True


def check(label, is_ok, detail=""):
    global is_all_passed
    mark = "PASS" if is_ok else "FAIL"
    if not is_ok:
        is_all_passed = False
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))


def main():
    config = load_config(DEFAULT_CONFIG_PATH)

    expected_python = config["runtime"]["python_version"]
    actual_python = platform.python_version()
    # 버전 불일치는 경고만 — 시험 장비(맥 등 Python 3.10)에서는 정상 상황.
    # 배포(윈도우) PC에서는 반드시 3.11.5인지 이 줄의 출력을 눈으로 확인할 것
    check(
        f"파이썬 버전 (배포 기준 {expected_python})",
        True,
        f"현재 {actual_python}" + ("" if actual_python == expected_python else " ← 배포 기준과 다름 (시험 장비면 무시)"),
    )

    for module_name in ("onnxruntime", "rtmlib", "cv2", "yaml", "numpy", "mediapipe"):
        try:
            __import__(module_name)
            check(f"{module_name} 임포트", True)
        except ImportError as error:
            check(f"{module_name} 임포트", False, str(error))

    try:
        import onnxruntime as ort

        # 통합판(2026-07-24): install.bat이 GPU 감지 시 torch(cu128)+onnxruntime-gpu를
        # 설치한다 — torch가 CUDA를 보는데 ORT에 CUDA가 없으면 rtmlib이 끌고 온
        # CPU판 onnxruntime이 GPU판을 덮어쓴 상태(복구 필요)라 FAIL로 잡는다
        try:
            import torch
            gpu_stack_installed = torch.cuda.is_available()
        except ImportError:
            gpu_stack_installed = False

        if "CUDAExecutionProvider" in ort.get_available_providers():
            check("실행 장치 — GPU 가속 (onnxruntime CUDA)", True)
        elif gpu_stack_installed:
            check("실행 장치", False,
                  "torch는 CUDA 사용 가능인데 onnxruntime이 CPU — GPU판 복구 필요 (install.bat 재실행 또는 설치가이드 G절)")
        else:
            check("실행 장치 — CPU", True,
                  "GPU 미감지 — CPU 추론(정상). NVIDIA GPU가 있는 PC라면 install.bat gpu 재실행")
    except ImportError:
        pass  # 위 임포트 검사에서 이미 FAIL 처리됨


    # 포즈 = 사람 검출·잠금·어깨 자, 손 = 손 모양·손 중심 (2026-07-28 모델 분리)
    try:
        import numpy as np

        from src.inference.pose_estimator import PoseEstimator

        pose = PoseEstimator(config)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        pose.infer(dummy)
        check("더미 프레임 추론 (포즈 — 검출·잠금·어깨 자)", True)
    except Exception as error:  # 모델 누락·드라이버 문제 등 — 원인 그대로 보여준다
        check("더미 프레임 추론 (포즈 — 검출·잠금·어깨 자)", False, repr(error))

    try:
        import numpy as np

        from src.inference.hand_tracker import HandTracker

        hand_tracker = HandTracker(config)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_tracker.infer(dummy)
        check("더미 프레임 추론 (손 — 손 모양·손 중심)", True)
    except Exception as error:  # hand_landmarker.task 누락 등 — 원인 그대로 보여준다
        check("더미 프레임 추론 (손 — 손 모양·손 중심)", False, repr(error))

    print()
    if is_all_passed:
        print("[OK] 스모크 테스트 통과 — run.bat으로 실행하세요")
        return 0
    print("[NG] 실패 항목이 있습니다 — 설치가이드.md 문제 해결 절 참고")
    return 1


if __name__ == "__main__":
    sys.exit(main())
