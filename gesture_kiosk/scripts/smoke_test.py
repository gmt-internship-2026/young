"""설치 검증(스모크 테스트) — 카메라 없이 설치가 제대로 됐는지 확인한다.

install.bat 마지막 단계에서 자동 실행된다. 확인 항목:
1. 파이썬 버전 (배포 기준 3.11.5 — 시험 장비의 다른 버전은 경고만)
2. 핵심 패키지 임포트 (onnxruntime·rtmlib·cv2)
3. GPU 가속 확인 (onnxruntime CUDA — 맥 시험 장비는 CPU 안내)
4. 더미 프레임 추론 (포즈 — 유일한 모델: 손 모양·궤적·잠금 전부 이걸로 판정)

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

    for module_name in ("onnxruntime", "rtmlib", "cv2", "yaml", "numpy"):
        try:
            __import__(module_name)
            check(f"{module_name} 임포트", True)
        except ImportError as error:
            check(f"{module_name} 임포트", False, str(error))

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            check("실행 장치", True, "CUDA 가용 PC — 이 CPU판은 CPU로만 실행 (GPU판: feat/think_win_gpu)")
        else:
            check("실행 장치 (CPU)", True, "CPU 추론판 — 정상")
    except ImportError:
        pass  # 위 임포트 검사에서 이미 FAIL 처리됨


    # 포즈가 유일한 모델 — 손 모양(주먹/한 손가락)·궤적·잠금이 전부 이걸로 판정 (2026-07-23)
    try:
        import numpy as np

        from src.inference.pose_estimator import PoseEstimator

        pose = PoseEstimator(config)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        pose.infer(dummy)
        check("더미 프레임 추론 (포즈 — 손 모양·궤적·잠금)", True)
    except Exception as error:  # 모델 누락·드라이버 문제 등 — 원인 그대로 보여준다
        check("더미 프레임 추론 (포즈 — 손 모양·궤적·잠금)", False, repr(error))

    print()
    if is_all_passed:
        print("[OK] 스모크 테스트 통과 — run.bat으로 실행하세요")
        return 0
    print("[NG] 실패 항목이 있습니다 — 설치가이드.md 문제 해결 절 참고")
    return 1


if __name__ == "__main__":
    sys.exit(main())
