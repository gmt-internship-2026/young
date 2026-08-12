"""설치 검증(스모크 테스트) — 카메라 없이 설치가 제대로 됐는지 확인한다.

install.bat 마지막 단계에서 자동 실행된다. 확인 항목:
1. 파이썬 버전 (배포 기준 3.11.5 — 시험 장비의 다른 버전은 경고만)
2. 핵심 패키지 임포트 (mediapipe·cv2·yaml·numpy)
3. 더미 프레임 추론 — 손(MediaPipe HandLandmarker): 2026-07-29 포즈 제거로
   유일한 추론 모델 (손 모양·궤적·선별 전부 이걸로 판정)

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
    # 버전 불일치는 경고만 — 시험 장비에서는 정상 상황.
    # 배포(윈도우) PC에서는 반드시 3.11.5인지 이 줄의 출력을 눈으로 확인할 것
    check(
        f"파이썬 버전 (배포 기준 {expected_python})",
        True,
        f"현재 {actual_python}" + ("" if actual_python == expected_python else " ← 배포 기준과 다름 (시험 장비면 무시)"),
    )

    for module_name in ("mediapipe", "cv2", "yaml", "numpy"):
        try:
            __import__(module_name)
            check(f"{module_name} 임포트", True)
        except ImportError as error:
            check(f"{module_name} 임포트", False, str(error))

    # 손이 유일한 추론 모델(2026-07-29 포즈 제거) — 모양·궤적·선별 전부 이걸로 판정
    try:
        import numpy as np

        from src.inference.hand_tracker import HandTracker

        hand_tracker = HandTracker(config)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        hand_tracker.infer(dummy)
        check("더미 프레임 추론 (손 — 유일한 추론 모델)", True)
    except Exception as error:  # hand_landmarker.task 누락 등 — 원인 그대로 보여준다
        check("더미 프레임 추론 (손 — 유일한 추론 모델)", False, repr(error))

    print()
    if is_all_passed:
        print("[OK] 스모크 테스트 통과 — 실행: py main.py")
        return 0
    print("[NG] 실패 항목이 있습니다 — 설치가이드.md 문제 해결 절 참고")
    return 1


if __name__ == "__main__":
    sys.exit(main())
