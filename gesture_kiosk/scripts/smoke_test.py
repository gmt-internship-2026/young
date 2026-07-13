"""설치 검증(스모크 테스트) — 카메라 없이 설치가 제대로 됐는지 확인한다.

install.bat 마지막 단계에서 자동 실행된다. 확인 항목:
1. 파이썬 버전 (배포 기준 3.11.5 — 시험 장비의 다른 버전은 경고만)
2. 핵심 패키지 임포트 (onnxruntime·rtmlib·cv2·fastapi / 선택: easyocr·pyttsx3·torch)
3. GPU 가속 확인 (onnxruntime CUDA — 맥 시험 장비는 CPU 안내)
4. 제스처 ONNX 존재 + 더미 프레임 추론 (제스처 + 포즈)

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

    for module_name in ("mediapipe", "onnxruntime", "rtmlib", "cv2", "fastapi", "yaml", "numpy"):
        try:
            __import__(module_name)
            check(f"{module_name} 임포트", True)
        except ImportError as error:
            check(f"{module_name} 임포트", False, str(error))

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            check("GPU 가속 (onnxruntime CUDA)", True)
        else:
            check("GPU 가속", sys.platform == "darwin",
                  "CPU 실행 — 맥 시험 장비면 정상, 배포(윈도우+NVIDIA) PC라면 onnxruntime-gpu 설치 확인")
    except ImportError:
        pass  # 위 임포트 검사에서 이미 FAIL 처리됨

    if config["ocr"]["enabled"]:
        for module_name, label in (("torch", "torch (EasyOCR용)"), ("easyocr", "easyocr (주민등록증 OCR)")):
            try:
                __import__(module_name)
                check(f"{label} 임포트", True)
            except ImportError as error:
                check(f"{label} 임포트", False, str(error))
    if config["announce"]["enabled"] and config["announce"]["backend"] == "tts":
        try:
            __import__("pyttsx3")
            check("pyttsx3 임포트 (음성 안내)", True)
        except ImportError as error:
            check("pyttsx3 임포트 (음성 안내)", False, str(error))

    check("제스처 ONNX 파일", os.path.exists(config["model"]["gesture_onnx_path"]),
          config["model"]["gesture_onnx_path"])

    try:
        import numpy as np

        from src.inference.detector import create_gesture_detector

        detector = create_gesture_detector(config)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.infer(dummy)
        check("더미 프레임 추론 (제스처)", True,
              f"engine={config['model'].get('gesture_engine', 'mediapipe')}")
    except Exception as error:  # 모델 누락·드라이버 문제 등 — 원인 그대로 보여준다
        check("더미 프레임 추론 (제스처)", False, repr(error))

    if config["person_lock"]["enabled"]:
        try:
            import numpy as np

            from src.inference.pose_estimator import PoseEstimator

            pose = PoseEstimator(config)
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            pose.infer(dummy)
            check("더미 프레임 추론 (포즈 — 사용자 잠금)", True)
        except Exception as error:
            check("더미 프레임 추론 (포즈 — 사용자 잠금)", False, repr(error))

    print()
    if is_all_passed:
        print("[OK] 스모크 테스트 통과 — run.bat으로 실행하세요")
        return 0
    print("[NG] 실패 항목이 있습니다 — 설치가이드.md 문제 해결 절 참고")
    return 1


if __name__ == "__main__":
    sys.exit(main())
