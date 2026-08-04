"""inference 전처리 — 추론 입력용으로 프레임을 가공한다 (기획서 4.6 계약).

현재 백엔드(ultralytics)는 리사이즈·정규화를 내부에서 처리하므로
여기서는 거울 반전만 담당한다. 추후 TensorRT 바인딩을 직접 다루게 되면
letterbox·정규화·CHW 변환이 이 모듈로 들어온다.
"""
import cv2


class Preprocessor:
    def __init__(self, config):
        self._is_mirror = config["camera"]["mirror"]

    def preprocess_frame(self, frame):
        """frame(BGR) -> input_tensor. 거울 모드면 좌우 반전한다."""
        if self._is_mirror:
            frame = cv2.flip(frame, 1)
        return frame
