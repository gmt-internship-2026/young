"""inference 모듈 — 얼굴 검출 (MediaPipe Face Detector/BlazeFace, Apache-2.0).

2026-07-30 도입(사용자 결정 — 손 인식이 옆 사람·다른 손으로 왔다갔다 재발):
카메라에서 가장 가까운(=가장 크게 보이는) 얼굴을 앵커로 잡아 **그 사람의 손만**
인식하기 위한 검출기다. 앵커 선별·도달 반경 게이트는 hand_select.py가 담당하고,
이 모듈은 얼굴 상자만 보고한다.

라이선스: HandLandmarker와 동일 Apache-2.0 — 상업 사용 가능·코드 공개(카피레프트)
의무 없음·제품 화면 표시 의무 없음 (2026-07-11 라이선스 B안 기준 유지, №9).

모델 파일: models/weights/blaze_face_short_range.tflite (0.2MB) —
download_weights.py가 받는다. short-range 모델은 2m 이내 근접용 — 키오스크
사용 거리(0.5~1.5m)와 일치하고, 멀리 있는 대기줄 얼굴은 애초에 잘 안 잡혀
앵커 오염이 적다.
"""
import time
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("inference")


@dataclass
class FaceDetection:
    """얼굴 1개의 검출 결과 — 앵커 선별(hand_select)에 필요한 최소 정보."""

    center_x_px: float
    center_y_px: float
    width_px: float     # 얼굴 상자 폭 — 클수록 가깝다 (앵커 선별 기준 + 도달 반경 자)
    conf: float


class FaceDetector:
    """MediaPipe Face Detector 래퍼. infer(frame) -> list[FaceDetection]."""

    def __init__(self, config):
        detector_cfg = config["face_anchor"]
        self._model_path = detector_cfg["model_path"]
        # mediapipe는 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=detector_cfg.get("min_detection_conf", 0.5),
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        # VIDEO 모드는 단조 증가 타임스탬프(ms)가 필수 (hand_tracker와 동일 규약)
        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info("얼굴 모델 로딩 완료: MediaPipe Face Detector (%s)", self._model_path)

    def infer(self, frame):
        """프레임(BGR·거울 반전 후)의 얼굴들 -> list[FaceDetection].

        거울 보정 불필요 — 얼굴에는 좌/우 라벨이 없고 좌표는 손과 같은
        (반전 후) 프레임 좌표계라 그대로 비교 가능하다.
        """
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame[:, :, ::-1].copy()
        )
        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1   # 단조 증가 보장 (고FPS 보호)
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        faces = []
        for detection in result.detections:
            box = detection.bounding_box
            conf = (float(detection.categories[0].score)
                    if detection.categories else 0.0)
            faces.append(FaceDetection(
                center_x_px=box.origin_x + box.width / 2.0,
                center_y_px=box.origin_y + box.height / 2.0,
                width_px=float(box.width),
                conf=conf,
            ))
        return faces
