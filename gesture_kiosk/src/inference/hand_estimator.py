"""inference 모듈 — 손 랜드마크(MediaPipe HandLandmarker)를 추론해 편 손가락 개수를 센다.

2026-07-16 추가 — 선택(select)이 "고개 꾸벅 2회"에서 "손가락 1개 인식"으로 바뀌면서
필요해졌다(무손·무지 접근성 요건 제외 — 회사 확인 사항, docs/TODO.md 참고). rtmlib의
Hand/Wholebody(검출+포즈 2단계, mmpose 계열)보다 CPU 실시간 손 추적에 강점이 있는
MediaPipe Tasks API를 썼다. 모델(hand_landmarker.task)은 저장소에 이미 포함돼 있어
재다운로드가 필요 없다(2026-07-15 개편 때 지우다 남은 파일을 재사용).

person_lock이 잠근 사용자의 bbox로 크롭한 영역만 추론한다(pipeline 쪽 책임) — 프레임
전체를 돌리면 CPU 30 FPS 예산을 넘길 위험이 크고, 다른 사람 손을 오인식할 수도 있다.

count_extended_fingers()는 mediapipe 없이 좌표만으로 판정하는 순수 함수라 단위
테스트가 가볍다(카메라·모델 없이 테스트 원칙 — tests/test_hand_estimator.py).
"""
import os

import cv2

from src.utils.logger import get_logger

logger = get_logger("inference")

# MediaPipe 21포인트 손 랜드마크 인덱스 — 검지~새끼(엄지 제외)의 MCP·PIP·DIP·TIP.
# 엄지(1~4)는 손 방향(좌/우 손 · 거울 반전)에 따라 신전 판정 축이 달라져 복잡하므로
# 판정에서 제외한다 — "엄지 이외 정확히 N개 신전"만 센다 (2026-07-16 사용자 확정)
FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def count_extended_fingers(landmarks):
    """21개 (x, y) 좌표 -> 편 손가락 개수(0~4, 엄지 제외).

    TIP의 y가 PIP보다 작으면(이미지 좌표는 y가 아래로 증가) 편 것으로 본다 — 카메라를
    정면으로 보고 손을 세워 드는 키오스크 사용 자세를 전제한 단순 판정이다.
    """
    count = 0
    for _mcp, pip_idx, _dip, tip_idx in FINGER_JOINTS.values():
        if landmarks[tip_idx][1] < landmarks[pip_idx][1]:
            count += 1
    return count


class HandEstimator:
    """MediaPipe HandLandmarker 래퍼. infer(crop_frame) -> list[list[(x, y)]] (손별 21점, 픽셀 좌표)."""

    def __init__(self, config):
        from mediapipe import Image, ImageFormat
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker, HandLandmarkerOptions, RunningMode,
        )

        hand_cfg = config["hand_model"]
        weights_path = os.path.join(config["root_dir"], hand_cfg["weights_path"])
        # model_asset_path가 아니라 bytes 버퍼로 넘긴다 — mediapipe 네이티브 파일 열기가
        # 비ASCII 경로(사용자 폴더명에 한글 등)를 못 여는 경우가 있어(2026-07-16 실측),
        # 파이썬 쪽에서 직접 읽어 우회한다.
        with open(weights_path, "rb") as f:
            model_bytes = f.read()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=RunningMode.IMAGE,
            num_hands=hand_cfg["num_hands"],
            min_hand_detection_confidence=hand_cfg["min_hand_detection_conf"],
            min_hand_presence_confidence=hand_cfg["min_hand_presence_conf"],
            min_tracking_confidence=hand_cfg["min_tracking_conf"],
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._Image = Image
        self._ImageFormat = ImageFormat
        logger.info(
            "손 모델 로딩 완료: MediaPipe HandLandmarker(num_hands=%d)", hand_cfg["num_hands"]
        )

    def infer(self, crop_frame):
        """BGR 크롭 프레임 -> 손별 21랜드마크(픽셀 좌표) 리스트. 손이 없거나 크롭이 비면 빈 리스트."""
        if crop_frame.size == 0:
            return []
        rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)
        h_px, w_px = crop_frame.shape[:2]
        return [
            [(lm.x * w_px, lm.y * h_px) for lm in hand_landmarks]
            for hand_landmarks in result.hand_landmarks
        ]
