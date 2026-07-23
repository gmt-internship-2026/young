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

2026-07-22 판정식 교체(사용자 실기 리포트 — "손가락 선택이 여러 번 해야 한 번 잡힘"):
구 판정(TIP.y < PIP.y, 손가락이 화면에서 수직으로 곧게 위를 향할 때만 "폄"으로 인식)은
손을 살짝 기울여 들면(예: 얼굴 옆, 가슴 높이) 실제로 편 손가락도 놓치는 각도 의존
문제가 있었다. GMtech_project(gesture_kiosk 자매 코드베이스)가 이미 쓰는 "TIP-손목
거리가 PIP-손목 거리보다 충분히 크면 폄"(거리 비율) 방식으로 교체 — 손이 화면에서
어느 방향으로 돌아가 있어도(면내 회전) 같은 결과를 준다.

2026-07-23 2차 교체(x,y만 쓰던 걸 x,y,z 3차원 거리로) — 실기 리포트: "카메라 쪽으로
손을 쭉 내밀고 손가락을 펴면 오히려 주먹(0개)으로 잡힘". 이 프로젝트의 제스처는
손을 카메라 쪽으로 내밀며 하는 동작이라, 편 손가락이 화면 깊이 방향(카메라 시선과
거의 나란한 방향)으로 뻗는다 — x,y만 보면 원근 단축(foreshortening)으로 화면상
길이가 짧게 눌려 찍혀 임계값을 아무리 조절해도 안정적으로 판정이 안 됐다.
MediaPipe가 이미 제공하는 z(깊이, 손목 기준 상대값 — x와 같은 스케일이라고 공식
문서에 명시)를 x,y와 함께 3차원 거리로 써서, 손가락이 옆으로 뻗든 카메라 쪽으로
뻗든 같은 기준으로 판정되게 한다. infer()가 이제 (x, y, z) 3튜플을 돌려준다.
"""
import math
import os

import cv2

from src.utils.logger import get_logger

logger = get_logger("inference")

WRIST = 0   # MediaPipe 21포인트 손 랜드마크 — 손목(모든 손가락 신전 판정의 기준점)

# MediaPipe 21포인트 손 랜드마크 인덱스 — 검지~새끼(엄지 제외)의 MCP·PIP·DIP·TIP.
# 엄지(1~4)는 손 방향(좌/우 손 · 거울 반전)에 따라 신전 판정 축이 달라져 복잡하므로
# 판정에서 제외한다 — "엄지 이외 정확히 N개 신전"만 센다 (2026-07-16 사용자 확정)
FINGER_JOINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

DEFAULT_EXTENDED_RATIO = 1.3   # gestures.select.extended_ratio 미설정 시 기본값


def count_extended_fingers(landmarks, extended_ratio=DEFAULT_EXTENDED_RATIO):
    """21개 (x, y, z) 좌표 -> 편 손가락 개수(0~4, 엄지 제외).

    TIP-손목 거리가 PIP-손목 거리의 extended_ratio배 이상이면 "폄"으로 본다 — 편
    손가락은 TIP이 관절을 지나 쭉 뻗어 손목에서 멀고, 접힌 손가락은 TIP이 손목 쪽으로
    말려 들어와 가깝다(2026-07-22, GMtech_project finger_select.py와 같은 원리).
    거리는 z(깊이)까지 포함한 3차원 유클리드 거리다(2026-07-23) — 손이 화면에서
    기울어 있어도(면내 회전) 흔들리지 않고, 손가락이 카메라 쪽으로(화면 깊이 방향)
    뻗어도 원근 단축 없이 같은 기준으로 판정된다. math.dist는 튜플 길이가 같으면
    차원 수 무관하게 유클리드 거리를 계산하므로 (x,y,z) 3튜플을 그대로 넣으면 된다.
    """
    wrist = landmarks[WRIST]
    count = 0
    for _mcp, pip_idx, _dip, tip_idx in FINGER_JOINTS.values():
        pip_dist = math.dist(wrist, landmarks[pip_idx])
        if pip_dist < 1e-6:
            continue   # 손목·PIP가 겹치는 퇴화 좌표 — 판단 불가로 접힘 취급
        tip_dist = math.dist(wrist, landmarks[tip_idx])
        if tip_dist >= pip_dist * extended_ratio:
            count += 1
    return count


class HandEstimator:
    """MediaPipe HandLandmarker 래퍼. infer(crop_frame) -> list[list[(x, y, z)]] (손별 21점,
    x·y는 크롭 기준 픽셀 좌표, z는 손목 기준 상대 깊이를 x와 같은 스케일(픽셀)로 맞춘 값 —
    작을수록(음수) 카메라에 더 가깝다. count_extended_fingers()의 3차원 거리 판정용."""

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
        """BGR 크롭 프레임 -> 손별 21랜드마크((x,y,z), 픽셀 스케일) 리스트. 손이 없거나
        크롭이 비면 빈 리스트."""
        if crop_frame.size == 0:
            return []
        rgb_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._Image(image_format=self._ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)
        h_px, w_px = crop_frame.shape[:2]
        # z는 MediaPipe 공식 문서 기준 "x와 대략 같은 스케일"이라 w_px로 함께 맞춘다
        # (정규화 좌표계 기준값이 폭이라 x·z 모두 폭 기준 — 높이 h_px가 아님에 주의)
        return [
            [(lm.x * w_px, lm.y * h_px, lm.z * w_px) for lm in hand_landmarks]
            for hand_landmarks in result.hand_landmarks
        ]
