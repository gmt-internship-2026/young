"""inference 모듈 — 손 랜드마크 추적 (MediaPipe HandLandmarker, Apache-2.0).

2026-07-28 도입 (wholebody 손 21점 교체 — 노트북 실기에서 확인된 구조적 문제 2건):
1. 톱다운 포즈(wholebody)는 안 보이는 손도 133점을 강제로 찍는다 — 가려진 손의
   21점이 보이는 손 위에 얹혀 **한 팔에 좌/우 손이 겹치고** 궤적·다수결이 오염됐다.
2. 2D 투영 길이만으로는 카메라를 가리키는 손가락(원근 단축)과 굽힘을 구분하지
   못해 한 손가락이 주먹으로 오판됐다 (임계값 튜닝으로 해결 불가 — 07-28 실측).
MediaPipe는 손바닥 검출기가 **실제 보이는 손만** 보고하고(유령 손 소멸) 좌/우
판별(handedness)을 내장하며, 랜드마크에 z(깊이)가 있어 3D 거리 판별이 가능하다
(hand_shape.py가 사용). 라이선스 Apache-2.0 — rtmlib과 동일 기준(상업 사용 가능,
코드 공개 의무 없음): 2026-07-11 라이선스 B안 유지.

모델 파일: models/weights/hand_landmarker.task — download_weights.py가 받는다
(mediapipe 1.0은 구 Solutions API가 제거돼 모델을 wheel에 담지 않는다).

랜드마크 번호(MediaPipe 21점): 0=손목뿌리, 1~4=엄지, 5~8=검지, 9~12=중지,
13~16=약지, 17~20=새끼 — 각 손가락은 (MCP, PIP, DIP, TIP) 순서.
"""
import time
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

HAND_KPT_COUNT = 21
DUPLICATE_CENTER_SPAN_RATIO = 0.5   # 중복 검출 판정 — 두 검출의 중심 거리가 손 크기의
                                    # 이 비율 이내면 같은 물리적 손 (서로 다른 두 손은
                                    # 물리적으로 겹칠 수 없다 — 2026-07-29 실기)


def suppress_duplicate_hands(hands):
    """같은 물리적 손의 중복 검출 억제 (2026-07-29 실기 — 유령 라벨 재발 대응).

    MediaPipe가 드물게 한 손을 좌/우 라벨로 **두 번** 보고한다 — 한 손에 L·R이
    겹쳐 붙어 활성 팔 선택·궤적·래치가 오염된다(실기: 오른손에 왼손 라벨 동반).
    중심 거리가 손 크기(랜드마크 폭)의 절반 이내면 같은 손으로 보고 handedness
    신뢰도가 높은 검출만 남긴다. 순수 함수 — tests/test_hand_tracker.py.
    """
    kept = []
    for hand in sorted(hands, key=lambda entry: -entry.conf):
        center_x = float(hand.landmarks[:, 0].mean())
        center_y = float(hand.landmarks[:, 1].mean())
        span_px = max(
            float(hand.landmarks[:, 0].max() - hand.landmarks[:, 0].min()),
            float(hand.landmarks[:, 1].max() - hand.landmarks[:, 1].min()),
        )
        is_duplicate = False
        for other_center_x, other_center_y, other_span_px, _ in kept:
            dist_px = ((center_x - other_center_x) ** 2
                       + (center_y - other_center_y) ** 2) ** 0.5
            if dist_px < DUPLICATE_CENTER_SPAN_RATIO * max(span_px, other_span_px):
                is_duplicate = True   # 신뢰도 내림차순 순회 — 먼저 남은 쪽이 더 확실한 라벨
                break
        if not is_duplicate:
            kept.append((center_x, center_y, span_px, hand))
    return [entry[3] for entry in kept]


@dataclass
class HandDetection:
    """손 1개의 추적 결과 (기획서 4.6 공통 데이터 구조 스타일).

    user_side: **사용자 기준** "left"/"right" — HandLandmarker(Tasks API)의
    handedness는 반전 없는 원본 영상 기준이라(2026-07-28 실기 확인), 거울 모드
    (camera.mirror=true — 배포 기본)의 프레임에서는 이 모듈이 라벨을 뒤집어
    사용자 기준으로 맞춘다 (person_lock의 좌/우 스왑 대상이 아니다).
    """

    user_side: str
    landmarks: np.ndarray        # shape (21, 3) — (x_px, y_px, z_px). 화면 좌표 —
                                 #   손 중심 궤적·사용자 귀속(어깨 거리)용
    world_landmarks: np.ndarray  # shape (21, 3) — 미터 단위 월드 좌표(손 중심 원점).
                                 #   손 모양 판별용 (2026-07-28 실기: 화면 z는 노이즈가
                                 #   커서 한 손가락이 주먹으로 오판 — 시점 불변인
                                 #   월드 기하로 판별해야 한다, hand_shape.py)
    conf: float                  # handedness 신뢰도


class HandTracker:
    """MediaPipe HandLandmarker 래퍼. infer(frame) -> list[HandDetection]."""

    def __init__(self, config):
        tracker_cfg = config["hand_tracker"]
        self._model_path = tracker_cfg["model_path"]
        self._is_mirror = config["camera"]["mirror"]
        # mediapipe는 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        # model_asset_path가 아니라 buffer로 넘긴다(2026-08-03 실기) — mediapipe
        # 네이티브 로더가 비ASCII(한글 등) 경로에서 파일을 못 연다(errno=-1).
        # 파이썬 open()은 유니코드 경로를 문제없이 읽으므로, 바이트를 직접 넘겨
        # 우회한다 — 배포 PC 계정명이 한글이어도(흔한 사례) 깨지지 않는다
        with open(self._model_path, "rb") as weights_file:
            model_bytes = weights_file.read()
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=tracker_cfg["max_num_hands"],
            min_hand_detection_confidence=tracker_cfg["min_detection_conf"],
            min_hand_presence_confidence=tracker_cfg["min_presence_conf"],
            min_tracking_confidence=tracker_cfg["min_tracking_conf"],
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        # VIDEO 모드는 단조 증가 타임스탬프(ms)가 필수 — 프레임 간 추적에 쓰인다
        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info(
            "손 모델 로딩 완료: MediaPipe HandLandmarker (max_num_hands=%d, %s)",
            tracker_cfg["max_num_hands"], self._model_path,
        )

    def infer(self, frame):
        """프레임(BGR·거울 반전 후)에서 보이는 손을 추적한다 -> list[HandDetection]."""
        h_px, w_px = frame.shape[:2]
        # MediaPipe는 RGB 입력 — cv2 프레임(BGR)을 뒤집는다
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame[:, :, ::-1].copy()
        )
        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1   # 단조 증가 보장 (고FPS 보호)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = []
        for hand_landmarks, world_landmarks, handedness in zip(
                result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            category = handedness[0]
            # 실기 정정(2026-07-28 노트북): HandLandmarker(Tasks API)의 handedness는
            # **반전 없는 원본 영상 기준**으로 관찰됐다 — 거울 프레임에서는 라벨을
            # 뒤집어야 사용자 기준과 일치한다 (구 Solutions 문서의 "셀피 기준" 각주와
            # 반대 동작 — 문서보다 실측을 따른다)
            side = category.category_name.lower()
            if self._is_mirror:
                side = "right" if side == "left" else "left"
            landmarks = np.array(
                # z도 프레임 폭 스케일 — MediaPipe z는 x와 같은 정규화 스케일이라
                # 폭을 곱하면 x_px·y_px와 단위가 맞는 3D 거리 계산이 가능하다
                [(lm.x * w_px, lm.y * h_px, lm.z * w_px) for lm in hand_landmarks],
                dtype=np.float32,
            )
            world = np.array(
                [(lm.x, lm.y, lm.z) for lm in world_landmarks], dtype=np.float32,
            )
            hands.append(
                HandDetection(user_side=side, landmarks=landmarks,
                              world_landmarks=world, conf=float(category.score))
            )
        return suppress_duplicate_hands(hands)
