"""inference 모듈 — MediaPipe 손 랜드마크로 손등/손바닥 방향을 검출한다 (라이선스 C안).

2026-07-15 동작 개편: 제스처 어휘를 손가락 모양(주먹/OK 등)에서 손 방향
(손등/손바닥)으로 교체했다 — 선택(OK)이 '손등 보이기'로 바뀌었기 때문이다
(범용 설계: 손목 돌리기만 하면 되므로 손가락이 없는 사용자도 가능).

판정 원리: (검지MCP-손목)×(새끼MCP-손목) 외적의 z부호는 손등/손바닥 중
어느 쪽이 카메라를 향하는지에 따라 뒤집히고, 화면 내 회전에는 불변이다.
손가락 끝(TIP)을 쓰지 않아 손가락이 일부 없는 손에도 강건하다.
부호 규약은 tests/test_mediapipe_classify.py 8조합(좌우×거울×등/바닥)으로 고정.

출력 계약: infer(frame) -> list[Detection], class_name ∈ {back_of_hand, palm}.
class_map(back_of_hand -> dorsum)을 거쳐 gesture_filter의 select 판정에 쓰인다.

랜드마크 번호(MediaPipe Hands 규격): 0=손목, 5=검지 MCP, 9=중지 MCP, 17=새끼 MCP.
MCP = 손가락 뿌리 관절 — 손바닥 몸통에 있어 손가락 유무와 무관하게 추정된다.
"""
import math
import time

from src.inference.detector import Detection
from src.utils.logger import get_logger

logger = get_logger("inference")

# 랜드마크 인덱스 — MediaPipe Hands 고정 규격
LM_WRIST = 0
LM_INDEX_MCP = 5
LM_MIDDLE_MCP = 9
LM_PINKY_MCP = 17

CLASS_IDS = {"back_of_hand": 0, "palm": 1}

OPPOSITE_SIDE = {"left": "right", "right": "left"}


def user_side_from_label(label, is_mirror, flip_handedness):
    """MediaPipe handedness 라벨 -> 사용자 기준 좌/우 (없거나 모르면 None).

    공식 문서는 "거울(셀피) 입력 가정" 라벨이라고 하지만, Tasks API에서 라벨이
    반대로 나오는 사례가 보고되어 있다(google-ai-edge/mediapipe#4724 —
    0.10.35 윈도우에서도 반대로 실측됨, 2026-07-10). 그래서 문서 기준 매핑 위에
    flip_handedness(config)를 두었다 — 데모 화면 손 박스의 L/R 표시가 실제와
    뒤집혀 보이면 이 값을 반전하면 된다.
    """
    if label not in OPPOSITE_SIDE:
        return None
    side = label if is_mirror else OPPOSITE_SIDE[label]  # 문서 기준 매핑
    return OPPOSITE_SIDE[side] if flip_handedness else side


def _dist(a, b):
    return math.dist((a[0], a[1]), (b[0], b[1]))


def _hand_size(landmarks):
    """손 크기 기준값 — 손목~중지 MCP 거리 (회전·거리 불변 정규화용)."""
    return max(_dist(landmarks[LM_WRIST], landmarks[LM_MIDDLE_MCP]), 1e-6)


def classify_hand_orientation(landmarks, user_side, is_mirror, back_facing_threshold):
    """21개 랜드마크 -> "back_of_hand" | "palm" | None (순수 함수 — 단위 테스트 가능).

    user_side: 사용자 기준 손 좌/우 (user_side_from_label 결과). None이면 판정 불가 —
    좌/우를 모르면 외적 부호의 기준을 정할 수 없다.
    |외적|을 손 크기 제곱으로 정규화해 임계 미만(옆면·뒤집는 도중)은 None.
    """
    if user_side not in ("left", "right"):
        return None
    wrist = landmarks[LM_WRIST]
    ax = landmarks[LM_INDEX_MCP][0] - wrist[0]
    ay = landmarks[LM_INDEX_MCP][1] - wrist[1]
    bx = landmarks[LM_PINKY_MCP][0] - wrist[0]
    by = landmarks[LM_PINKY_MCP][1] - wrist[1]
    cross_norm = (ax * by - ay * bx) / _hand_size(landmarks) ** 2
    if abs(cross_norm) < back_facing_threshold:
        return None

    # 부호 규약: 손바닥을 카메라로 향한 오른손은 엄지·검지가 화면 오른쪽에 온다
    # (해부학 — 오른손 '정지' 자세를 정면에서 보면 엄지가 보는 사람 오른쪽).
    # 이때 외적이 음수 → 비거울 오른손은 손등=양수, 거울 프레임은 x반전으로 부호가 뒤집힌다
    back_sign = -1.0 if (user_side == "right") == is_mirror else 1.0
    return "back_of_hand" if cross_norm * back_sign > 0 else "palm"


def _bbox_from_landmarks(landmarks, frame_shape, pad_ratio):
    """랜드마크를 감싸는 픽셀 박스 — person_lock의 손목 귀속 거리 계산에 쓰인다."""
    h_px, w_px = frame_shape[:2]
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    pad = max(max(xs) - min(xs), max(ys) - min(ys)) * pad_ratio
    return (
        max(0.0, min(xs) - pad), max(0.0, min(ys) - pad),
        min(w_px - 1.0, max(xs) + pad), min(h_px - 1.0, max(ys) + pad),
    )


class MediaPipeGestureDetector:
    """MediaPipe 손 랜드마크 기반 손등/손바닥 검출기. infer(frame) -> list[Detection]."""

    def __init__(self, config):
        import mediapipe as mp  # 무거운 의존 — 생성 시점 지연 임포트
        from mediapipe.tasks.python import BaseOptions, vision

        mp_cfg = config["model"]["mediapipe"]
        self._back_facing_threshold = mp_cfg["back_facing_threshold"]
        self._bbox_pad_ratio = mp_cfg["bbox_pad_ratio"]
        self._conf_threshold = config["detect"]["conf_threshold"]
        self._is_mirror = config["camera"]["mirror"]
        self._flip_handedness = mp_cfg["flip_handedness"]  # user_side_from_label 참고
        self._mp = mp
        self._clock = time.monotonic
        self._last_timestamp_ms = -1

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=mp_cfg["hand_landmarker_path"]),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=mp_cfg["num_hands"],
            min_hand_detection_confidence=self._conf_threshold,
            min_tracking_confidence=mp_cfg["min_tracking_confidence"],
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        logger.info("제스처 모델 로딩 완료: MediaPipe HandLandmarker (%s)",
                    mp_cfg["hand_landmarker_path"])

    def infer(self, frame):
        """프레임(BGR)에서 손을 찾아 손등/손바닥 방향을 판정한다."""
        h_px, w_px = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=rgb.copy(order="C"))
        # VIDEO 모드는 단조 증가 타임스탬프(ms)를 요구한다 — 추론 루프 단일 스레드 전제
        timestamp_ms = max(int(self._clock() * 1000.0), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            conf = float(handedness[0].score) if handedness else 1.0
            if conf < self._conf_threshold:
                continue
            hand_side = None
            if handedness:
                hand_side = user_side_from_label(
                    handedness[0].category_name.lower(),
                    self._is_mirror,
                    self._flip_handedness,
                )
            points_px = [(lm.x * w_px, lm.y * h_px) for lm in hand_landmarks]
            class_name = classify_hand_orientation(
                points_px, hand_side, self._is_mirror, self._back_facing_threshold
            )
            if class_name is None:
                continue
            detections.append(
                Detection(
                    class_id=CLASS_IDS[class_name],
                    class_name=class_name,
                    conf=conf,
                    bbox=_bbox_from_landmarks(points_px, frame.shape, self._bbox_pad_ratio),
                    hand_side=hand_side,
                )
            )
        return detections
