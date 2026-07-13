"""inference 모듈 — MediaPipe 손 랜드마크로 제스처를 검출한다 (라이선스 C안).

2026-07-10 교체(라이선스 C안): HaGRID YOLOv10 ONNX는 AGPL(ultralytics/YOLOv10)
학습·변환 계열이라 비공개 상업 배포에 위험 — MediaPipe Hand Landmarker
(Apache-2.0, 구글 배포 .task 모델)로 교체했다. 저작자 표시 의무도 없다.

동작 방식: 손 21 랜드마크를 추정한 뒤, 손 크기로 정규화한 기하 규칙로
제스처를 판정한다 — 별도 학습 0회, 판정 규칙은 단위 테스트로 검증된다
(tests/test_mediapipe_classify.py).

출력 계약은 detector.GestureDetector와 동일: infer(frame) -> list[Detection].
class_name은 기존 class_map 키(fist/palm/ok/one/like)를 그대로 내보내므로
person_lock·gesture_filter는 수정 없이 동작한다.

랜드마크 번호(MediaPipe Hands 규격): 0=손목, 4=엄지 끝, 8=검지 끝,
12=중지 끝, 16=약지 끝, 20=새끼 끝. 각 손가락은 MCP-PIP-DIP-TIP 순.
"""
import math
import time

from src.inference.detector import Detection
from src.utils.logger import get_logger

logger = get_logger("inference")

# 랜드마크 인덱스 — MediaPipe Hands 고정 규격
LM_WRIST = 0
LM_THUMB_IP, LM_THUMB_TIP = 3, 4
LM_INDEX_MCP = 5
LM_MIDDLE_MCP = 9
# (검지, 중지, 약지, 새끼): (PIP, TIP)
FINGER_PIP_TIP = ((6, 8), (10, 12), (14, 16), (18, 20))

CLASS_IDS = {"fist": 0, "palm": 1, "ok": 2, "one": 3, "like": 4}

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


def _extended_fingers(landmarks, extended_ratio):
    """(검지, 중지, 약지, 새끼) 각각의 펴짐 여부 — TIP이 PIP보다 손목에서 충분히 멀면 폄."""
    wrist = landmarks[LM_WRIST]
    return tuple(
        _dist(landmarks[tip], wrist) > _dist(landmarks[pip], wrist) * extended_ratio
        for pip, tip in FINGER_PIP_TIP
    )


def classify_hand_landmarks(landmarks, extended_ratio, ok_pinch_ratio):
    """21개 랜드마크 [(x, y), ...] -> 제스처 이름(class_map 키) 또는 None.

    순수 함수 — mediapipe 없이 단위 테스트 가능. 좌표 단위는 무엇이든
    (픽셀/정규화) 일관되기만 하면 된다 (내부에서 손 크기로 정규화).
    """
    size = _hand_size(landmarks)
    index_ext, middle_ext, ring_ext, pinky_ext = _extended_fingers(landmarks, extended_ratio)
    pinch = _dist(landmarks[LM_THUMB_TIP], landmarks[8]) / size
    thumb_ext = (
        _dist(landmarks[LM_THUMB_TIP], landmarks[LM_WRIST])
        > _dist(landmarks[LM_THUMB_IP], landmarks[LM_WRIST]) * extended_ratio
    )

    # OK: 엄지-검지 끝 맞닿음 + 나머지 세 손가락 폄 (검지는 굽어 있어 palm과 배타)
    if pinch < ok_pinch_ratio and middle_ext and ring_ext and pinky_ext:
        return "ok"
    # palm(손바닥): 네 손가락 모두 폄
    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "palm"
    # one(포인트): 검지만 폄 — 레거시 point로 매핑된다
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and pinch >= ok_pinch_ratio:
        return "one"
    # like(따봉): 엄지만 폄 + 엄지 끝이 손목보다 위 (화면 y는 아래로 증가)
    if (
        thumb_ext
        and not any((index_ext, middle_ext, ring_ext, pinky_ext))
        and landmarks[LM_THUMB_TIP][1] < landmarks[LM_WRIST][1]
    ):
        return "like"
    # fist(주먹): 네 손가락 모두 굽힘
    if not any((index_ext, middle_ext, ring_ext, pinky_ext)):
        return "fist"
    return None


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
    """MediaPipe 손 랜드마크 기반 제스처 검출기. infer(frame) -> list[Detection]."""

    def __init__(self, config):
        import mediapipe as mp  # 무거운 의존 — onnx 엔진 선택 시 임포트하지 않는다
        from mediapipe.tasks.python import BaseOptions, vision

        mp_cfg = config["model"]["mediapipe"]
        self._extended_ratio = mp_cfg["finger_extended_ratio"]
        self._ok_pinch_ratio = mp_cfg["ok_pinch_ratio"]
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
        """프레임(BGR)에서 손을 찾아 기하 규칙으로 제스처를 판정한다."""
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
            points_px = [(lm.x * w_px, lm.y * h_px) for lm in hand_landmarks]
            class_name = classify_hand_landmarks(
                points_px, self._extended_ratio, self._ok_pinch_ratio
            )
            if class_name is None:
                continue
            hand_side = None
            if handedness:
                hand_side = user_side_from_label(
                    handedness[0].category_name.lower(),
                    self._is_mirror,
                    self._flip_handedness,
                )
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
