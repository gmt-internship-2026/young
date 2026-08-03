"""inference 모듈 — 머리 앵커 검출 (MediaPipe Pose Landmarker/BlazePose, Apache-2.0).

2026-07-31 도입(사용자 결정 — 얼굴 검출 앵커의 가림 한계): 마스크·썬글라스·모자·
색상 조합으로 얼굴 랜드마크가 가려지면 BlazeFace가 실패했다. 포즈는 **몸
실루엣으로 사람을 잡고 머리 위치를 골격의 일부로** 출력하므로 얼굴에 뭘 썼든
무관하다 — 학습·액세서리별 대응이 아예 불필요. 얼굴 검출기(face_detector.py)는
전면 교체·삭제(사용자 결정: 회사 시선추적 모듈과의 얼굴 처리 중복·충돌 회피 —
엔진은 얼굴을 다루지 않는다).

머리 관측 정의: 중심 = 양귀 키포인트(7·8) 중점, 폭 = 귀-귀 거리(≈얼굴 폭) —
구 face_anchor의 "폭 × reach 배수 = 팔 도달" 체계를 숫자 그대로 잇는다.
단일 인물 모델(num_poses=1)은 가장 두드러진(=가까운) 사람을 잡는다 — 앵커
목적과 일치. 귀 키포인트는 가려져도 몸 맥락으로 추정돼 좌표가 나온다.

라이선스: HandLandmarker와 동일 Apache-2.0 — 상업 사용 가능·코드 공개(카피레프트)
의무 없음·제품 화면 표시 의무 없음 (라이선스 B안 유지, №9).
모델 파일: models/weights/pose_landmarker_lite.task (~5MB) — download_weights.py.
※7-29에 제거한 포즈(rtmlib+ONNX Runtime — 별도 의존 2개·다인·~190MB)와 달리
mediapipe 내장이라 의존성 추가가 없다.
"""
import time
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger("inference")

NOSE_IDX = 0
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8


@dataclass
class HeadDetection:
    """머리 1개의 관측 — 앵커 선별(hand_select)에 필요한 최소 정보."""

    center_x_px: float
    center_y_px: float
    width_px: float     # 귀-귀 거리 — 클수록 가깝다 (앵커 선별 기준 + 도달 반경 자)
    conf: float


class HeadDetector:
    """MediaPipe Pose Landmarker 래퍼. infer(frame) -> list[HeadDetection] (0~1개)."""

    def __init__(self, config):
        detector_cfg = config["head_anchor"]
        self._model_path = detector_cfg["model_path"]
        # mediapipe는 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        # model_asset_path가 아니라 buffer로 넘긴다 — 비ASCII 경로 미개봉 버그
        # 우회(2026-08-03, hand_tracker.py와 동일 사유)
        with open(self._model_path, "rb") as weights_file:
            model_bytes = weights_file.read()
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,   # 가장 두드러진(가까운) 1인 — 앵커 목적과 일치
            min_pose_detection_confidence=detector_cfg.get("min_detection_conf", 0.5),
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        # VIDEO 모드는 단조 증가 타임스탬프(ms)가 필수 (hand_tracker와 동일 규약)
        self._start_sec = time.monotonic()
        self._last_timestamp_ms = -1
        logger.info("머리(포즈) 모델 로딩 완료: MediaPipe Pose Landmarker (%s)",
                    self._model_path)

    def infer(self, frame):
        """프레임(BGR·거울 반전 후)의 머리 관측 -> list[HeadDetection].

        거울 보정 불필요 — 머리에는 좌/우 라벨이 없고 좌표는 손과 같은
        (반전 후) 프레임 좌표계라 그대로 비교 가능하다.
        """
        h_px, w_px = frame.shape[:2]
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=frame[:, :, ::-1].copy()
        )
        timestamp_ms = int((time.monotonic() - self._start_sec) * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1   # 단조 증가 보장 (고FPS 보호)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        heads = []
        for pose_landmarks in result.pose_landmarks:
            left_ear = pose_landmarks[LEFT_EAR_IDX]
            right_ear = pose_landmarks[RIGHT_EAR_IDX]
            width_px = (((left_ear.x - right_ear.x) * w_px) ** 2
                        + ((left_ear.y - right_ear.y) * h_px) ** 2) ** 0.5
            if width_px <= 0.0:
                continue
            # 신뢰도 = 머리 키포인트 가시성 평균 — 가려져도 몸 맥락 추정값이 나오며
            # visibility가 낮아질 뿐이다 (마스크·모자는 좌표 정확도에 영향 적음)
            conf = (pose_landmarks[NOSE_IDX].visibility
                    + left_ear.visibility + right_ear.visibility) / 3.0
            heads.append(HeadDetection(
                center_x_px=(left_ear.x + right_ear.x) / 2.0 * w_px,
                center_y_px=(left_ear.y + right_ear.y) / 2.0 * h_px,
                width_px=width_px,
                conf=float(conf),
            ))
        return heads
