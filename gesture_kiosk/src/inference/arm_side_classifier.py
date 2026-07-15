"""inference 모듈 — 전완(팔뚝) 등쪽/안쪽 분류로 '팔등 보이기' 선택을 지원한다.

2026-07-15 신설(범용 설계): 손·손가락이 없는 사용자는 손 랜드마크 검출 자체가
안 되므로, 포즈의 팔꿈치~손목 구간을 잘라 자체 학습 소형 CNN(ONNX)으로
등쪽(팔등)이 카메라를 향하는지 분류한다. 결과는 손등 판정과 같은
'dorsum' 관측으로 합류한다 — gesture_filter는 출처를 구분하지 않는다.

라이선스: 가중치는 scripts/collect_arm_side.py 로 모은 자체 데이터로
scripts/train_arm_side.py 가 사전학습 없이 처음부터 학습한 자산이다 —
제3자 모델이 없어 상업 사용·비공개 배포에 아무 의무가 없다 (기획서 9장 №9).

모델 파일이 없으면 자동 비활성(경고 1회) — 손등 판정만으로 동작한다.
"""
import math
import os

import cv2
import numpy as np

from src.postprocess.gesture_filter import DORSUM
from src.postprocess.person_lock import HandObservation
from src.utils.logger import get_logger

logger = get_logger("inference")

MIN_FOREARM_LEN_PX = 40.0  # 이보다 짧으면 팔이 카메라 축과 겹친 것 — 크롭이 의미 없다


def forearm_crop_corners(elbow, wrist, crop_scale):
    """팔꿈치→손목 축을 세로(+y)로 세운 정사각 크롭의 세 꼭짓점 (순수 함수).

    반환: [좌상, 우상, 좌하] 픽셀 좌표 (cv2.getAffineTransform 입력 순서).
    전완이 너무 짧으면(팔이 카메라를 향해 접힘) None.
    회전 정규화로 촬영 각도 영향을 없애고, 수집(collect_arm_side)과 추론이
    같은 함수를 써서 학습·추론 입력 분포를 일치시킨다.
    """
    axis_x = wrist[0] - elbow[0]
    axis_y = wrist[1] - elbow[1]
    length_px = math.hypot(axis_x, axis_y)
    if length_px < MIN_FOREARM_LEN_PX:
        return None
    ux, uy = axis_x / length_px, axis_y / length_px   # 크롭 +y (팔꿈치→손목)
    px, py = uy, -ux                                   # 크롭 +x — 좌우 반전 없는 회전이 되는 방향
    side_px = crop_scale * length_px
    center_x = (elbow[0] + wrist[0]) / 2.0
    center_y = (elbow[1] + wrist[1]) / 2.0
    top_left = (center_x - (px + ux) * side_px / 2.0, center_y - (py + uy) * side_px / 2.0)
    top_right = (top_left[0] + px * side_px, top_left[1] + py * side_px)
    bottom_left = (top_left[0] + ux * side_px, top_left[1] + uy * side_px)
    return [top_left, top_right, bottom_left]


def crop_forearm(frame, elbow, wrist, crop_scale, out_size_px):
    """전완 정사각 크롭(회전 정규화) — 실패 시 None."""
    corners = forearm_crop_corners(elbow, wrist, crop_scale)
    if corners is None:
        return None
    src_points = np.float32(corners)
    dst_points = np.float32([(0, 0), (out_size_px, 0), (0, out_size_px)])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    return cv2.warpAffine(frame, matrix, (out_size_px, out_size_px))


def preprocess_crop(crop):
    """BGR 크롭 -> (1, 3, H, W) float32 [0,1] — train_arm_side.py와 동일 규약."""
    rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


class ArmSideClassifier:
    """전완 등쪽 분류기. observe(frame, arm_points, taken_sides) -> list[HandObservation]."""

    def __init__(self, config):
        arm_cfg = config["model"]["arm_side"]
        self._crop_scale = arm_cfg["crop_scale"]
        self._input_size_px = arm_cfg["input_size_px"]
        self._dorsal_prob_threshold = arm_cfg["dorsal_prob_threshold"]
        self._session = None

        onnx_path = arm_cfg["onnx_path"]
        if not os.path.exists(onnx_path):
            logger.warning(
                "팔등 분류 모델 없음(%s) — 손등 판정만 동작. 데이터 수집: "
                "scripts/collect_arm_side.py → 학습: scripts/train_arm_side.py", onnx_path
            )
            return
        import onnxruntime as ort  # 무거운 의존 — 모델이 있을 때만 임포트
        from src.inference.detector import resolve_providers

        providers = resolve_providers(config["model"]["device"], False, "")
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        logger.info("팔등 분류 모델 로딩 완료: %s (providers=%s)",
                    onnx_path, self._session.get_providers())

    @property
    def enabled(self):
        return self._session is not None

    def infer_dorsal_prob(self, frame, elbow, wrist):
        """전완 1개의 등쪽 확률(0~1). 크롭 불가(팔이 카메라 방향)면 None."""
        crop = crop_forearm(frame, elbow, wrist, self._crop_scale, self._input_size_px)
        if crop is None:
            return None
        (output,) = self._session.run(None, {self._input_name: preprocess_crop(crop)})
        return float(np.asarray(output).reshape(-1)[0])

    def observe(self, frame, arm_points, taken_sides):
        """손 관측이 없는 쪽 팔만 분류해 dorsum 관측으로 만든다.

        arm_points: {"left": ((elbow), (wrist)) | None, ...} — 사용자 기준 픽셀 좌표.
        taken_sides: 이번 프레임에 손 관측이 이미 있는 쪽 — 손 랜드마크 판정이
        더 정확하므로 그쪽을 우선하고 팔 분류는 건너뛴다.
        """
        if self._session is None:
            return []
        h_px, w_px = frame.shape[:2]
        observations = []
        for side, points in arm_points.items():
            if points is None or side in taken_sides:
                continue
            elbow, wrist = points
            prob = self.infer_dorsal_prob(frame, elbow, wrist)
            if prob is None or prob < self._dorsal_prob_threshold:
                continue
            observations.append(
                HandObservation(
                    side=side, gesture=DORSUM, conf=prob,
                    cx_ratio=wrist[0] / w_px, cy_ratio=wrist[1] / h_px,
                )
            )
        return observations
