"""inference 모듈 — 사람 포즈(RTMPose)를 추론해 얼굴·어깨·손목 키포인트를 얻는다.

2026-07-15 2차 개편으로 쓸기(손목 궤적)·사용자 잠금(얼굴)의 유일한 추론 모델이 됐다
(손 검출 MediaPipe·팔등 CNN 제거). 2026-07-16: 선택(select)이 손가락 인식으로
바뀌면서 손 랜드마크는 별도 모델(src/inference/hand_estimator.py)이 다시 맡는다 —
이 모듈은 여전히 쓸기·잠금 전용이라 포즈 키포인트만 다룬다.

2026-07-11 교체(라이선스 B안): ultralytics yolo11n-pose(AGPL-3.0)를 제거하고
rtmlib(Apache-2.0, RTMPose 계열 + ONNX Runtime)로 바꿨다.

모델 파일은 첫 실행 때 자동으로 내려받아 캐시(~/.cache/rtmlib)에 둔다 —
내부망 반입 시에는 make_offline_bundle.bat이 이 캐시를 함께 담는다.

키포인트 번호는 COCO 17 규격이다 (0=코, 1·2=눈, 3·4=귀, 5·6=어깨, 9·10=손목).
주의: 이 라벨은 "화면에 보이는 사람" 기준의 해부학적 좌/우다. 거울 반전된
프레임에서는 사용자의 실제 좌/우와 반대가 되며, 그 보정은 person_lock이 담당한다.

2026-07-16 성능 최적화(검출 스킵 패스트패스): 이 CPU 실측상 rtmlib Body의 검출
단계(YOLOX-tiny)만으로 ~89ms가 들고, 이미 아는 bbox로 포즈만 재추정하면 ~22ms —
5배 이상 차이난다(RTMO 단일모델 시도는 오히려 3 FPS로 더 느려 폐기, det_input_size는
ONNX 고정 입력이라 축소 불가 — 둘 다 실측 후 기각). person_lock이 사람을 잠근 뒤에는
`infer_at_bbox()`로 검출을 건너뛰고, `realtime_loop`가 주기적으로만(redetect_interval_frames)
`infer()`(검출+포즈)를 다시 돌려 위치를 재동기화한다.
"""
import sys
from dataclasses import dataclass, field

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# COCO 17 키포인트 인덱스 (RTMPose body 계열 출력 순서)
KPT_NOSE = 0
KPT_HEAD_INDICES = (0, 1, 2, 3, 4)  # 코·양눈·양귀 — 얼굴 영역 추정에 사용
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10


def ensure_cuda_dlls():
    """윈도우: onnxruntime CUDA는 torch(cu128)가 등록하는 CUDA DLL 경로에 의존한다.

    onnxruntime-gpu가 설치되면 CUDAExecutionProvider는 항상 목록에 보이지만
    DLL 로드는 세션 생성 시점에 일어나므로(실패 시 조용히 CPU 폴백), CUDA를
    쓰려면 세션을 만들기 전에 반드시 torch를 먼저 임포트해 둬야 한다.
    (2026-07-15 2차: 구 detector.py 삭제로 이 모듈로 옮겨 왔다)"""
    if sys.platform.startswith("win"):
        try:
            import torch  # noqa: F401 — DLL 경로 등록 부수효과만 목적
        except ImportError:
            pass


@dataclass
class PersonPose:
    """사람 1명의 포즈 추정 결과 (기획서 4.6 공통 데이터 구조 스타일)."""

    bbox: tuple                 # (x1, y1, x2, y2) 픽셀 좌표 — 키포인트 묶음 기반
    conf: float
    keypoints: np.ndarray       # shape (17, 3) — (x_px, y_px, conf)
    head_points: list = field(default_factory=list)  # 신뢰도 통과한 머리 키포인트 [(x, y)]

    def keypoint(self, index, min_conf):
        """키포인트 신뢰도가 통과하면 (x_px, y_px), 아니면 None (손목·팔꿈치 공용)."""
        x, y, conf = self.keypoints[index]
        if conf < min_conf:
            return None
        return float(x), float(y)


def _resolve_device(device):
    """auto -> onnxruntime에 CUDA가 있으면 cuda, 없으면 cpu."""
    if device == "cpu":
        return device
    # rtmlib의 ORT 세션도 torch의 CUDA DLL 경로가 있어야 GPU로 열린다 —
    # 없으면 조용히 CPU로 폴백해 30 FPS가 무너진다 (2026-07-10 실측: 10 FPS)
    ensure_cuda_dlls()
    if device != "auto":
        return device
    import onnxruntime as ort

    return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"


def _bbox_from_keypoints(keypoints, kpt_conf, frame_shape, bbox_pad_ratio):
    """신뢰도 통과 키포인트를 감싸는 박스. 통과점이 없으면 None.

    bbox_pad_ratio: person_lock.keypoint_bbox_pad_ratio(config) — 손을 옆으로 뻗을 때
    손목 키포인트 신뢰도가 떨어져 박스에서 빠지면 이 박스로 크롭하는 손 인식이
    손을 놓친다(2026-07-23 실기 리포트 — "오른쪽으로 뻗으면 잘 안 잡힘"). 값을
    키우면 신뢰도 통과 키포인트 묶음보다 더 넉넉하게 박스를 잡아 이런 경우를 줄인다.
    """
    valid = keypoints[keypoints[:, 2] >= kpt_conf]
    if len(valid) == 0:
        return None
    x1, y1 = valid[:, 0].min(), valid[:, 1].min()
    x2, y2 = valid[:, 0].max(), valid[:, 1].max()
    pad = max(x2 - x1, y2 - y1, 20.0) * bbox_pad_ratio
    h_px, w_px = frame_shape[:2]
    return (
        max(0.0, float(x1 - pad)), max(0.0, float(y1 - pad)),
        min(w_px - 1.0, float(x2 + pad)), min(h_px - 1.0, float(y2 + pad)),
    )


def pad_bbox(bbox, pad_ratio, frame_width_px, frame_height_px):
    """bbox(x1,y1,x2,y2)를 각 변 기준 pad_ratio만큼 넓히고 프레임 경계로 clamp한다.

    검출 스킵 패스트패스(infer_at_bbox) 입력용 — 잠금 직전 프레임의 bbox가
    이번 프레임에서도 사람을 덮도록 여유를 준다. rtmlib 없이 단위 테스트 가능한
    순수 함수로 둔다(tests/test_pose_estimator.py).
    """
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    return (
        max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
        min(float(frame_width_px - 1), x2 + pad_x),
        min(float(frame_height_px - 1), y2 + pad_y),
    )


def _persons_from_keypoints(keypoints_xy, scores, frame_shape, kpt_conf, bbox_pad_ratio):
    """(N,17,2)+(N,17) 원시 출력 -> list[PersonPose] (infer/infer_at_bbox 공용)."""
    persons = []
    for xy, score in zip(keypoints_xy, scores):
        keypoints = np.concatenate([xy, score[:, None]], axis=1).astype(np.float32)
        bbox = _bbox_from_keypoints(keypoints, kpt_conf, frame_shape, bbox_pad_ratio)
        if bbox is None:
            continue
        head_points = [
            (float(keypoints[i][0]), float(keypoints[i][1]))
            for i in KPT_HEAD_INDICES
            if keypoints[i][2] >= kpt_conf
        ]
        persons.append(
            PersonPose(
                bbox=bbox, conf=float(score.mean()), keypoints=keypoints, head_points=head_points
            )
        )
    return persons


class PoseEstimator:
    """RTMPose 포즈 추정기. infer(frame) -> list[PersonPose] (검출+포즈),
    infer_at_bbox(frame, bbox) -> list[PersonPose] (검출 생략, 추적용)."""

    def __init__(self, config):
        from rtmlib import Body  # 무거운 의존 — person_lock을 끈 환경에선 임포트하지 않는다

        model = config["model"]
        device = _resolve_device(model["device"])
        self._kpt_conf_threshold = config["person_lock"]["kpt_conf_threshold"]
        self._bbox_pad_ratio = config["person_lock"]["keypoint_bbox_pad_ratio"]
        # mode: lightweight(빠름) | balanced(기본) | performance(정확) — 첫 실행 시 자동 다운로드
        self._body = Body(mode=model["pose_mode"], backend="onnxruntime", device=device)
        logger.info("포즈 모델 로딩 완료: rtmlib Body(mode=%s, device=%s)", model["pose_mode"], device)

    def infer(self, frame):
        """프레임 전체에서 사람을 검출+추정한다 (검출기 포함 — 무겁다).

        사용자를 찾는 중(미잠금)이거나 재동기화 시점에만 쓴다. 이미 잠근 사용자를
        매 프레임 쫓을 때는 infer_at_bbox()가 검출을 건너뛰어 훨씬 빠르다.
        """
        keypoints_xy, scores = self._body(frame)  # (N,17,2), (N,17)
        return _persons_from_keypoints(
            keypoints_xy, scores, frame.shape, self._kpt_conf_threshold, self._bbox_pad_ratio
        )

    def infer_at_bbox(self, frame, bbox):
        """bbox 안에서만 포즈를 추정한다 (검출기 생략 — 5배 이상 빠름, 2026-07-16).

        person_lock이 잠근 사용자를 계속 추적할 때 쓴다. bbox는 이전 프레임
        위치를 pad_bbox()로 넉넉히 넓혀서 넘겨야 이번 프레임의 움직임을 놓치지
        않는다. 결과가 비면(사람이 패딩 밖으로 나감 등) 호출부가 다음 재동기화
        프레임까지 기존 잠금을 유지하거나 해제한다(person_lock 로직 그대로).
        """
        bboxes = np.array([bbox], dtype=np.float32)
        keypoints_xy, scores = self._body.pose_model(frame, bboxes=bboxes)
        return _persons_from_keypoints(
            keypoints_xy, scores, frame.shape, self._kpt_conf_threshold, self._bbox_pad_ratio
        )
