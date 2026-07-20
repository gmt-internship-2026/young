"""inference 모듈 — 사람 포즈(RTMPose)를 추론해 어깨·손목·손끝 키포인트를 얻는다.

2026-07-15 2차 개편으로 **유일한 추론 모델**이 됐다 — 쓸기(손목 궤적)와
사용자 잠금(몸 박스 — 2026-07-20 얼굴 기반 제거)이 전부 이 포즈 키포인트로
판정된다 (손 검출 MediaPipe·팔등 CNN 제거).

2026-07-11 교체(라이선스 B안): ultralytics yolo11n-pose(AGPL-3.0)를 제거하고
rtmlib(Apache-2.0, RTMPose 계열 + ONNX Runtime)로 바꿨다.

모델 파일은 첫 실행 때 자동으로 내려받아 캐시(~/.cache/rtmlib)에 둔다 —
내부망 반입 시에는 make_offline_bundle.bat이 이 캐시를 함께 담는다.

키포인트 번호는 COCO 17 규격이다 (0=코, 1·2=눈, 3·4=귀, 5·6=어깨, 9·10=손목).
pose_engine=wholebody(2026-07-16 손끝 추적)면 COCO-WholeBody 133 규격 — 앞 17개
번호는 COCO 17과 동일해 기존 판정 코드가 그대로 돌고, 91~132가 양손 21점씩이다
(손끝 인덱스는 person_lock 참고 — 유일한 사용처).
주의: 이 라벨은 "화면에 보이는 사람" 기준의 해부학적 좌/우다. 거울 반전된
프레임에서는 사용자의 실제 좌/우와 반대가 되며, 그 보정은 person_lock이 담당한다.
"""
import sys
from dataclasses import dataclass

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("inference")

# COCO 17 키포인트 인덱스 (RTMPose body 계열 출력 순서)
# (2026-07-20: 머리 5점 묶음 KPT_HEAD_INDICES는 얼굴 잠금 제거로 삭제 — person_lock 주석 참고)
KPT_NOSE = 0
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

BBOX_PAD_RATIO = 0.10  # 키포인트 묶음 -> 사람 박스로 넓히는 패딩 (추적용)


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
    keypoints: np.ndarray       # shape (17|133, 3) — (x_px, y_px, conf). 133=wholebody 엔진

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


def _bbox_from_keypoints(keypoints, kpt_conf, frame_shape):
    """신뢰도 통과 키포인트를 감싸는 박스. 통과점이 없으면 None."""
    valid = keypoints[keypoints[:, 2] >= kpt_conf]
    if len(valid) == 0:
        return None
    x1, y1 = valid[:, 0].min(), valid[:, 1].min()
    x2, y2 = valid[:, 0].max(), valid[:, 1].max()
    pad = max(x2 - x1, y2 - y1, 20.0) * BBOX_PAD_RATIO
    h_px, w_px = frame_shape[:2]
    return (
        max(0.0, float(x1 - pad)), max(0.0, float(y1 - pad)),
        min(w_px - 1.0, float(x2 + pad)), min(h_px - 1.0, float(y2 + pad)),
    )


class PoseEstimator:
    """RTMPose 포즈 추정기. infer(frame) -> list[PersonPose]."""

    def __init__(self, config):
        model = config["model"]
        device = _resolve_device(model["device"])
        self._kpt_conf_threshold = config["person_lock"]["kpt_conf_threshold"]
        engine = model.get("pose_engine", "body")
        # mode: lightweight(빠름) | balanced(기본) | performance(정확) — 첫 실행 시 자동 다운로드.
        # rtmlib은 무거운 의존이라 사용 시점에 임포트한다 (단위 테스트가 가벼워지게)
        if engine == "wholebody":
            # 전신 133 키포인트 — 손끝 추적(2026-07-16). body보다 무겁다: 저사양 기기는 body 유지
            from rtmlib import Wholebody as solution
        else:
            from rtmlib import Body as solution

        # det_interval_frames(2026-07-20 추론 부담 절감): rtmlib의 2단계(사람 검출 YOLOX
        # + 포즈 RTMPose) 중 검출을 N프레임에 1회만 돌리고, 사이에는 직전 포즈로 만든
        # 박스를 재사용한다(PoseTracker). 사람이 안 보이면 infer()가 reset을 걸어
        # 다음 프레임에 검출을 강제하므로 신규 접근자 포착 지연은 1프레임뿐이다.
        det_interval_frames = int(model.get("det_interval_frames", 1))
        if det_interval_frames > 1:
            from rtmlib import PoseTracker

            # tracking=False — 사람 식별(잠금)은 person_lock 담당이라 ID 부여가 불필요
            self._pose = PoseTracker(
                solution, det_frequency=det_interval_frames, tracking=False,
                mode=model["pose_mode"], backend="onnxruntime", device=device,
            )
            self._tracker = self._pose
        else:
            self._pose = solution(mode=model["pose_mode"], backend="onnxruntime", device=device)
            self._tracker = None
        logger.info(
            "포즈 모델 로딩 완료: rtmlib %s(mode=%s, device=%s, det_interval=%d프레임)",
            "Wholebody" if engine == "wholebody" else "Body",
            model["pose_mode"], device, det_interval_frames,
        )

    def infer(self, frame):
        """프레임에서 사람 포즈를 추정한다."""
        keypoints_xy, scores = self._pose(frame)  # (N,17|133,2), (N,17|133)
        persons = []
        for xy, score in zip(keypoints_xy, scores):
            keypoints = np.concatenate([xy, score[:, None]], axis=1).astype(np.float32)
            bbox = _bbox_from_keypoints(keypoints, self._kpt_conf_threshold, frame.shape)
            if bbox is None:
                continue
            persons.append(
                PersonPose(
                    bbox=bbox,
                    # 몸 17점만 평균 — wholebody의 얼굴 68점이 사람 신뢰도를 지배하지 않게
                    conf=float(score[:17].mean()),
                    keypoints=keypoints,
                )
            )
        if not persons and self._tracker is not None:
            # 화면에 사람이 없다 — 묵은 박스를 버리고 다음 프레임에 검출을 강제한다.
            # (검출 간격 대기 중 새 사용자가 와도 즉시 포착되게 하는 안전장치)
            self._tracker.reset()
        return persons
