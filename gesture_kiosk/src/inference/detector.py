"""inference 모듈 — ONNX Runtime으로 제스처를 검출한다 (기획서 2.2).

2026-07-11 교체(라이선스 B안): ultralytics(AGPL-3.0) 실행기를 제거하고
ONNX Runtime(MIT)으로 직접 추론한다. 모델은 HaGRIDv2 YOLOv10n을 ONNX로
변환한 것(scripts/export_onnx.py — 개발 PC에서 1회)이며, YOLOv10은
NMS가 필요 없는 구조라 출력 (1, 300, 6) = [x1, y1, x2, y2, 확신도, 클래스]를
그대로 읽으면 된다.

가속: model.device(auto/cuda/cpu)로 실행 장치를 고르고, model.use_tensorrt를
켜면 TensorRT 실행기(EP)가 첫 실행 때 이 PC 전용 엔진 캐시를 만든다
(별도 빌드 스크립트 불필요 — 구 build_engine.py 대체).
"""
import ast
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from src.utils.logger import get_logger

logger = get_logger("inference")


def ensure_cuda_dlls():
    """윈도우: onnxruntime CUDA는 torch(cu128)가 등록하는 CUDA DLL 경로에 의존한다.

    onnxruntime-gpu가 설치되면 CUDAExecutionProvider는 항상 목록에 보이지만
    DLL 로드는 세션 생성 시점에 일어나므로(실패 시 조용히 CPU 폴백), CUDA를
    쓰려면 세션을 만들기 전에 반드시 torch를 먼저 임포트해 둬야 한다.
    rtmlib(pose_estimator) 등 다른 ORT 세션 생성부도 이 함수를 먼저 부른다."""
    if sys.platform.startswith("win"):
        try:
            import torch  # noqa: F401 — DLL 경로 등록 부수효과만 목적
        except ImportError:
            pass

LETTERBOX_FILL = 114  # 비율 유지 리사이즈 후 여백 색 (YOLO 학습 관례)


@dataclass
class Detection:
    """검출 결과 1건 (기획서 4.6 공통 데이터 구조)."""

    class_id: int
    class_name: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) 좌상단·우하단 픽셀 좌표 (원본 프레임 기준)
    hand_side: str = None  # "left"|"right" — 검출기가 손 좌/우를 알면 채운다 (사용자 기준).
                           # 있으면 person_lock이 손목 거리 대신 이 값으로 좌/우를 정한다
                           # — 한쪽 팔이 없는 사용자도 인식된다. ONNX 엔진은 None


def create_gesture_detector(config):
    """config model.gesture_engine에 따라 제스처 검출기를 만든다.

    - mediapipe(기본, 라이선스 C안): MediaPipe 손 랜드마크 + 기하 판정 (Apache-2.0)
    - onnx: 구 HaGRID YOLOv10 — AGPL(ultralytics) 리스크로 납품 금지, 비교 시험용만
    """
    engine = config["model"].get("gesture_engine", "mediapipe")
    if engine == "onnx":
        logger.warning("gesture_engine=onnx — HaGRID YOLOv10은 AGPL 리스크로 납품 금지 (README 라이선스 절)")
        return GestureDetector(config)
    from src.inference.detector_mediapipe import MediaPipeGestureDetector  # 순환 임포트 방지

    return MediaPipeGestureDetector(config)


def resolve_providers(device, use_tensorrt, cache_dir):
    """실행 장치 설정 -> onnxruntime 프로바이더 목록 (항상 CPU 폴백 포함)."""
    if device in ("auto", "cuda"):
        ensure_cuda_dlls()
    available = ort.get_available_providers()
    providers = []
    if use_tensorrt and "TensorrtExecutionProvider" in available:
        providers.append(("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_fp16_enable": True,
        }))
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def letterbox(frame, target_px):
    """비율을 유지하며 target 정사각형에 맞추고 여백을 채운다.

    반환: (가공된 이미지, 배율, (x 여백, y 여백)) — 좌표 역변환에 쓴다.
    """
    height, width = frame.shape[:2]
    scale = min(target_px / width, target_px / height)
    new_w, new_h = round(width * scale), round(height * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    pad_x = (target_px - new_w) // 2
    pad_y = (target_px - new_h) // 2
    canvas = np.full((target_px, target_px, 3), LETTERBOX_FILL, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def unletterbox_box(box, scale, pad, frame_shape):
    """모델 좌표(640 기준) -> 원본 프레임 픽셀 좌표."""
    x1, y1, x2, y2 = box
    pad_x, pad_y = pad
    height, width = frame_shape[:2]
    x1 = min(max((x1 - pad_x) / scale, 0.0), width - 1.0)
    x2 = min(max((x2 - pad_x) / scale, 0.0), width - 1.0)
    y1 = min(max((y1 - pad_y) / scale, 0.0), height - 1.0)
    y2 = min(max((y2 - pad_y) / scale, 0.0), height - 1.0)
    return (x1, y1, x2, y2)


class GestureDetector:
    """ONNX 제스처 검출기. infer(frame) -> list[Detection]."""

    def __init__(self, config):
        model = config["model"]
        onnx_path = model["gesture_onnx_path"]
        cache_dir = os.path.join(os.path.dirname(onnx_path), "trt_cache")
        providers = resolve_providers(model["device"], model["use_tensorrt"], cache_dir)

        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._input_size_px = model["input_size_px"]
        self._conf_threshold = config["detect"]["conf_threshold"]
        self._class_names = self._load_class_names()

        # 워밍업 — 첫 프레임 지연(세션 초기화·엔진 캐시)을 시작 시점으로 당긴다
        dummy = np.zeros((self._input_size_px, self._input_size_px, 3), dtype=np.uint8)
        self.infer(dummy)
        logger.info("제스처 모델 로딩 완료: %s (providers=%s)",
                    os.path.basename(onnx_path), self._session.get_providers())

    def _load_class_names(self):
        """클래스 번호 -> 이름 표. 변환 시 ONNX 안에 넣어 둔 목록(names)을 읽는다."""
        meta = self._session.get_modelmeta().custom_metadata_map
        names_text = meta.get("names")
        if not names_text:
            raise RuntimeError(
                "ONNX에 클래스 목록(names)이 없습니다 — scripts/export_onnx.py로 다시 변환하세요"
            )
        return {int(k): v for k, v in ast.literal_eval(names_text).items()}

    def infer(self, frame):
        """프레임(BGR)에서 제스처를 검출한다."""
        canvas, scale, pad = letterbox(frame, self._input_size_px)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        (output,) = self._session.run(None, {self._input_name: blob})

        detections = []
        for x1, y1, x2, y2, conf, class_id in output[0]:
            if conf < self._conf_threshold:
                continue  # 출력은 확신도 내림차순 — 이후 행은 전부 미달이지만 안전하게 계속 확인
            class_id = int(class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, f"cls_{class_id}"),
                    conf=float(conf),
                    bbox=unletterbox_box((x1, y1, x2, y2), scale, pad, frame.shape),
                )
            )
        return detections
