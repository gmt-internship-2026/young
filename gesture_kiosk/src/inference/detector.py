"""inference 모듈 — 검출 공통 구조(Detection)와 엔진 팩토리, ORT 실행 장치 헬퍼.

2026-07-15 정리: 구 ONNX 제스처 엔진(HaGRID YOLOv10)을 제거했다.
동작 개편(손등/팔등·쓸기)으로 손가락 모양 클래스(fist/palm/ok …)를 내는
엔진은 신규 스펙을 판정할 수 없어 죽은 코드가 됐고, AGPL(ultralytics 계열)
리스크(기획서 9장 №9)도 함께 사라졌다. 제스처 검출은 MediaPipe 손 랜드마크
(detector_mediapipe.py) 단일 엔진이다.

resolve_providers / ensure_cuda_dlls 는 ORT 세션을 여는 다른 모듈
(pose_estimator·arm_side_classifier)이 공용으로 쓴다.
"""
import sys
from dataclasses import dataclass

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


@dataclass
class Detection:
    """검출 결과 1건 (기획서 4.6 공통 데이터 구조)."""

    class_id: int
    class_name: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) 좌상단·우하단 픽셀 좌표 (원본 프레임 기준)
    hand_side: str = None  # "left"|"right" — 검출기가 손 좌/우를 알면 채운다 (사용자 기준).
                           # 있으면 person_lock이 손목 거리 대신 이 값으로 좌/우를 정한다
                           # — 한쪽 팔이 없는 사용자도 인식된다


def create_gesture_detector(config):
    """제스처(손등/손바닥) 검출기를 만든다 — MediaPipe 손 랜드마크 (라이선스 C안)."""
    from src.inference.detector_mediapipe import MediaPipeGestureDetector  # 지연 임포트

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
