"""INT8 양자화 — 검출(YOLOX)·포즈(RTMPose) 모델을 8비트 정수로 변환한다 (2026-07-22).

저사양 CPU(라즈베리파이 5 등 — Cortex-A76은 INT8 전용 명령 보유)에서 추론을
1.5~2배 가속하기 위한 도구. **배포 기기에서 직접 실행**하는 것을 권장한다 —
보정(calibration) 샘플을 그 기기의 카메라가 실제로 보는 화면에서 뽑아야
입력 분포가 일치해 정확도 손실이 최소가 된다.

사용법 (프로젝트 루트, 사람이 카메라 앞에 서서):
    python scripts/quantize_models.py --source camera            # 보정 100프레임 수집
    python scripts/quantize_models.py --source dummy             # 기계 검증용(정확도 무의미)

필요 패키지: onnx (pip install onnx — 양자화 때만 필요, 실행에는 불필요)
출력: models/quantized/<원본이름>.int8.onnx + fp32 대비 키포인트 오차 리포트
적용: configs/config*.yaml 의 model.quantized: true 로 켠다 (끄면 즉시 fp32 복귀)
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import numpy as np

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
OUTPUT_DIR = os.path.join(ROOT_DIR, "models", "quantized")
CALIB_FRAME_COUNT = 100
CALIB_INTERVAL_SEC = 0.1     # 프레임 간격 — 자세가 조금씩 다른 샘플을 모은다
ACCURACY_WARN_PX = 5.0       # fp32 대비 평균 키포인트 오차 경고 임계


def build_input(pre_img):
    """rtmlib BaseTool.inference와 동일한 입력 텐서 — (1,3,H,W) float32."""
    return np.ascontiguousarray(pre_img.transpose(2, 0, 1), dtype=np.float32)[None]


def collect_frames(config, source):
    if source == "camera":
        from src.capture.camera_stream import CameraStream

        camera = CameraStream(config).start()
        print(f"[INFO] 카메라에서 보정 프레임 {CALIB_FRAME_COUNT}장 수집 — 카메라 앞에서 "
              "천천히 팔을 움직여 주세요 (다양한 자세 = 좋은 보정)")
        frames = []
        for i in range(CALIB_FRAME_COUNT):
            frames.append(camera.capture_frame())
            time.sleep(CALIB_INTERVAL_SEC)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{CALIB_FRAME_COUNT}")
        camera.stop()
        return frames
    rng = np.random.default_rng(0)
    return [
        (rng.random((config["camera"]["height_px"], config["camera"]["width_px"], 3)) * 255)
        .astype(np.uint8)
        for _ in range(20)
    ]


def _ensure_opset13(onnx_path):
    """채널별 양자화(DequantizeLinear axis)는 opset 13+ 필요 — 낮으면 상향 변환.

    변환 실패 시 원본 경로와 함께 per_channel 불가를 알린다.
    """
    import onnx

    model = onnx.load(onnx_path)
    opset = max(imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx"))
    if opset >= 13:
        return onnx_path, True
    try:
        from onnx import version_converter

        converted = version_converter.convert_version(model, 13)
        converted_path = onnx_path.replace(".onnx", ".opset13.tmp.onnx")
        onnx.save(converted, converted_path)
        print(f"  (opset {opset}→13 상향 변환)")
        return converted_path, True
    except Exception as error:   # 일부 연산은 변환 미지원 — 채널별 없이 진행
        print(f"  (opset 변환 실패: {error} — per_channel 없이 양자화)")
        return onnx_path, False


def quantize(onnx_path, samples, output_path, label):
    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType, quantize_static,
    )

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._iter = iter(samples)

        def get_next(self):
            sample = next(self._iter, None)
            return None if sample is None else {"input": sample}

    source_path, per_channel = _ensure_opset13(onnx_path)
    quantize_static(
        source_path, output_path, _Reader(),
        quant_format=QuantFormat.QDQ, per_channel=per_channel,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
    )
    if source_path != onnx_path and os.path.exists(source_path):
        os.remove(source_path)   # 임시 변환본 정리
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[OK] {label} 양자화 완료 → {os.path.relpath(output_path, ROOT_DIR)} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="검출·포즈 모델 INT8 양자화")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--source", choices=["camera", "dummy"], default="camera")
    args = parser.parse_args()

    try:
        import onnx  # noqa: F401 — quantize_static의 실제 의존
    except ImportError:
        raise SystemExit("[FAIL] onnx 패키지가 필요합니다: pip install onnx")

    config = load_config(args.config)
    engine = config["model"].get("pose_engine", "body")
    if engine == "wholebody":
        from rtmlib import Wholebody as solution
    else:
        from rtmlib import Body as solution
    fp32 = solution(mode=config["model"]["pose_mode"], backend="onnxruntime", device="cpu")
    det, pose = fp32.det_model, fp32.pose_model

    frames = collect_frames(config, args.source)

    # 보정 샘플: 실제 파이프라인과 동일한 전처리를 거친 입력 텐서
    det_samples, pose_samples = [], []
    for frame in frames:
        pre_det, _ = det.preprocess(frame)
        det_samples.append(build_input(pre_det))
        bboxes = det(frame)
        if len(bboxes) == 0:
            bboxes = [[0, 0, frame.shape[1], frame.shape[0]]]   # 무인 프레임 — 전체 화면으로라도 보정
        for bbox in list(bboxes)[:1]:
            pre_pose, _, _ = pose.preprocess(frame, bbox)
            pose_samples.append(build_input(pre_pose))
    print(f"[INFO] 보정 샘플 — 검출 {len(det_samples)}장 · 포즈 {len(pose_samples)}장")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    det_out = os.path.join(OUTPUT_DIR, os.path.basename(det.onnx_model).replace(".onnx", ".int8.onnx"))
    pose_out = os.path.join(OUTPUT_DIR, os.path.basename(pose.onnx_model).replace(".onnx", ".int8.onnx"))
    quantize(det.onnx_model, det_samples, det_out, "검출(YOLOX)")
    quantize(pose.onnx_model, pose_samples, pose_out, "포즈(RTMPose)")

    # ── 자가 검증: fp32 vs int8 키포인트 오차 + 속도 ──
    import onnxruntime as ort

    int8_pose = ort.InferenceSession(pose_out, providers=["CPUExecutionProvider"])
    diffs = []
    t_fp32 = t_int8 = 0.0
    for sample in pose_samples[:30]:
        t0 = time.perf_counter()
        out_fp32 = pose.session.run(None, {"input": sample})
        t_fp32 += time.perf_counter() - t0
        t0 = time.perf_counter()
        out_int8 = int8_pose.run(None, {"input": sample})
        t_int8 += time.perf_counter() - t0
        # SimCC 출력 → argmax 좌표 비교 (x·y 각각) — 픽셀 단위 오차
        for fp32_axis, int8_axis, scale in ((out_fp32[0], out_int8[0], 192 / out_fp32[0].shape[-1]),
                                            (out_fp32[1], out_int8[1], 256 / out_fp32[1].shape[-1])):
            diff = np.abs(fp32_axis.argmax(-1) - int8_axis.argmax(-1)) * scale
            diffs.append(float(diff.mean()))
    count = max(len(pose_samples[:30]), 1)
    print()
    print("=" * 60)
    print(f"정확도: fp32 대비 평균 키포인트 오차 {np.mean(diffs):.2f} px "
          f"(경고 임계 {ACCURACY_WARN_PX} px)")
    if np.mean(diffs) > ACCURACY_WARN_PX:
        print("[경고] 오차가 큽니다 — 보정 프레임을 사람이 나온 실카메라로 다시 수집하세요")
    print(f"속도(포즈 1장): fp32 {t_fp32 / count * 1000:.1f} ms → int8 {t_int8 / count * 1000:.1f} ms")
    print("적용: configs/config*.yaml → model.quantized: true (끄면 즉시 fp32 복귀)")
    print("=" * 60)


if __name__ == "__main__":
    main()
