"""제스처 모델 파인튜닝 — 사전학습(HaGRIDv2 YOLOv10n)에서 이어서 학습한다.

학습은 OS 무관 (CUDA → MPS → CPU 자동 감지). 결과 .pt는
export/to_inference.py로 추론(gesture_kiosk) 쪽에 이식한다.

사용법 (training/ 루트에서, venv_train 활성화 후):
    python gesture/train.py --data gesture/dataset_v1.yaml --epochs 50
    python gesture/train.py --weights ../gesture_kiosk/models/weights/YOLOv10n_gestures.pt

주의(기획서 5.4): 데이터셋은 인물 단위로 train/val 분할 — 같은 사람이 양쪽에 있으면
검증 수치가 부풀어 오른다 (데이터 유출).
"""
import argparse
import os

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "gesture_kiosk", "models", "weights", "YOLOv10n_gestures.pt",
)


def resolve_device():
    """학습 디바이스 자동 감지 — CUDA(윈도우/리눅스) → MPS(맥) → CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="제스처 파인튜닝 (OS 무관)")
    parser.add_argument("--data", required=True, help="YOLO 데이터셋 yaml (dataset_template.yaml 참고)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="시작 가중치 (.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from ultralytics import YOLO

    device = resolve_device()
    print(f"[INFO] device={device} / weights={os.path.basename(args.weights)}")

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
    )
    print("[DONE] 학습 완료 — best.pt 경로는 위 로그(runs/...) 참고")
    print("[다음] python export/to_inference.py --weights <best.pt> --dataset <이름> --metric <지표>")
    return results


if __name__ == "__main__":
    main()
