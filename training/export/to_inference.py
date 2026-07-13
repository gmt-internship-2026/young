"""학습 결과 이식 — 학습한 .pt를 추론(gesture_kiosk) 쪽으로 복사한다.

기획서 4.8 명명 규칙을 적용한다: {모델}_{데이터셋}_{지표}_{날짜}.pt
어느 OS에서 학습했든 .pt는 그대로 이식된다 — 추론 PC에서는 TensorRT 엔진만
다시 빌드하면 된다 (gesture_kiosk/scripts/build_engine.py).

사용법:
    python export/to_inference.py --weights runs/detect/train/weights/best.pt \
        --dataset v1 --metric map85
"""
import argparse
import datetime
import os
import shutil

INFERENCE_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "gesture_kiosk", "models", "weights",
)


def main():
    parser = argparse.ArgumentParser(description="학습 가중치 -> 추론 폴더 이식")
    parser.add_argument("--weights", required=True, help="학습 결과 .pt (예: runs/.../best.pt)")
    parser.add_argument("--model", default="YOLOv10n", help="모델 이름 (명명 규칙용)")
    parser.add_argument("--dataset", required=True, help="데이터셋 이름 (예: v1)")
    parser.add_argument("--metric", required=True, help="대표 지표 (예: map85)")
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"가중치가 없습니다: {args.weights}")

    date_str = datetime.date.today().strftime("%Y%m%d")
    target_name = f"{args.model}_{args.dataset}_{args.metric}_{date_str}.pt"
    target_path = os.path.abspath(os.path.join(INFERENCE_WEIGHTS_DIR, target_name))

    os.makedirs(INFERENCE_WEIGHTS_DIR, exist_ok=True)
    shutil.copy2(args.weights, target_path)
    print(f"[DONE] 이식 완료: {target_path}")
    print("[다음 — 추론 PC에서]")
    print(f"  1) configs/config.yaml의 weights_path를 models/weights/{target_name} 로 변경")
    print("  2) python scripts/smoke_test.py 로 로드 확인")
    print("  3) (엔진 백엔드면) python scripts/build_engine.py 재실행 — 엔진은 PC 전용")


if __name__ == "__main__":
    main()
