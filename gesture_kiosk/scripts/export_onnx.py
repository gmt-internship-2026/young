"""[개발 PC 전용] HaGRID .pt -> ONNX 변환 — 배포 환경에서는 실행할 일 없다.

변환 결과(models/weights/YOLOv10n_gestures.onnx)는 저장소에 포함되어 있어,
평소에는 이 스크립트를 쓸 일이 없다. 새 가중치(파인튜닝 결과 등)를 배포
형식으로 바꿀 때만 개발 PC에서 1회 실행한다.

주의(라이선스 B안, 2026-07-11): 변환 도구인 ultralytics는 AGPL-3.0이라
배포 저장소의 의존성에서 제외했다. 이 스크립트를 돌릴 때만 개발 PC에
일시 설치한다 — 변환 도구 사용은 배포가 아니므로 공개 의무와 무관하다는
일반적 해석을 따르되, 최종 확인은 법무 검토 목록(№9)에 있다.

사용법 (개발 PC, 1회):
    pip install ultralytics          # 일시 설치
    python scripts/export_onnx.py --weights models/weights/YOLOv10n_gestures.pt
    pip uninstall ultralytics        # 정리(선택)
"""
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DEFAULT_WEIGHTS = os.path.join(ROOT_DIR, "models", "weights", "YOLOv10n_gestures.pt")


def main():
    parser = argparse.ArgumentParser(description=".pt -> .onnx 변환 (개발 PC 전용)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FAIL] ultralytics가 없습니다 — 개발 PC에서 `pip install ultralytics` 후 재실행")
        sys.exit(1)

    model = YOLO(args.weights)
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True)
    print(f"[DONE] ONNX 저장: {onnx_path}")
    print("[확인] 출력이 (1, 300, 6)이고 메타데이터에 names가 있어야 한다 — detector.py가 읽는다")


if __name__ == "__main__":
    main()
