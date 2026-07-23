"""손 모양(point/fist/none) 분류기 학습 — collect_hand_shape_data.py가 모은 CSV로
로지스틱 회귀를 학습하고, models/weights/에 가중치를 내보낸다 (2026-07-23 도입).

scikit-learn은 이 학습 스크립트에서만 쓴다(BSD 라이선스, ultralytics/AGPL과 무관) —
추론 쪽(hand_shape_classifier.py)에는 sklearn을 설치하지 않는다. 내보낸 .npz는
계수 행렬 + 절편 + 클래스 이름뿐이라 numpy 행렬곱 하나로 추론된다.

사용법 (gesture_kiosk 폴더에서, venv_win 활성화 후 — scikit-learn만 추가 설치 필요):
    venv_win\\Scripts\\pip.exe install scikit-learn
    venv_win\\Scripts\\python.exe scripts\\train_hand_shape_classifier.py

인물 단위 분할(기획서 5.4) — CSV의 person_id 열 기준으로 train/val을 나눈다(같은
사람이 양쪽에 있으면 안 됨). --val-person으로 검증에 뺄 사람을 지정한다(수집자가
1명뿐이면 생략 가능 — 그 경우 검증은 건너뛰고 전량 학습에만 쓴다).
"""
import argparse
import csv
import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, "data", "hand_shape", "landmarks.csv")
DEFAULT_OUT_PATH = os.path.join(ROOT_DIR, "models", "weights", "hand_shape_classifier.npz")


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    person_ids = [row[0] for row in rows]
    labels = [row[1] for row in rows]
    features = np.array([[float(v) for v in row[2:]] for row in rows], dtype=np.float64)
    return person_ids, labels, features, header[2:]


def main():
    parser = argparse.ArgumentParser(description="손 모양 분류기 학습(로지스틱 회귀)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                         help="collect_hand_shape_data.py가 만든 CSV")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="내보낼 .npz 경로")
    parser.add_argument(
        "--val-person", action="append", default=[],
        help="검증셋으로 뺄 person_id (인물 단위 분할, 여러 번 지정 가능)",
    )
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    person_ids, labels, features, feature_names = load_csv(args.data)
    val_mask = np.array([pid in args.val_person for pid in person_ids])
    if not args.val_person:
        print("[WARN] --val-person 미지정 — 전량을 학습에 쓰고 검증은 건너뜁니다"
              "(과적합 여부를 알 수 없으니 실기 확인 필수)")

    x_train, y_train = features[~val_mask], np.array(labels)[~val_mask]
    classes = sorted(set(labels))
    print(f"[INFO] 학습 샘플 {len(x_train)}건, 클래스 {classes}")
    if len(x_train) < 30:
        print("[WARN] 샘플이 너무 적습니다(30건 미만) — 각 라벨당 최소 수십 장은 더 모으는 걸 권장")

    model = LogisticRegression(max_iter=2000)
    model.fit(x_train, y_train)

    if val_mask.any():
        x_val, y_val = features[val_mask], np.array(labels)[val_mask]
        pred = model.predict(x_val)
        print(classification_report(y_val, pred))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        coef=model.coef_, intercept=model.intercept_,
        classes=np.array(model.classes_), feature_names=np.array(feature_names),
    )
    print(f"[DONE] 저장: {args.out}")
    print("[다음 단계] configs/config.yaml의 gestures.shapes.classifier_weights_path를 "
          "'models/weights/hand_shape_classifier.npz'로 설정하면 적용됩니다")


if __name__ == "__main__":
    main()
