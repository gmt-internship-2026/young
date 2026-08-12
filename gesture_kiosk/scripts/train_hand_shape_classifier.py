"""손 모양(finger/fist/open/none) 분류기 학습 — collect_hand_shape_data.py가 모은
CSV로 로지스틱 회귀를 학습하고, models/weights/에 가중치를 내보낸다
(2026-07-23 CPU 브랜치 도입, 2026-08-03 이식).

scikit-learn은 이 학습 스크립트에서만 쓴다(BSD 라이선스, ultralytics/AGPL과 무관) —
추론 쪽(hand_shape_classifier.py)에는 sklearn을 설치하지 않는다. 내보낸 .npz는
계수 행렬 + 절편 + 클래스 이름 + (2026-08-07 신설) 클래스별 중심점·반경뿐이라
numpy 행렬곱 하나로 추론된다.

사용법 (gesture_kiosk 폴더에서, venv_win 활성화 후 — scikit-learn만 추가 설치 필요):
    venv_win\\Scripts\\pip.exe install scikit-learn
    venv_win\\Scripts\\python.exe scripts\\train_hand_shape_classifier.py

인물 단위 분할(기획서 5.4) — CSV의 person_id 열 기준으로 train/val을 나눈다(같은
사람이 양쪽에 있으면 안 됨). --val-person으로 검증에 뺄 사람을 지정한다(수집자가
1명뿐이면 생략 가능 — 그 경우 검증은 건너뛰고 전량 학습에만 쓴다).

★2026-08-07 클래스 중심점·반경 신설(사용자 보고 — "정의된 제스처가 아닌데
비슷한 제스처로 인식한다, 정의 밖 동작은 철저하게 none으로 잡혀야"):
로지스틱 회귀는 선형 경계 바깥으로 갈수록 오히려 더 확신에 찬 점수를 내는
구조라(경계에서 멀수록 로짓이 커짐), hand_shape_classifier.classify의
min_conf(소프트맥스 확률 임계값)만으로는 학습 데이터와 전혀 안 닮은 입력도
못 걸러낼 수 있었다(실기 확인됨 — "여전히 다 잡아 이상한 제스처도"). 이
스크립트가 클래스별 중심점(centroids, 학습 특징의 평균)과 반경(max_dist,
그 클래스 학습 샘플 중 중심점에서 가장 먼 거리)을 같이 저장해두면, 추론
쪽이 "예측 자체"가 아니라 "가장 가까운 중심점까지 실제 거리"로 정의 밖
입력을 걸러낼 수 있다(classify의 max_dist_ratio 참고) — 선형 분류기의
무한 외삽 문제를 우회하는 별도 채널.

★2026-08-07 none 낱개 샘플 신설(사용자 보고 — "V사인은 여전히 다 잡아,
비슷하면 다 select로 잡아"): 중심점 방식은 "none" 클래스 자체에는 잘 안
맞았다 — none은 V사인·중지만·약지만·손흔들기처럼 서로 완전히 다른 모양을
전부 한 라벨로 뭉친 **다봉(multi-modal) 클래스**라, 중심점(평균)이
그 다양한 모양들의 애매한 중간값이 되고 반경(제일 먼 샘플까지 거리)은
그 다양성을 다 덮으려고 크게 부풀어(fist=1.98 vs none은 훨씬 큼) — 정작
V사인처럼 finger_up과 진짜 닮은 경계 사례는 이 느슨해진 반경 안에 들어가
버렸다(뻐큐처럼 finger_up과 확실히 다른 모양은 그래도 걸러짐 — 사용자
실기 확인). 그래서 "none 전체 평균과의 거리"가 아니라 **"실제로 본 개별
none 샘플들 중 제일 가까운 거 하나"**까지 거리로 판단하도록 none_features
(none 라벨의 원본 학습 특징 전부)를 따로 저장한다(classify의
none_neighbor_ratio 참고) — none만 이 낱개-최근접 방식을 쓰고 나머지
(실제 제스처) 클래스는 여전히 중심점 방식을 쓴다: 제스처 클래스는 한
자세로 비교적 균일해 중심점이 잘 대표하지만, none은 애초에 "그 외 전부"라
하나의 대표점으로 요약이 안 된다.
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

    # 클래스별 중심점·반경(2026-08-07 신설) — model.classes_ 순서와 맞춰야
    # classify()가 coef/intercept와 같은 인덱스로 대응시킬 수 있다
    centroids = np.array([x_train[y_train == c].mean(axis=0) for c in model.classes_])
    max_dist = np.array([
        np.linalg.norm(x_train[y_train == c] - centroids[idx], axis=1).max()
        for idx, c in enumerate(model.classes_)
    ])
    print("[INFO] 클래스별 반경(정규화 특징 공간 유클리드, 학습 샘플 중 최대):")
    for c, dist in zip(model.classes_, max_dist):
        print(f"       {c}: {dist:.3f}")

    save_kwargs = dict(
        coef=model.coef_, intercept=model.intercept_,
        classes=np.array(model.classes_), feature_names=np.array(feature_names),
        centroids=centroids, max_dist=max_dist,
    )
    # none 낱개 샘플(2026-08-07 신설 — 위 모듈 독스트링 참고): "none"이 있을
    # 때만 저장 — 없는 구성(예: hand_shape_classifier.npz, fist/finger/open
    # 3모양)에서는 그냥 생략(하위 호환, 파일 크게 안 불어남)
    if "none" in model.classes_:
        none_features = x_train[y_train == "none"]
        save_kwargs["none_features"] = none_features
        # none_typical_gap — none 샘플들끼리 서로 얼마나 촘촘한지의 척도(각
        # 샘플의 최근접 "다른 none 샘플"까지 거리, 중앙값). none_neighbor_ratio
        # 검사(hand_shape_classifier.classify 참고)의 자연스러운 거리 단위 —
        # none 자체가 이질적인 자세들의 잡탕이라 이 값이 클 수 있는데, 그만큼
        # "none끼리도 이 정도는 떨어져 있다"는 실측 기준이 된다
        if len(none_features) >= 2:
            pair_dists = np.linalg.norm(
                none_features[:, None, :] - none_features[None, :, :], axis=2)
            np.fill_diagonal(pair_dists, np.inf)
            none_typical_gap = float(np.median(pair_dists.min(axis=1)))
        else:
            none_typical_gap = 0.0
        save_kwargs["none_typical_gap"] = np.array(none_typical_gap)
        print(f"[INFO] none 낱개 샘플 {len(none_features)}건 별도 저장 "
              f"(none_neighbor_ratio 방어용, 샘플 간 전형적 거리 {none_typical_gap:.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **save_kwargs)
    print(f"[DONE] 저장: {args.out}")
    print("[다음 단계] venv_win\\Scripts\\python.exe main.py로 바로 반영됩니다.")
    print("           exe 배포판(dist\\gesture_kiosk\\)을 쓰는 중이면 이 .npz 파일을")
    print("           gesture_kiosk.exe와 같은 폭에 복사만 해도 반영됩니다 —")
    print("           재빌드 불필요(hand_select._resolve_classifier_path 참고)")


if __name__ == "__main__":
    main()
