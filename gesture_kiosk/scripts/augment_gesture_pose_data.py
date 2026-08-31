"""제스처 자세 학습 데이터 증강 — 특정 라벨(기본 "ok")의 샘플 수는 늘리지 않고
"다양성"만 늘린다 (2026-08-31 신설, 사용자 보고: "confirm 인식이 잘 안 된다,
다양한 ok사인을 학습으로 보완해야할 것 같다").

data/gesture_pose/landmarks.csv의 "ok" 라벨은 694건이지만 person_id가 사실상
2명(jaeyoung 188 · 나머지 506)뿐이다 — 카메라 앞에서 손을 쥔 각도·손목 기울기가
그 두 사람의 습관 범위에서만 수집됐을 가능성이 높다. 실기에서 다른 사람이 조금
다른 각도로 OK 사인을 하면 학습 데이터와 안 닮아 hand_shape_classifier.classify의
max_dist_ratio·none_margin 방어선(hand_shape_classifier.py 독스트링 참고)에
걸려 None(정의 밖 동작)으로 튕길 수 있다.

이 스크립트는 새로 카메라 앞에 서지 않고, **기존 "ok" 샘플의 손목 기준 좌표에
작은 3D 회전 + 랜덤 노이즈**를 가해 "같은 손 모양을 다른 각도·다른 손 크기
오차로 봤을 때"의 합성 샘플을 만든다 — 실제 촬영 각도 편차·MediaPipe 랜드마크
검출 잡음을 흉내 낸 것이라 결정 경계가 촘촘한 한 방향으로만 좁게 잡히는 걸
막아준다. 진짜 카메라 재수집(collect_gesture_pose_data.py)을 대체하지는
않는다 — 완전히 다른 손 크기·조명·사람은 여전히 실측으로만 커버된다. 이 증강은
그 실측 데이터 사이의 "빈틈"을 메우는 보조 수단이다.

person_id는 원본 그대로 유지한다(접미사 안 붙임) — train_hand_shape_classifier.py
--val-person이 person_id로 train/val을 가르므로, 원본과 그 증강 사본을 같은
person_id로 두어야 검증에서 뺀 사람의 데이터(증강분 포함)가 학습에 새지 않는다
(기획서 5.4 인물 단위 분할 원칙 유지).

사용법 (gesture_kiosk 폴더에서):
    python scripts\\augment_gesture_pose_data.py --label ok --per-sample 5
    python scripts\\train_hand_shape_classifier.py ^
        --data data/gesture_pose/landmarks_augmented.csv ^
        --out models/weights/gesture_pose_classifier.npz
"""
import argparse
import csv
import os

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(ROOT_DIR, "data", "gesture_pose", "landmarks.csv")
DEFAULT_OUT_PATH = os.path.join(ROOT_DIR, "data", "gesture_pose", "landmarks_augmented.csv")


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    return header, rows


def random_rotation_matrix(max_angle_deg, rng):
    """축별로 [-max_angle_deg, +max_angle_deg] 사이 랜덤 각도를 뽑아 합성한
    3x3 회전 행렬 — 카메라를 살짝 다른 각도에서 본 것과 같은 효과를 낸다."""
    angles = np.radians(rng.uniform(-max_angle_deg, max_angle_deg, size=3))
    rx, ry, rz = angles
    rot_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    rot_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    rot_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def augment_features(features, max_angle_deg, jitter_std, rng):
    """정규화된 60차원 특징(20점 x,y,z, hand_shape_features.normalize_landmarks
    출력) 1건 -> 회전+잡음을 더한 새 60차원 특징.

    손목이 원점(0,0,0)인 좌표계라 원점 기준 회전만으로 손 전체가 자연스럽게
    돌아간다(별도 중심 이동 불필요). jitter_std는 랜드마크 검출 미세 오차를
    흉내 낸다 — 너무 크면 다른 모양이 돼버리므로 작게 잡는다(기본 0.03,
    정규화 단위 — 손목-중지MCP 거리의 3% 안팎 흔들림에 해당).
    """
    points = np.array(features, dtype=np.float64).reshape(20, 3)
    rotated = points @ random_rotation_matrix(max_angle_deg, rng).T
    noisy = rotated + rng.normal(scale=jitter_std, size=rotated.shape)
    return noisy.reshape(-1).tolist()


def main():
    parser = argparse.ArgumentParser(description="제스처 자세 학습 데이터 증강(회전+잡음)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                         help="collect_gesture_pose_data.py가 만든 원본 CSV")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH,
                         help="원본 + 증강 샘플을 함께 저장할 CSV(원본 파일은 건드리지 않음)")
    parser.add_argument("--label", action="append", default=[],
                         help="증강할 라벨(여러 번 지정 가능). 미지정 시 'ok' 하나만 증강")
    parser.add_argument("--per-sample", type=int, default=5,
                         help="원본 샘플 1건당 만들 증강 샘플 수(기본 5 — 694건이면 +3470건)")
    parser.add_argument("--max-angle-deg", type=float, default=15.0,
                         help="축별 최대 회전 각도(기본 15도 — 너무 크면 다른 모양이 될 수 있음)")
    parser.add_argument("--jitter-std", type=float, default=0.03,
                         help="랜드마크 좌표 잡음 표준편차(정규화 단위, 기본 0.03)")
    parser.add_argument("--seed", type=int, default=0, help="재현용 난수 시드")
    args = parser.parse_args()

    labels_to_augment = set(args.label) if args.label else {"ok"}
    rng = np.random.default_rng(args.seed)

    header, rows = load_csv(args.data)
    augmented_rows = []
    per_label_count = {label: 0 for label in labels_to_augment}
    for row in rows:
        person_id, label = row[0], row[1]
        if label not in labels_to_augment:
            continue
        features = [float(v) for v in row[2:]]
        for _ in range(args.per_sample):
            new_features = augment_features(features, args.max_angle_deg, args.jitter_std, rng)
            augmented_rows.append([person_id, label, *new_features])
        per_label_count[label] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerows(augmented_rows)

    for label, count in per_label_count.items():
        print(f"[INFO] {label}: 원본 {count}건 -> 증강 {count * args.per_sample}건 추가")
    print(f"[DONE] 저장: {args.out} (원본 {len(rows)}건 + 증강 {len(augmented_rows)}건)")
    print("[다음 단계] 아래로 재학습:")
    print(f"    python scripts\\train_hand_shape_classifier.py --data {args.out} "
          "--out models/weights/gesture_pose_classifier.npz")
    print("           --val-person을 쓸 거면 원본과 같은 person_id를 그대로 지정할 것")
    print("           (증강 사본도 같은 person_id라 자동으로 같이 검증셋에서 빠짐)")


if __name__ == "__main__":
    main()
