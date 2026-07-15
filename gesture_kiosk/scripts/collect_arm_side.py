"""팔등/팔안쪽 학습 데이터 수집 — 포즈로 전완을 자동 크롭해 라벨 폴더에 저장한다.

팔등 분류기(arm_side_classifier)의 자체 학습용 데이터를 만든다 (2026-07-15).
추론과 같은 crop_forearm()을 써서 학습·추론 입력 분포를 일치시킨다.

수집 요령:
- 양팔을 카메라 쪽으로 들고, 등쪽(팔등·손등 방향)이 보이면 d, 안쪽이 보이면 f.
  누를 때마다 화면에 보이는 양쪽 전완 크롭이 각각 1장씩 저장된다.
- 각도·거리·조명·옷(반팔/긴팔)을 바꿔가며 사람당 라벨별 300장 이상 권장.
- 인물 단위 분할(기획서 5.4)을 위해 --person 태그를 사람마다 바꿔서 수집한다.

사용법 (맥은 카메라 TCC 때문에 scripts/collect_arm_side_mac.command 로 실행):
    python scripts/collect_arm_side.py --config configs/config_mac.yaml --person p01
키: d=등쪽(dorsal) 저장 · f=안쪽(front) 저장 · q=종료
저장: data/raw/arm_side/<person>/<label>/<타임스탬프>_<side>.jpg
"""
import argparse
import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2

from src.inference.arm_side_classifier import crop_forearm
from src.inference.pose_estimator import (
    KPT_LEFT_ELBOW, KPT_LEFT_WRIST, KPT_RIGHT_ELBOW, KPT_RIGHT_WRIST, PoseEstimator,
)
from src.postprocess.person_lock import user_side_points
from src.utils.config_loader import load_config
from src.utils.logger import init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw", "arm_side")
LABEL_BY_KEY = {ord("d"): "dorsal", ord("f"): "front"}
PREVIEW_CROP_PX = 120  # 미리보기 창 구석에 띄우는 크롭 크기


def get_arm_points(person, kpt_conf, is_mirror):
    """사람 1명 -> 사용자 기준 {"left": (팔꿈치, 손목)|None, "right": ...}."""

    def arm_pair(elbow_idx, wrist_idx):
        elbow = person.keypoint(elbow_idx, kpt_conf)
        wrist = person.keypoint(wrist_idx, kpt_conf)
        if elbow is None or wrist is None:
            return None
        return (elbow, wrist)

    return user_side_points(
        arm_pair(KPT_LEFT_ELBOW, KPT_LEFT_WRIST),
        arm_pair(KPT_RIGHT_ELBOW, KPT_RIGHT_WRIST),
        is_mirror,
    )


def save_crops(crops, person_tag, label, saved_counts):
    label_dir = os.path.join(DATA_DIR, person_tag, label)
    os.makedirs(label_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    for side, crop in crops.items():
        cv2.imwrite(os.path.join(label_dir, f"{stamp}_{side}.jpg"), crop)
        saved_counts[label] += 1


def main():
    parser = argparse.ArgumentParser(description="팔등/팔안쪽 학습 데이터 수집")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--person", default="p01", help="수집 대상 사람 태그 (인물 단위 분할용)")
    args = parser.parse_args()

    config = load_config(args.config)
    init_logging(config)
    arm_cfg = config["model"]["arm_side"]
    is_mirror = config["camera"]["mirror"]
    kpt_conf = config["person_lock"]["kpt_conf_threshold"]

    pose_estimator = PoseEstimator(config)
    capture = cv2.VideoCapture(config["camera"]["device_id"])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width_px"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height_px"])

    saved_counts = {"dorsal": 0, "front": 0}
    print(f"[collect] 저장 위치: {os.path.join(DATA_DIR, args.person)}")
    print("[collect] d=등쪽 저장 · f=안쪽 저장 · q=종료")

    while True:
        is_read, frame = capture.read()
        if not is_read:
            print("[collect] 카메라 프레임 없음 — 종료")
            break
        if is_mirror:
            frame = cv2.flip(frame, 1)

        persons = pose_estimator.infer(frame)
        crops = {}
        if persons:
            best_person = max(persons, key=lambda p: p.conf)
            arm_points = get_arm_points(best_person, kpt_conf, is_mirror)
            for side, points in arm_points.items():
                if points is None:
                    continue
                crop = crop_forearm(frame, points[0], points[1],
                                    arm_cfg["crop_scale"], arm_cfg["input_size_px"])
                if crop is not None:
                    crops[side] = crop

        preview = frame.copy()
        for slot_idx, (side, crop) in enumerate(sorted(crops.items())):
            thumb = cv2.resize(crop, (PREVIEW_CROP_PX, PREVIEW_CROP_PX))
            x0 = 10 + slot_idx * (PREVIEW_CROP_PX + 10)
            preview[10:10 + PREVIEW_CROP_PX, x0:x0 + PREVIEW_CROP_PX] = thumb
            cv2.putText(preview, side, (x0, 10 + PREVIEW_CROP_PX + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
        cv2.putText(
            preview,
            f"person={args.person}  dorsal={saved_counts['dorsal']}  front={saved_counts['front']}"
            f"  arms={len(crops)}   [d]=dorsal [f]=front [q]=quit",
            (10, preview.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imshow("collect_arm_side", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key in LABEL_BY_KEY and crops:
            save_crops(crops, args.person, LABEL_BY_KEY[key], saved_counts)

    capture.release()
    cv2.destroyAllWindows()
    print(f"[collect] 종료 — dorsal {saved_counts['dorsal']}장, front {saved_counts['front']}장 저장")


if __name__ == "__main__":
    main()
