"""팔등 분류기 학습 — 자체 수집 데이터로 소형 CNN을 처음부터 학습해 ONNX로 내보낸다.

사전학습 가중치를 일부러 쓰지 않는다 (2026-07-15) — 제3자 모델의 라이선스 의무
(고지문 동봉 등)를 0으로 만들기 위해서다 (기획서 9장 №9). 등쪽/안쪽 이진 분류
+ 회전 정규화된 크롭이라 소형 CNN으로 충분하다는 판단 — 정확도 미달이면
구조 확장보다 데이터 추가(사람·조명·옷 다양화)가 우선이다.

데이터: data/raw/arm_side/<person>/{dorsal,front}/*.jpg (scripts/collect_arm_side.py)
분할: 인물 단위(기획서 5.4) — --val-persons p03,p04 (미지정 시 마지막 태그 1명)
사용법:
    python scripts/train_arm_side.py --epochs 30
출력:
    models/weights/arm_side_cnn.onnx                      # 배포 고정 이름 (config 참조)
    models/weights/armside-cnn_own_acc{...}_{날짜}.pt      # 기록용 (기획서 4.8 명명)

torch는 학습 시에만 필요 — 배포 추론은 onnxruntime만 쓴다.
학습은 아무 환경에서나 가능 (5080 / 맥북 M5 Pro 등 — docs/TODO.md 메모).
"""
import argparse
import glob
import os
import random
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw", "arm_side")
LABEL_IDS = {"front": 0, "dorsal": 1}   # dorsal 확률 하나를 내는 이진 분류
SEED = 20260715
MAX_ROTATE_DEG = 10.0   # 크롭이 이미 회전 정규화돼 있어 작은 흔들림만 모사한다


def load_dataset(val_persons):
    """(경로, 라벨, 사람) 목록 -> 인물 단위 학습/검증 분할."""
    samples = []
    for path in glob.glob(os.path.join(DATA_DIR, "*", "*", "*.jpg")):
        label_name = os.path.basename(os.path.dirname(path))
        person_tag = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if label_name in LABEL_IDS:
            samples.append((path, LABEL_IDS[label_name], person_tag))
    if not samples:
        sys.exit(f"[train] 데이터 없음: {DATA_DIR} — scripts/collect_arm_side.py로 먼저 수집하세요")

    persons = sorted({person for _, _, person in samples})
    if not val_persons:
        val_persons = [persons[-1]]   # 미지정 시 마지막 태그 1명을 검증용으로
        if len(persons) == 1:
            print("[train] 경고: 수집 인물이 1명 — 인물 단위 검증 불가라 같은 사람으로 검증합니다.")
            print("        배포 판단용으로는 반드시 다른 사람 데이터로 다시 검증할 것 (기획서 5.4)")
            train = val = samples
            return train, val, val_persons
    train = [s for s in samples if s[2] not in val_persons]
    val = [s for s in samples if s[2] in val_persons]
    if not train or not val:
        sys.exit(f"[train] 분할 실패 — 인물 태그 확인: 전체 {persons}, 검증 지정 {val_persons}")
    return train, val, val_persons


def augment(image):
    """가벼운 증강 — 좌우반전(반대쪽 팔 모사)·밝기·소회전. 등/안쪽 라벨은 불변."""
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
    if random.random() < 0.8:
        image = cv2.convertScaleAbs(image, alpha=random.uniform(0.8, 1.2),
                                    beta=random.uniform(-20, 20))
    if random.random() < 0.5:
        h_px, w_px = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w_px / 2, h_px / 2),
                                         random.uniform(-MAX_ROTATE_DEG, MAX_ROTATE_DEG), 1.0)
        image = cv2.warpAffine(image, matrix, (w_px, h_px))
    return image


def load_batch(samples, input_size_px, is_train):
    """샘플 목록 -> (N,3,H,W) float32 텐서와 (N,1) 라벨 — preprocess_crop과 같은 규약."""
    import torch

    frames = []
    labels = []
    for path, label_id, _ in samples:
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.resize(image, (input_size_px, input_size_px))
        if is_train:
            image = augment(image)
        rgb = image[:, :, ::-1].astype(np.float32) / 255.0
        # 연속 메모리 보장 — 전치 뷰를 그대로 쌓으면 MPS backward가 stride 오류를 낸다 (2026-07-15 실측)
        frames.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        labels.append([float(label_id)])
    return torch.from_numpy(np.stack(frames)), torch.tensor(labels)


def build_model():
    """소형 CNN — 3개 conv 블록 + GAP + 선형 1출력(등쪽 logit). 사전학습 없음."""
    import torch.nn as nn

    def block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(3, 16), block(16, 32), block(32, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
    )


def eval_accuracy(model, val_samples, input_size_px, batch_size, device):
    import torch

    model.eval()
    correct_count = 0
    with torch.no_grad():
        for start in range(0, len(val_samples), batch_size):
            frames, labels = load_batch(
                val_samples[start:start + batch_size], input_size_px, is_train=False
            )
            probs = torch.sigmoid(model(frames.to(device)))
            correct_count += ((probs.cpu() >= 0.5).float() == labels).sum().item()
    return correct_count / len(val_samples)


def export_onnx(model, input_size_px, onnx_path):
    """sigmoid까지 포함해 내보낸다 — 런타임(arm_side_classifier)은 확률만 읽는다."""
    import torch
    import torch.nn as nn

    class _WithSigmoid(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return torch.sigmoid(self.net(x))

    model = model.cpu().eval()
    dummy = torch.zeros(1, 3, input_size_px, input_size_px)
    # dynamo=False: torch 2.9+ 기본 익스포터는 onnxscript 의존이 추가로 필요 — 레거시로 고정
    torch.onnx.export(_WithSigmoid(model), dummy, onnx_path,
                      input_names=["input"], output_names=["dorsal_prob"],
                      opset_version=13, dynamo=False)


def main():
    parser = argparse.ArgumentParser(description="팔등 분류기 학습 (자체 데이터·무 사전학습)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-persons", default="",
                        help="검증 전용 인물 태그 (쉼표 구분, 예: p03,p04)")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    config = load_config(args.config)
    input_size_px = config["model"]["arm_side"]["input_size_px"]
    onnx_path = config["model"]["arm_side"]["onnx_path"]

    val_persons = [p for p in args.val_persons.split(",") if p]
    train_samples, val_samples, val_persons = load_dataset(val_persons)
    print(f"[train] 학습 {len(train_samples)}장 / 검증 {len(val_samples)}장 "
          f"(검증 인물: {','.join(val_persons)})")

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_accuracy = 0.0
    for epoch_idx in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_samples)
        loss_sum = 0.0
        batch_count = 0
        for start in range(0, len(train_samples), args.batch_size):
            frames, labels = load_batch(
                train_samples[start:start + args.batch_size], input_size_px, is_train=True
            )
            optimizer.zero_grad()
            loss = loss_fn(model(frames.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batch_count += 1

        accuracy = eval_accuracy(model, val_samples, input_size_px, args.batch_size, device)
        print(f"[train] epoch {epoch_idx:02d}/{args.epochs} "
              f"loss={loss_sum / max(batch_count, 1):.4f} val_acc={accuracy:.3f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            date_tag = time.strftime("%Y-%m-%d")
            pt_path = os.path.join(os.path.dirname(onnx_path),
                                   f"armside-cnn_own_acc{accuracy:.2f}_{date_tag}.pt")
            torch.save(model.state_dict(), pt_path)
            export_onnx(model, input_size_px, onnx_path)
            model.to(device)

    print(f"[train] 완료 — 최고 검증 정확도 {best_accuracy:.3f}")
    print(f"[train] 배포 모델: {onnx_path} (config model.arm_side.onnx_path)")
    print("[train] 검증: python scripts/smoke_test.py 후 실기 확인")


if __name__ == "__main__":
    main()
