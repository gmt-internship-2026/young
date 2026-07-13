"""합성 주민등록증 이미지 생성 골격 — 개인정보 없이 카드 검출/OCR 학습 데이터를 만든다.

가짜 이름 + 검증공식에 맞춘 가상 주민번호로 카드 레이아웃을 렌더링하고,
배경 사진 위에 원근·회전을 줘서 붙이면 카드 검출기 학습 데이터가 된다.

사용법:
    python idcard/make_synthetic.py --count 20 --out idcard/synthetic
(골격 단계 — 실제 가동 시 서체·레이아웃을 실물 규격에 맞춰 보강한다. README 참고)
"""
import argparse
import os
import random

from PIL import Image, ImageDraw

CARD_SIZE_PX = (860, 540)  # 실물 비율(85.6×54mm) 근사
RRN_CHECK_WEIGHTS = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)

FAKE_SURNAMES = "김이박최정강조윤장임"
FAKE_GIVEN = ["민준", "서연", "도윤", "지우", "하은", "시우", "수아", "예준"]


def make_fake_rrn(rng):
    """검증공식을 통과하는 가상 주민번호 (실존 번호와 무관한 난수)."""
    front6 = f"{rng.randint(50, 99):02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
    back6 = str(rng.randint(1, 4)) + f"{rng.randint(0, 99999):05d}"
    digits12 = front6 + back6
    check = (11 - sum(int(d) * w for d, w in zip(digits12, RRN_CHECK_WEIGHTS)) % 11) % 10
    return f"{front6}-{back6}{check}"


def render_card(name, rrn):
    card = Image.new("RGB", CARD_SIZE_PX, (235, 233, 225))
    draw = ImageDraw.Draw(card)
    # 골격: 기본 서체로 레이아웃만 잡는다 — 가동 시 실물 규격 서체·위치로 교체
    draw.text((60, 50), "주민등록증", fill=(30, 30, 30))
    draw.text((60, 180), name, fill=(20, 20, 20))
    draw.text((60, 260), rrn, fill=(20, 20, 20))
    draw.text((60, 420), "서울특별시장", fill=(60, 60, 60))
    draw.rectangle((600, 140, 800, 400), outline=(150, 150, 150), width=2)  # 사진 자리
    return card


def main():
    parser = argparse.ArgumentParser(description="합성 주민등록증 생성 (개인정보 없음)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for index in range(args.count):
        name = rng.choice(FAKE_SURNAMES) + rng.choice(FAKE_GIVEN)
        card = render_card(name, make_fake_rrn(rng))
        card.save(os.path.join(args.out, f"synth_card_{index:04d}.png"))
    print(f"[DONE] 합성 카드 {args.count}장: {args.out}")
    print("[다음] 배경 합성·원근 변형을 더해 검출기 학습 데이터로 확장 (README 참고)")


if __name__ == "__main__":
    main()
