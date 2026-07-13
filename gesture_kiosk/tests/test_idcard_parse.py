"""주민등록증 파싱 단위 테스트 — easyocr 없이 순수 파싱 로직만 검증한다.

OCR 결과 형태의 [(문장, 신뢰도)] 목록을 넣어 이름·주민번호 추출을 확인한다.
테스트 주민번호는 검증 공식에 맞춘 가상의 번호다.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.idcard_reader import (
    mask_name,
    mask_rrn,
    parse_id_fields,
    validate_rrn_checksum,
)

MIN_CONF = 0.4
# 검증 공식을 통과하는 가상 번호: 950101-1 + 23456 + 검증숫자 계산
VALID_RRN_DIGITS = "9501011234563"  # 테스트에서 공식으로 재확인한다


class ChecksumTest(unittest.TestCase):
    def test_valid_checksum(self):
        digits = VALID_RRN_DIGITS[:12]
        weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
        check = (11 - sum(int(d) * w for d, w in zip(digits, weights)) % 11) % 10
        self.assertTrue(validate_rrn_checksum(digits + str(check)))

    def test_invalid_checksum(self):
        self.assertFalse(validate_rrn_checksum("9501011234567"))
        self.assertFalse(validate_rrn_checksum("123"))
        self.assertFalse(validate_rrn_checksum("abcdefghijklm"))


class MaskTest(unittest.TestCase):
    def test_mask_name(self):
        self.assertEqual(mask_name("홍길동"), "홍*동")
        self.assertEqual(mask_name("김구"), "김*")
        self.assertEqual(mask_name("남궁민수"), "남**수")

    def test_mask_rrn(self):
        self.assertEqual(mask_rrn("950101", "1234567"), "950101-1******")


class ParseTest(unittest.TestCase):
    def test_standard_card_layout(self):
        items = [
            ("주민등록증", 0.99),
            ("홍길동(洪吉童)", 0.92),
            ("950101-1234567", 0.95),
            ("서울특별시 어딘가구 어딘가로 12", 0.8),
            ("2020. 1. 1.", 0.7),
            ("서울특별시장", 0.9),
        ]
        fields = parse_id_fields(items, MIN_CONF)
        self.assertIsNotNone(fields)
        self.assertEqual(fields["name"], "홍길동")
        self.assertEqual(fields["rrn"], "950101-1234567")

    def test_rrn_with_spaces_and_variants(self):
        items = [("김구", 0.9), ("950101 - 2345678", 0.9)]
        fields = parse_id_fields(items, MIN_CONF)
        self.assertEqual(fields["rrn"], "950101-2345678")

    def test_stopword_is_not_a_name(self):
        # 이름 줄을 못 읽고 서식 단어만 있으면 이름 없음 -> None
        items = [("주민등록증", 0.99), ("950101-1234567", 0.95)]
        self.assertIsNone(parse_id_fields(items, MIN_CONF))

    def test_name_below_rrn_is_not_used(self):
        # 이름 후보는 주민번호 '위' 줄에서만 찾는다 (발급기관장 이름 오인 방지)
        items = [("950101-1234567", 0.95), ("홍길동", 0.9)]
        self.assertIsNone(parse_id_fields(items, MIN_CONF))

    def test_low_conf_items_ignored(self):
        items = [("홍길동", 0.2), ("950101-1234567", 0.95)]  # 이름 신뢰도 미달
        self.assertIsNone(parse_id_fields(items, MIN_CONF))

    def test_no_rrn_returns_none(self):
        items = [("주민등록증", 0.99), ("홍길동", 0.9)]
        self.assertIsNone(parse_id_fields(items, MIN_CONF))

    def test_back7_first_digit_must_be_1_to_8(self):
        items = [("홍길동", 0.9), ("950101-9234567", 0.95)]
        self.assertIsNone(parse_id_fields(items, MIN_CONF))

    def test_hanja_parenthesis_stripped(self):
        items = [("홍길동 (洪吉童)", 0.9), ("950101-1234567", 0.95)]
        fields = parse_id_fields(items, MIN_CONF)
        self.assertEqual(fields["name"], "홍길동")


if __name__ == "__main__":
    unittest.main()
