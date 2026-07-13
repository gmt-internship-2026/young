"""detector_mediapipe.classify_hand_landmarks 단위 테스트 — mediapipe 설치 없이 실행 가능.

합성 랜드마크로 기하 판정 규칙(펴짐/굽힘/핀치)을 검증한다. 좌표는 화면 픽셀
관례(y는 아래로 증가)를 따르고, 손목(100, 200) 위로 손가락이 뻗은 자세를 기본으로 한다.
"""
import unittest

from src.inference.detector_mediapipe import classify_hand_landmarks, user_side_from_label

EXTENDED_RATIO = 1.15
OK_PINCH_RATIO = 0.35

WRIST = (100.0, 200.0)
MIDDLE_MCP = (100.0, 140.0)  # 손 크기 기준 = 손목~중지MCP = 60px

# 손가락 x 위치 (검지, 중지, 약지, 새끼)
FINGER_X = {"index": 85.0, "middle": 100.0, "ring": 115.0, "pinky": 130.0}
# (PIP, TIP) 랜드마크 번호
FINGER_LM = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}

THUMB_IP = (70.0, 170.0)          # 손목에서 42px
THUMB_TIP_NEUTRAL = (80.0, 180.0)  # 굽힘 (28px < 42*1.15)
THUMB_TIP_UP = (60.0, 130.0)       # 폄 + 손목보다 위 (81px > 42*1.15)


def build_hand(extended, thumb_tip=THUMB_TIP_NEUTRAL, index_tip_override=None):
    """21개 랜드마크 합성. extended: 펴는 손가락 이름 목록."""
    points = [(100.0, 180.0)] * 21  # 미사용 랜드마크는 손바닥 안 중립 위치
    points[0] = WRIST
    points[9] = MIDDLE_MCP
    points[3] = THUMB_IP
    points[4] = thumb_tip
    for name, (pip, tip) in FINGER_LM.items():
        x = FINGER_X[name]
        points[pip] = (x, 120.0)                            # PIP — 손목에서 약 81px
        points[tip] = (x, 80.0) if name in extended else (x, 150.0)  # 폄 121px / 굽힘 52px
    if index_tip_override is not None:
        points[8] = index_tip_override
    return points


def classify(points):
    return classify_hand_landmarks(points, EXTENDED_RATIO, OK_PINCH_RATIO)


class ClassifyHandLandmarksTest(unittest.TestCase):
    def test_fist_all_fingers_curled(self):
        """네 손가락 굽힘 + 엄지 중립 = fist."""
        self.assertEqual(classify(build_hand(extended=[])), "fist")

    def test_palm_all_fingers_extended(self):
        """네 손가락 폄 = palm (open_hand로 매핑됨)."""
        self.assertEqual(
            classify(build_hand(extended=["index", "middle", "ring", "pinky"])), "palm"
        )

    def test_ok_pinch_with_three_fingers_extended(self):
        """엄지-검지 끝 맞닿음 + 중지·약지·새끼 폄 = ok."""
        points = build_hand(
            extended=["middle", "ring", "pinky"],
            thumb_tip=(74.0, 156.0),
            index_tip_override=(75.0, 155.0),  # 엄지 끝과 1.4px — 핀치
        )
        self.assertEqual(classify(points), "ok")

    def test_one_index_only(self):
        """검지만 폄 = one (point로 매핑됨)."""
        self.assertEqual(classify(build_hand(extended=["index"])), "one")

    def test_like_thumb_up_only(self):
        """엄지만 폄 + 엄지 끝이 손목보다 위 = like (thumbs_up으로 매핑됨)."""
        self.assertEqual(
            classify(build_hand(extended=[], thumb_tip=THUMB_TIP_UP)), "like"
        )

    def test_victory_is_unknown(self):
        """검지+중지만 폄(브이) — 정의된 제스처가 아니므로 None."""
        self.assertIsNone(classify(build_hand(extended=["index", "middle"])))

    def test_pinch_without_extended_fingers_is_not_ok(self):
        """핀치여도 중지·약지·새끼가 굽어 있으면 ok가 아니다 — fist로 본다."""
        points = build_hand(
            extended=[],
            thumb_tip=(86.0, 160.0),  # 굽힌 검지 끝(85, 150)과 10px — 핀치이되 엄지는 굽힘
        )
        self.assertEqual(classify(points), "fist")

    def test_ok_beats_palm_priority(self):
        """핀치 + 네 손가락 폄이 동시에 성립해도 ok가 우선한다."""
        points = build_hand(
            extended=["index", "middle", "ring", "pinky"],
            thumb_tip=(84.0, 81.0),
            index_tip_override=(85.0, 80.0),
        )
        self.assertEqual(classify(points), "ok")


class UserSideFromLabelTest(unittest.TestCase):
    """handedness 라벨 -> 사용자 좌/우 매핑 — 거울·flip_handedness 4조합 + 무효 라벨."""

    def test_mirror_with_flip_swaps_label(self):
        # 배포 기본 조합 (mirror=true + flip=true, 0.10.35 실측 보정)
        self.assertEqual(user_side_from_label("left", True, True), "right")
        self.assertEqual(user_side_from_label("right", True, True), "left")

    def test_mirror_without_flip_keeps_label(self):
        # 문서 기준 매핑 (거울 입력 가정 라벨)
        self.assertEqual(user_side_from_label("left", True, False), "left")

    def test_no_mirror_without_flip_swaps_label(self):
        self.assertEqual(user_side_from_label("left", False, False), "right")

    def test_no_mirror_with_flip_keeps_label(self):
        self.assertEqual(user_side_from_label("left", False, True), "left")

    def test_unknown_label_returns_none(self):
        self.assertIsNone(user_side_from_label("unknown", True, True))


if __name__ == "__main__":
    unittest.main()
