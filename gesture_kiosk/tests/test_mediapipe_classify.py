"""detector_mediapipe.classify_hand_orientation 단위 테스트 — mediapipe 설치 없이 실행.

합성 랜드마크로 손등/손바닥 외적 부호 규약을 8조합(좌우×거울×등/바닥) 전부 고정한다.
좌표는 화면 픽셀 관례(y 아래로 증가), 손목(100,200) 위로 손가락이 뻗은 자세.

부호의 물리 근거: 손바닥을 카메라로 향한 오른손('정지' 자세)을 비거울 화면에서
보면 엄지·검지가 화면 오른쪽에 온다 — 왼손·손등·거울은 각각 그 반대.
"""
import unittest

from src.inference.detector_mediapipe import classify_hand_orientation, user_side_from_label

BACK_FACING_THRESHOLD = 0.15

WRIST = (100.0, 200.0)
MIDDLE_MCP = (100.0, 140.0)  # 손 크기 기준 = 손목~중지MCP = 60px


def build_hand(index_mcp_x, pinky_mcp_x, mcp_y=145.0):
    """21개 랜드마크 합성 — 검지/새끼 MCP의 x 위치로 엄지쪽 방향을 정한다."""
    points = [(100.0, 170.0)] * 21  # 미사용 랜드마크는 손바닥 안 중립 위치
    points[0] = WRIST
    points[9] = MIDDLE_MCP
    points[5] = (index_mcp_x, mcp_y)
    points[17] = (pinky_mcp_x, mcp_y)
    return points


# 엄지쪽(검지MCP)이 화면 오른쪽 / 왼쪽에 있는 손 — |외적|/크기² ≈ 0.61 (임계 0.15 초과)
THUMB_SCREEN_RIGHT = build_hand(index_mcp_x=120.0, pinky_mcp_x=80.0)
THUMB_SCREEN_LEFT = build_hand(index_mcp_x=80.0, pinky_mcp_x=120.0)


def classify(points, user_side, is_mirror):
    return classify_hand_orientation(points, user_side, is_mirror, BACK_FACING_THRESHOLD)


class ClassifyHandOrientationTest(unittest.TestCase):
    def test_no_mirror_right_hand(self):
        """비거울 오른손: 엄지가 화면 오른쪽 = 손바닥, 왼쪽 = 손등."""
        self.assertEqual(classify(THUMB_SCREEN_RIGHT, "right", False), "palm")
        self.assertEqual(classify(THUMB_SCREEN_LEFT, "right", False), "back_of_hand")

    def test_no_mirror_left_hand(self):
        """비거울 왼손: 오른손과 대칭."""
        self.assertEqual(classify(THUMB_SCREEN_LEFT, "left", False), "palm")
        self.assertEqual(classify(THUMB_SCREEN_RIGHT, "left", False), "back_of_hand")

    def test_mirror_right_hand(self):
        """거울(배포 기본) 오른손: x반전으로 비거울과 부호가 뒤집힌다."""
        self.assertEqual(classify(THUMB_SCREEN_LEFT, "right", True), "palm")
        self.assertEqual(classify(THUMB_SCREEN_RIGHT, "right", True), "back_of_hand")

    def test_mirror_left_hand(self):
        self.assertEqual(classify(THUMB_SCREEN_RIGHT, "left", True), "palm")
        self.assertEqual(classify(THUMB_SCREEN_LEFT, "left", True), "back_of_hand")

    def test_edge_on_hand_is_held(self):
        """옆면(뒤집는 도중) — MCP들이 거의 일직선이라 |외적| 작음 → 판정 보류."""
        edge_on = build_hand(index_mcp_x=103.0, pinky_mcp_x=97.0)  # 외적/크기² ≈ 0.09
        self.assertIsNone(classify(edge_on, "right", True))

    def test_unknown_side_is_held(self):
        """좌/우를 모르면 부호 기준이 없어 판정하지 않는다."""
        self.assertIsNone(classify(THUMB_SCREEN_RIGHT, None, True))

    def test_rotation_invariance(self):
        """화면 내 회전(손끝이 옆을 향해도)에도 판정이 유지된다 — 외적 부호는 회전 불변."""
        # THUMB_SCREEN_RIGHT를 90도 돌린 손: 손목(100,200) 기준 (x,y)->(y축 회전)
        def rotate90(p):
            dx, dy = p[0] - WRIST[0], p[1] - WRIST[1]
            return (WRIST[0] - dy, WRIST[1] + dx)

        rotated = [rotate90(p) for p in THUMB_SCREEN_RIGHT]
        self.assertEqual(classify(rotated, "right", False), "palm")


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
