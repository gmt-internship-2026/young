"""announcer 단위 테스트 — pyttsx3·오디오 없이 순수 판정 로직만 검증한다.

select_voice/status_announcement는 pyttsx3에 의존하지 않는 순수 함수라
가짜 보이스 객체·평범한 dict만으로 결정적으로 테스트한다.
"""
import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.announce.announcer import select_voice, status_announcement

FakeVoice = namedtuple("FakeVoice", ["id", "name"])


class SelectVoiceTest(unittest.TestCase):
    def test_matches_by_name(self):
        voices = [
            FakeVoice(id="v1", name="Microsoft Zira Desktop - English (United States)"),
            FakeVoice(id="v2", name="Microsoft Heami Desktop - Korean (Korea)"),
        ]
        self.assertEqual(select_voice(voices, "Korean"), "v2")

    def test_matches_by_id_when_name_lacks_keyword(self):
        # 맥 nsss 보이스는 이름 대신 id에 로캘 코드가 들어가는 경우가 많다
        voices = [FakeVoice(id="com.apple.voice.compact.ko-KR.Yuna", name="Yuna")]
        self.assertEqual(select_voice(voices, "ko"), "com.apple.voice.compact.ko-KR.Yuna")

    def test_case_insensitive(self):
        voices = [FakeVoice(id="v1", name="Heami - KOREAN")]
        self.assertEqual(select_voice(voices, "korean"), "v1")

    def test_no_match_returns_none(self):
        voices = [FakeVoice(id="v1", name="Microsoft Zira Desktop - English (United States)")]
        self.assertIsNone(select_voice(voices, "Korean"))

    def test_empty_voice_list_returns_none(self):
        self.assertIsNone(select_voice([], "Korean"))

    def test_empty_keyword_returns_none(self):
        voices = [FakeVoice(id="v1", name="Microsoft Heami Desktop - Korean (Korea)")]
        self.assertIsNone(select_voice(voices, ""))

    def test_first_match_wins(self):
        voices = [
            FakeVoice(id="v1", name="Korean A"),
            FakeVoice(id="v2", name="Korean B"),
        ]
        self.assertEqual(select_voice(voices, "Korean"), "v1")


class StatusAnnouncementTest(unittest.TestCase):
    TEMPLATES = {
        "locked": "사용자가 인식되었습니다",
        "released": "사용자 인식이 해제되었습니다. 화면 앞에 서 주세요",
    }

    def test_locked_returns_locked_text(self):
        self.assertEqual(status_announcement(True, self.TEMPLATES), self.TEMPLATES["locked"])

    def test_released_returns_released_text(self):
        self.assertEqual(status_announcement(False, self.TEMPLATES), self.TEMPLATES["released"])

    def test_missing_key_returns_none(self):
        self.assertIsNone(status_announcement(True, {"released": "x"}))

    def test_empty_templates_returns_none(self):
        self.assertIsNone(status_announcement(True, {}))
        self.assertIsNone(status_announcement(False, {}))


if __name__ == "__main__":
    unittest.main()
