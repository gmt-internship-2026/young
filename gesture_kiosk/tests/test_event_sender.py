"""이벤트 전송 규격 테스트 — 델파이7 텍스트 규격 + 파이프(stdio) 출력 검증.

2026-07-23: UDP·웹소켓 제거(회사 결정 — 네트워크 철회), stdio 전환.
stdout은 이벤트 전용 채널이므로 바이트 단위까지 규격을 고정한다.
"""
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.event_sender import (
    ConsoleEventSender, StdioEventSender, build_text_payload, create_event_sender,
)
from src.postprocess.gesture_filter import GestureEvent


def _event(class_name="confirm", hand_side="left"):
    return GestureEvent(class_name=class_name, conf=1.0, ts_sec=12345.6789, hand_side=hand_side)


class PayloadFormatTest(unittest.TestCase):
    def test_text_payload_is_delphi_line(self):
        # GESTURE|이벤트|손|신뢰도|시각 + CRLF — 델파이7 Pos/Copy 파싱 규격
        self.assertEqual(
            build_text_payload(_event()), b"GESTURE|confirm|left|1.00|12345.679\r\n"
        )

    def test_text_payload_without_hand_side(self):
        # 손 미상 — 3번째 필드는 빈 문자열 (구분자 수는 항상 4개로 고정)
        line = build_text_payload(_event(hand_side=None)).decode("ascii")
        self.assertEqual(line.count("|"), 4)
        self.assertEqual(line.split("|")[2], "")

    def test_text_payload_is_ascii_single_line(self):
        line = build_text_payload(_event("home", "right"))
        self.assertTrue(line.endswith(b"\r\n"))
        self.assertEqual(line.count(b"\n"), 1)      # 이벤트 1건 = 한 줄


class _FakeStdout:
    """sys.stdout 대역 — buffer에 쓰인 바이트를 그대로 보관한다."""

    def __init__(self):
        self.buffer = io.BytesIO()


class StdioSenderTest(unittest.TestCase):
    def setUp(self):
        self._real_stdout = sys.stdout
        sys.stdout = _FakeStdout()

    def tearDown(self):
        sys.stdout = self._real_stdout

    def test_stdio_writes_exact_line_to_stdout(self):
        StdioEventSender().send(_event("left", "right"))
        self.assertEqual(sys.stdout.buffer.getvalue(),
                         b"GESTURE|left|right|1.00|12345.679\r\n")

    def test_consecutive_events_are_separate_lines(self):
        sender = StdioEventSender()
        sender.send(_event("right", "right"))
        sender.send(_event("confirm", "left"))
        lines = sys.stdout.buffer.getvalue().split(b"\r\n")
        self.assertEqual(lines[0], b"GESTURE|right|right|1.00|12345.679")
        self.assertEqual(lines[1], b"GESTURE|confirm|left|1.00|12345.679")


class CreateSenderTest(unittest.TestCase):
    def test_stdio_mode(self):
        sender = create_event_sender({"event_output": {"mode": "stdio"}})
        self.assertIsInstance(sender, StdioEventSender)

    def test_console_mode(self):
        sender = create_event_sender({"event_output": {"mode": "console"}})
        self.assertIsInstance(sender, ConsoleEventSender)

    def test_unknown_mode_rejected_at_startup(self):
        # 오타는 이벤트 발생 시점이 아니라 시작 시점에 죽어야 한다
        with self.assertRaises(ValueError):
            create_event_sender({"event_output": {"mode": "udp"}})


if __name__ == "__main__":
    unittest.main()
