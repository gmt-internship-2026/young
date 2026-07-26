"""event_sender 단위 테스트 — 실제 델파이7·자식 프로세스·네트워크 서버 없이 검증한다.

UDP는 로컬 소켓 실전송(가볍고 결정적)으로, stdio는 sys.stdout을 캡처해 정확한
와이어 포맷(GESTURE|이벤트|손|신뢰도|시각)을 검증한다.
"""
import io
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.event_sender import (
    ConsoleEventSender, StdioEventSender, UdpEventSender, build_gesture_line,
    create_event_sender,
)
from src.postprocess.gesture_filter import GestureEvent


def make_event(class_name="right", shape="point", conf=1.0, ts_sec=123.456, data=None):
    return GestureEvent(class_name=class_name, conf=conf, ts_sec=ts_sec, shape=shape, data=data)


class BuildGestureLineTest(unittest.TestCase):
    def test_formats_fields_in_order(self):
        line = build_gesture_line(make_event(class_name="right", conf=1.0, ts_sec=12345.678))
        self.assertEqual(line, "GESTURE|right|{}|1.00|12345.678".format(""))

    def test_hand_field_is_always_empty(self):
        # young은 손 좌/우 정체성을 구분하지 않는다(shape는 손 모양이지 손 정체성이
        # 아니다) — 3번 필드는 항상 빈 문자열이어야 한다
        line = build_gesture_line(make_event(shape="fist"))
        fields = line.split("|")
        self.assertEqual(fields[2], "")

    def test_all_seven_classes_produce_valid_lines(self):
        # 「제스처 정의 보고서」(2026-07-23 회사 확정) 7개 고정 이벤트명
        for class_name in ("left", "right", "top", "bottom", "back", "home", "ok"):
            line = build_gesture_line(make_event(class_name=class_name))
            self.assertTrue(line.startswith(f"GESTURE|{class_name}|"), line)


class StdioEventSenderTest(unittest.TestCase):
    def _capture_stdout(self, gesture_event):
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            StdioEventSender().send(gesture_event)
            return sys.stdout.getvalue()
        finally:
            sys.stdout = original_stdout

    def test_writes_one_line_ending_in_newline(self):
        output = self._capture_stdout(make_event(class_name="left", conf=1.0, ts_sec=1.0))
        self.assertEqual(output, "GESTURE|left||1.00|1.000\n")

    def test_each_class_name_appears_verbatim(self):
        for class_name in ("left", "right", "top", "bottom", "back", "home", "ok"):
            output = self._capture_stdout(make_event(class_name=class_name))
            self.assertIn(f"GESTURE|{class_name}|", output)


class UdpEventSenderTest(unittest.TestCase):
    def test_send_delivers_json_datagram(self):
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.settimeout(1.0)
        port = recv_sock.getsockname()[1]
        try:
            sender = UdpEventSender({"event_output": {"udp": {"host": "127.0.0.1", "port": port}}})
            sender.send(make_event(class_name="ok", shape=None))
            data, _ = recv_sock.recvfrom(2048)
            self.assertIn(b'"class_name": "ok"', data)
        finally:
            recv_sock.close()


class CreateEventSenderTest(unittest.TestCase):
    def test_console_mode(self):
        self.assertIsInstance(
            create_event_sender({"event_output": {"mode": "console"}}), ConsoleEventSender
        )

    def test_udp_mode(self):
        config = {"event_output": {"mode": "udp", "udp": {"host": "127.0.0.1", "port": 9999}}}
        self.assertIsInstance(create_event_sender(config), UdpEventSender)

    def test_stdio_mode(self):
        config = {"event_output": {"mode": "stdio"}}
        self.assertIsInstance(create_event_sender(config), StdioEventSender)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            create_event_sender({"event_output": {"mode": "serial"}})


if __name__ == "__main__":
    unittest.main()
