"""이벤트 전송 규격 테스트 — 델파이7 텍스트 규격(2026-07-20)과 JSON 규격 검증.

UDP는 실제 루프백 왕복으로 확인한다 (127.0.0.1 임시 포트 — 외부 통신 없음).
"""
import json
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.event_sender import (
    UdpEventSender, build_json_payload, build_text_payload, create_event_sender,
)
from src.postprocess.gesture_filter import GestureEvent


def _event(class_name="select", hand_side="left"):
    return GestureEvent(class_name=class_name, conf=1.0, ts_sec=12345.6789, hand_side=hand_side)


def _udp_config(port, payload_format=None):
    udp = {"host": "127.0.0.1", "port": port}
    if payload_format is not None:
        udp["format"] = payload_format
    return {"event_output": {"mode": "udp", "udp": udp}}


class PayloadFormatTest(unittest.TestCase):
    def test_text_payload_is_delphi_line(self):
        # GESTURE|이벤트|손|신뢰도|시각 + CRLF — 델파이7 Pos/Copy 파싱 규격
        self.assertEqual(
            build_text_payload(_event()), b"GESTURE|select|left|1.00|12345.679\r\n"
        )

    def test_text_payload_without_hand_side(self):
        # 손 미상 — 3번째 필드는 빈 문자열 (구분자 수는 항상 4개로 고정)
        line = build_text_payload(_event(hand_side=None)).decode("ascii")
        self.assertEqual(line.count("|"), 4)
        self.assertEqual(line.split("|")[2], "")

    def test_text_payload_is_ascii_single_line(self):
        line = build_text_payload(_event("go_home", "right"))
        self.assertTrue(line.endswith(b"\r\n"))
        self.assertEqual(line.count(b"\n"), 1)      # 데이터그램 1개 = 한 줄

    def test_json_payload_roundtrip(self):
        payload = json.loads(build_json_payload(_event()).decode("utf-8"))
        self.assertEqual(payload["class_name"], "select")
        self.assertEqual(payload["hand_side"], "left")


class UdpLoopbackTest(unittest.TestCase):
    def setUp(self):
        self._recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv.bind(("127.0.0.1", 0))           # 임시 포트 — 충돌 없음
        self._recv.settimeout(2.0)
        self._port = self._recv.getsockname()[1]

    def tearDown(self):
        self._recv.close()

    def test_text_format_over_udp(self):
        sender = UdpEventSender(_udp_config(self._port, "text"))
        sender.send(_event("move_right", "right"))
        data, _ = self._recv.recvfrom(4096)
        self.assertEqual(data, b"GESTURE|move_right|right|1.00|12345.679\r\n")

    def test_missing_format_key_defaults_to_json(self):
        # 구 config(format 키 없음) — 종전 JSON 동작 유지
        sender = UdpEventSender(_udp_config(self._port))
        sender.send(_event())
        data, _ = self._recv.recvfrom(4096)
        self.assertEqual(json.loads(data.decode("utf-8"))["class_name"], "select")

    def test_unknown_format_rejected_at_startup(self):
        # 오타는 이벤트 발생 시점이 아니라 시작 시점에 죽어야 한다
        with self.assertRaises(ValueError):
            UdpEventSender(_udp_config(self._port, "xml"))

    def test_create_event_sender_udp_mode(self):
        sender = create_event_sender(_udp_config(self._port, "text"))
        self.assertIsInstance(sender, UdpEventSender)


if __name__ == "__main__":
    unittest.main()
