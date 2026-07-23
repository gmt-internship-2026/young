"""event_sender 단위 테스트 — 실제 델파이7·네임드 파이프·네트워크 서버 없이 검증한다.

UDP는 로컬 소켓 실전송(가볍고 결정적)으로, 파이프는 open_fn·clock 주입으로
연결/재접속 backoff/전송 실패 경로를 검증한다 (실제 윈도우 네임드 파이프 없이,
GMtech_project의 WebSocketEventSender 테스트와 같은 패턴).
"""
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.event_sender import (
    ConsoleEventSender, PIPE_COMMAND_BY_CLASS_NAME, PipeEventSender, UdpEventSender,
    create_event_sender,
)
from src.postprocess.gesture_filter import GestureEvent


def make_event(class_name="move_right", shape="point", data=None):
    return GestureEvent(class_name=class_name, conf=0.987654, ts_sec=123.0, shape=shape, data=data)


class UdpEventSenderTest(unittest.TestCase):
    def test_send_delivers_json_datagram(self):
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", 0))
        recv_sock.settimeout(1.0)
        port = recv_sock.getsockname()[1]
        try:
            sender = UdpEventSender({"event_output": {"udp": {"host": "127.0.0.1", "port": port}}})
            sender.send(make_event(class_name="select", shape=None))
            data, _ = recv_sock.recvfrom(2048)
            self.assertIn(b'"class_name": "select"', data)
        finally:
            recv_sock.close()


class PipeCommandMappingTest(unittest.TestCase):
    def test_all_seven_classes_map_to_distinct_tokens(self):
        # 델파이7 파이프 프로토콜 7개 고정 명령어 (2026-07-23 팀 확정)
        expected = {
            "move_left": "left", "move_right": "right", "move_up": "up", "move_down": "down",
            "select": "ok", "go_back": "back", "go_home": "home",
        }
        self.assertEqual(PIPE_COMMAND_BY_CLASS_NAME, expected)
        self.assertEqual(len(set(expected.values())), 7)   # 토큰 7개 모두 서로 다름


class FakePipeFile:
    """PipeEventSender가 기대하는 파일류 인터페이스(write/flush)만 흉내낸 테스트 대역."""

    def __init__(self, fail_write=False):
        self.written = []
        self._fail_write = fail_write

    def write(self, data):
        if self._fail_write:
            raise OSError("연결 끊김(테스트)")
        self.written.append(data)

    def flush(self):
        pass


def make_pipe_config(name=r"\\.\pipe\fake", reconnect_backoff_sec=2.0):
    return {"event_output": {"pipe": {"name": name, "reconnect_backoff_sec": reconnect_backoff_sec}}}


class PipeEventSenderTest(unittest.TestCase):
    def test_connects_lazily_and_writes_line_terminated_command(self):
        pipe_file = FakePipeFile()
        open_calls = []

        def fake_open(name):
            open_calls.append(name)
            return pipe_file

        sender = PipeEventSender(make_pipe_config(), open_fn=fake_open, clock=lambda: 0.0)
        sender.send(make_event(class_name="go_home"))
        self.assertEqual(open_calls, [r"\\.\pipe\fake"])
        self.assertEqual(pipe_file.written, [b"home\n"])   # 평문 토큰 + 개행, ascii 인코딩

    def test_each_class_name_writes_its_protocol_token(self):
        for class_name, token in PIPE_COMMAND_BY_CLASS_NAME.items():
            pipe_file = FakePipeFile()
            sender = PipeEventSender(
                make_pipe_config(), open_fn=lambda name: pipe_file, clock=lambda: 0.0
            )
            sender.send(make_event(class_name=class_name))
            self.assertEqual(pipe_file.written, [f"{token}\n".encode("ascii")], class_name)

    def test_connect_failure_respects_backoff_before_retry(self):
        attempts = []

        def failing_open(name):
            attempts.append(name)
            raise OSError("파이프 없음(테스트)")

        now = [0.0]
        sender = PipeEventSender(
            make_pipe_config(reconnect_backoff_sec=2.0), open_fn=failing_open, clock=lambda: now[0],
        )
        sender.send(make_event())          # 1차 시도 — 실패
        sender.send(make_event())          # backoff 안 지남 — 재시도 안 함
        self.assertEqual(len(attempts), 1)
        now[0] = 2.1
        sender.send(make_event())          # backoff 경과 — 재시도
        self.assertEqual(len(attempts), 2)

    def test_send_failure_drops_connection_for_reconnect(self):
        bad_pipe = FakePipeFile(fail_write=True)
        good_pipe = FakePipeFile()
        pipes = [bad_pipe, good_pipe]

        def fake_open(name):
            return pipes.pop(0)

        now = [0.0]
        sender = PipeEventSender(
            make_pipe_config(reconnect_backoff_sec=1.0), open_fn=fake_open, clock=lambda: now[0],
        )
        sender.send(make_event())          # 연결 성공 후 전송 실패 — 연결 버림
        self.assertEqual(good_pipe.written, [])
        now[0] = 1.1
        sender.send(make_event())          # backoff 경과 — 재접속 후 정상 전송
        self.assertEqual(len(good_pipe.written), 1)

    def test_unmapped_class_name_is_skipped_without_writing(self):
        pipe_file = FakePipeFile()
        sender = PipeEventSender(make_pipe_config(), open_fn=lambda name: pipe_file, clock=lambda: 0.0)
        sender.send(make_event(class_name="legacy_unknown"))
        self.assertEqual(pipe_file.written, [])


class CreateEventSenderTest(unittest.TestCase):
    def test_console_mode(self):
        self.assertIsInstance(
            create_event_sender({"event_output": {"mode": "console"}}), ConsoleEventSender
        )

    def test_udp_mode(self):
        config = {"event_output": {"mode": "udp", "udp": {"host": "127.0.0.1", "port": 9999}}}
        self.assertIsInstance(create_event_sender(config), UdpEventSender)

    def test_pipe_mode(self):
        config = make_pipe_config()
        config["event_output"]["mode"] = "pipe"
        self.assertIsInstance(create_event_sender(config), PipeEventSender)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            create_event_sender({"event_output": {"mode": "serial"}})


if __name__ == "__main__":
    unittest.main()
