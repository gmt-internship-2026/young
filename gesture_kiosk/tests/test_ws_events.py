"""웹소켓 이벤트 브로드캐스트(2026-07-21 전환) 테스트 — 스레드 경계 전달 로직 검증.

실제 uvicorn·웹소켓 왕복은 무겁고 포트 의존이라, 여기서는 PipelineState의
구독 레지스트리(등록/해제/전파)를 즉시 실행 가짜 루프로 검증한다.
(엔드투엔드 왕복은 커밋 게이트에서 uvicorn 스모크로 별도 확인 — 작업내역서 참고)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.event_sender import build_text_payload
from src.pipeline.realtime_loop import PipelineState
from src.postprocess.gesture_filter import GestureEvent


class _ImmediateLoop:
    """asyncio 루프 대역 — call_soon_threadsafe를 그 자리에서 실행한다."""

    def call_soon_threadsafe(self, fn, *args):
        fn(*args)


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


class BroadcastTest(unittest.TestCase):
    def test_broadcast_reaches_all_listeners(self):
        state = PipelineState()
        loop = _ImmediateLoop()
        q1, q2 = _FakeQueue(), _FakeQueue()
        state.add_event_listener(loop, q1)
        state.add_event_listener(loop, q2)
        state.broadcast_event("GESTURE|select|left|1.00|1.234\r\n")
        self.assertEqual(q1.items, ["GESTURE|select|left|1.00|1.234\r\n"])
        self.assertEqual(q2.items, q1.items)

    def test_removed_listener_stops_receiving(self):
        state = PipelineState()
        loop = _ImmediateLoop()
        q1, q2 = _FakeQueue(), _FakeQueue()
        state.add_event_listener(loop, q1)
        state.add_event_listener(loop, q2)
        state.remove_event_listener(q1)
        state.broadcast_event("GESTURE|move_left|right|1.00|2.000\r\n")
        self.assertEqual(q1.items, [])
        self.assertEqual(len(q2.items), 1)

    def test_no_listener_is_no_op(self):
        PipelineState().broadcast_event("GESTURE|go_home||1.00|3.000\r\n")   # 예외 없이 통과

    def test_ws_payload_identical_to_udp_text(self):
        # 웹소켓 메시지 = UDP 텍스트 규격과 바이트 동일 — 델파이 파서 한 벌 보장
        event = GestureEvent(class_name="select", conf=1.0, ts_sec=12345.6789, hand_side="left")
        self.assertEqual(
            build_text_payload(event).decode("ascii"),
            "GESTURE|select|left|1.00|12345.679\r\n",
        )


if __name__ == "__main__":
    unittest.main()
