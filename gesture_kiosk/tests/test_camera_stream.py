"""카메라 새 프레임 동기화(2026-07-20)·자동 복구(2026-07-28) 테스트 — 실제 카메라 없이 검증.

capture_new_frame은 같은 프레임의 중복 추론(카메라 30 FPS < 추론 40+ FPS 낭비)을
막는 장치다: 새 일련번호가 나올 때까지 재우고, 카메라 멈칫 땐 기존 프레임으로 진행.
자동 복구는 끊긴 장치를 버리고 재연결을 반복한다 — init_camera를 대역으로 갈아
끼워 장치 없이 재연결 경로를 검증한다.
"""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import src.capture.camera_stream as camera_stream_module
from src.capture.camera_stream import CameraStream


def _make_stream():
    return CameraStream({"camera": {"device_id": 0}})   # start() 안 함 — 장치 미사용


def _frame(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


class CaptureNewFrameTest(unittest.TestCase):
    def setUp(self):
        # 대기 한도를 짧게 — 멈칫 폴백 테스트가 느려지지 않게
        self._saved_timeout = camera_stream_module.NEW_FRAME_TIMEOUT_SEC
        camera_stream_module.NEW_FRAME_TIMEOUT_SEC = 0.05

    def tearDown(self):
        camera_stream_module.NEW_FRAME_TIMEOUT_SEC = self._saved_timeout

    def test_returns_immediately_when_newer_frame_exists(self):
        stream = _make_stream()
        stream._publish_frame(_frame(10))
        frame, seq = stream.capture_new_frame(last_seq=0)
        self.assertEqual(seq, 1)
        self.assertEqual(frame[0, 0, 0], 10)

    def test_same_seq_waits_then_returns_stale(self):
        # 새 프레임이 안 오면(카메라 멈칫) 한도 후 기존 프레임으로 진행 — seq 불변
        stream = _make_stream()
        stream._publish_frame(_frame(10))
        frame, seq = stream.capture_new_frame(last_seq=1)   # 이미 본 프레임
        self.assertEqual(seq, 1)                            # 그대로 — 다음 호출도 새것 대기
        self.assertEqual(frame[0, 0, 0], 10)

    def test_wakes_up_when_frame_arrives_during_wait(self):
        # 대기 중 캡처 스레드가 게시하면 즉시 깨어난다 (조건변수 통지)
        stream = _make_stream()
        stream._publish_frame(_frame(10))
        timer = threading.Timer(0.01, lambda: stream._publish_frame(_frame(20)))
        timer.start()
        try:
            frame, seq = stream.capture_new_frame(last_seq=1)
        finally:
            timer.cancel()
        self.assertEqual(seq, 2)
        self.assertEqual(frame[0, 0, 0], 20)

    def test_no_frame_ever_raises(self):
        stream = _make_stream()
        with self.assertRaises(RuntimeError):
            stream.capture_new_frame(last_seq=0)


class _FakeCap:
    """cv2.VideoCapture 대역 — N프레임 성공 후 계속 실패(끊김 재현)."""

    def __init__(self, frames_then_fail=None, frame_value=10):
        self._reads_left = frames_then_fail   # None = 무한 성공
        self._value = frame_value
        self.released = False

    def read(self):
        time.sleep(0.01)   # 실제 캡처 페이싱 흉내 — 바쁜 루프 방지
        if self._reads_left is not None:
            if self._reads_left <= 0:
                return False, None
            self._reads_left -= 1
        return True, _frame(self._value)

    def release(self):
        self.released = True


class CameraRecoveryTest(unittest.TestCase):
    """런타임 자동 복구(2026-07-28) — 끊김 감지 → 재연결 → 프레임 재개."""

    def setUp(self):
        self._saved_init = camera_stream_module.init_camera

    def tearDown(self):
        camera_stream_module.init_camera = self._saved_init

    def _config(self):
        # 복구 판정·재시도를 짧게 — 테스트가 느려지지 않게
        return {"camera": {"device_id": 0,
                           "recovery_timeout_sec": 0.1, "recovery_retry_sec": 0.05}}

    def test_reopens_after_signal_loss(self):
        # 3프레임 후 끊기는 카메라 — 복구가 새 핸들(다른 프레임 값)로 갈아끼운다
        dead_cap = _FakeCap(frames_then_fail=3, frame_value=10)
        good_cap = _FakeCap(frame_value=20)
        opened_device_ids = []

        def fake_init_camera(config, device_id=None):
            opened_device_ids.append(device_id)
            return good_cap

        camera_stream_module.init_camera = fake_init_camera
        stream = CameraStream(self._config(), cap=dead_cap).start()
        try:
            deadline_sec = time.monotonic() + 2.0
            seq, value = 0, None
            while time.monotonic() < deadline_sec:
                frame, seq = stream.capture_new_frame(seq)
                value = int(frame[0, 0, 0])
                if value == 20:
                    break
        finally:
            stream.stop()
        self.assertEqual(value, 20)              # 복구된 카메라의 프레임이 흐른다
        self.assertTrue(dead_cap.released)       # 죽은 핸들은 반납됐다
        self.assertEqual(opened_device_ids, [0])  # 같은 장치 번호로 재연결

    def test_stop_interrupts_recovery_wait(self):
        # 재연결이 계속 실패해도 stop()이 복구 대기를 즉시 끊는다 (종료 멈춤 방지)
        def failing_init_camera(config, device_id=None):
            raise RuntimeError("장치 없음")

        camera_stream_module.init_camera = failing_init_camera
        stream = CameraStream(self._config(), cap=_FakeCap(frames_then_fail=1)).start()
        time.sleep(0.3)             # 끊김 판정을 지나 복구 재시도 루프에 진입할 시간
        stream.stop()
        stream._thread.join(timeout=1.0)
        self.assertFalse(stream._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
