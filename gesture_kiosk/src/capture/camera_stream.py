"""capture 모듈 — USB 웹캠에서 프레임을 실시간으로 읽어 온다 (기획서 2.2).

캡처는 전용 스레드에서 돌리고(기획서 3.2 멀티스레딩), capture_frame()은
항상 가장 최신 프레임을 돌려준다. 추론이 느려도 오래된 프레임이 쌓이지 않는다.
"""
import sys
import threading
import time

import cv2

from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter

logger = get_logger("capture")

FIRST_FRAME_TIMEOUT_SEC = 5.0


def init_camera(config):
    """config 기준으로 카메라 장치를 열어 cv2.VideoCapture를 돌려준다."""
    device_id = config["camera"]["device_id"]
    # 리눅스는 V4L2가 안정적. 윈도우는 MSMF가 열리는 데 수십 초 걸리는 장치가 있어
    # 기본을 DSHOW로 두고 config(camera.windows_backend)로 바꿀 수 있게 한다
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    elif sys.platform.startswith("win"):
        backend_name = config["camera"].get("windows_backend", "auto")
        windows_backends = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
        if backend_name in windows_backends:
            cap = cv2.VideoCapture(device_id, windows_backends[backend_name])
        else:
            cap = cv2.VideoCapture(device_id)
    else:
        cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(device_id={device_id})를 열 수 없습니다")
    # 무압축(YUY2) 1280x720은 USB 대역폭 한계로 캡처가 ~5 FPS에 묶인다 — MJPG 기본
    fourcc = config["camera"].get("fourcc", "mjpg")
    if fourcc and fourcc != "auto":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width_px"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height_px"])
    return cap


class CameraStream:
    """카메라 캡처 스레드. capture_frame()으로 최신 프레임(BGR)을 얻는다."""

    def __init__(self, config):
        self._config = config
        self._cap = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self.is_running = False
        self.fps_meter = FpsMeter()

    def start(self):
        self._cap = init_camera(self._config)
        self.is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("카메라 캡처 스레드 시작 (device_id=%s)", self._config["camera"]["device_id"])
        return self

    def _capture_loop(self):
        while self.is_running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            with self._frame_lock:
                self._frame = frame
            self.fps_meter.update()

    def capture_frame(self):
        """최신 프레임(np.ndarray, BGR)을 돌려준다. 첫 프레임은 잠시 대기한다."""
        deadline_sec = time.monotonic() + FIRST_FRAME_TIMEOUT_SEC
        while True:
            with self._frame_lock:
                if self._frame is not None:
                    return self._frame.copy()
            if time.monotonic() > deadline_sec:
                raise RuntimeError("카메라에서 프레임을 받지 못했습니다 (연결/장치 번호 확인)")
            time.sleep(0.01)

    def stop(self):
        self.is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
