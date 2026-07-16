"""이벤트 전송 — 확정된 제스처 이벤트를 회사 키오스크 프로그램으로 넘기는 접점.

TODO(기획서 9장 №7·№8): 회사 프로그램의 수신 규격(소켓/시리얼/공유메모리)이
확정되면 이 파일에 전용 Sender를 추가하고 config의 event_output.mode로 선택한다.
그 전까지는 console(로그 기록)과 udp(JSON 데이터그램) 예시 구현으로 동작한다.
"""
import json
import socket

from src.utils.logger import get_logger

logger = get_logger("pipeline")


class ConsoleEventSender:
    """예시 구현 1 — 이벤트를 로그로만 기록한다."""

    def send(self, gesture_event):
        logger.info(
            "event_output(console): %s (conf=%.2f)",
            gesture_event.class_name,
            gesture_event.conf,
        )


class UdpEventSender:
    """예시 구현 2 — 이벤트를 JSON으로 UDP 전송한다.

    회사 프로그램이 같은 규격(JSON: class_name/conf/ts_sec)을 수신하면
    코드 수정 없이 config의 host/port만 바꿔 연동할 수 있다.
    """

    def __init__(self, config):
        udp = config["event_output"]["udp"]
        self._addr = (udp["host"], udp["port"])
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, gesture_event):
        payload = {
            "class_name": gesture_event.class_name,
            "conf": round(gesture_event.conf, 4),
            "ts_sec": gesture_event.ts_sec,
        }
        if gesture_event.hand_side is not None:
            payload["hand_side"] = gesture_event.hand_side
        if gesture_event.data is not None:
            # 로그에는 어떤 경우에도 payload 내용물을 남기지 않는다 (개인정보 원칙)
            payload["data"] = gesture_event.data
        self._sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), self._addr)
        logger.info("event_output(udp %s:%s): %s", *self._addr, gesture_event.class_name)


def create_event_sender(config):
    """config의 event_output.mode에 맞는 Sender를 만든다."""
    mode = config["event_output"]["mode"]
    if mode == "udp":
        return UdpEventSender(config)
    if mode == "console":
        return ConsoleEventSender()
    raise ValueError(f"지원하지 않는 event_output.mode: {mode}")
