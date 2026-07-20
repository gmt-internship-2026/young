"""이벤트 전송 — 확정된 제스처 이벤트를 회사 키오스크 프로그램으로 넘기는 접점.

회사 GUI = **델파이7(Delphi 7) 네이티브 프로그램**으로 확정(2026-07-20 사용자).
델파이7은 JSON 표준 지원이 없어 UDP **텍스트 포맷**(`GESTURE|...` 한 줄 —
Pos/Copy로 파싱)을 기본 연동 규격으로 쓴다: udp.format=text.
수신 샘플 코드는 docs/델파이7_연동가이드.md 참고 (Indy TIdUDPServer).
JSON 포맷도 유지한다 — 수신기가 파서를 갖춘 경우 udp.format=json.
"""
import json
import socket

from src.utils.logger import get_logger

logger = get_logger("pipeline")


def build_json_payload(gesture_event):
    """JSON 규격 — {"class_name","conf","ts_sec"[,"hand_side"][,"data"]} 바이트."""
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
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def build_text_payload(gesture_event):
    """델파이7용 텍스트 규격 — `GESTURE|이벤트|손|신뢰도|시각` 한 줄(ASCII).

    구분자 '|'는 값에 절대 등장하지 않는다(이벤트명·손은 영문 고정, 숫자 2종).
    delphi Pos/Copy 또는 4번의 구분자 분리로 파싱 가능. 개행(#13#10)으로 끝나
    수신기가 줄 단위로 읽어도 된다.
    """
    return "GESTURE|{}|{}|{:.2f}|{:.3f}\r\n".format(
        gesture_event.class_name,
        gesture_event.hand_side or "",
        gesture_event.conf,
        gesture_event.ts_sec,
    ).encode("ascii")


class ConsoleEventSender:
    """예시 구현 1 — 이벤트를 로그로만 기록한다."""

    def send(self, gesture_event):
        logger.info(
            "event_output(console): %s (conf=%.2f)",
            gesture_event.class_name,
            gesture_event.conf,
        )


class UdpEventSender:
    """예시 구현 2 — 이벤트를 UDP 데이터그램으로 전송한다.

    udp.format: text(델파이7 기본 — build_text_payload 규격) | json.
    회사 프로그램(델파이7)은 Indy TIdUDPServer로 같은 포트를 열어 수신한다.
    """

    def __init__(self, config):
        udp = config["event_output"]["udp"]
        self._addr = (udp["host"], udp["port"])
        payload_format = udp.get("format", "json")   # 구 config(키 없음) = json 유지
        if payload_format not in ("json", "text"):
            raise ValueError(f"지원하지 않는 event_output.udp.format: {payload_format}")
        self._build_payload = (
            build_text_payload if payload_format == "text" else build_json_payload
        )
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, gesture_event):
        self._sock.sendto(self._build_payload(gesture_event), self._addr)
        logger.info("event_output(udp %s:%s): %s", *self._addr, gesture_event.class_name)


def create_event_sender(config):
    """config의 event_output.mode에 맞는 Sender를 만든다."""
    mode = config["event_output"]["mode"]
    if mode == "udp":
        return UdpEventSender(config)
    if mode == "console":
        return ConsoleEventSender()
    raise ValueError(f"지원하지 않는 event_output.mode: {mode}")
