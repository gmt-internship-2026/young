"""이벤트 전송 — 확정된 제스처 이벤트를 회사 키오스크 프로그램으로 넘기는 접점.

회사 GUI = **델파이7(Delphi 7) 네이티브 프로그램**. 연동 = **파이프(stdio)** 확정
(2026-07-23 — 회사 요청: 이벤트를 print만 하면 된다): 엔진이 stdout에 텍스트
한 줄을 찍으면, 델파이가 엔진을 자식 프로세스로 띄워 익명 파이프로 줄 단위
수신한다 (docs/델파이7_연동가이드.md — CreateProcess + ReadFile 샘플).
네트워크 연동(UDP·웹소켓)은 2026-07-23 전면 철회 — 포트·방화벽 이슈 소멸.

주의: 로그는 stderr·파일로만 나간다(logger.py의 StreamHandler 기본 = stderr) —
stdout은 이벤트 전용 채널이라 다른 출력이 섞이면 델파이 파서가 오염된다.
"""
import sys

from src.utils.logger import get_logger

logger = get_logger("pipeline")


def build_text_payload(gesture_event):
    """델파이7용 텍스트 규격 — `GESTURE|이벤트|손|신뢰도|시각` 한 줄(ASCII) 바이트.

    구분자 '|'는 값에 절대 등장하지 않는다(이벤트명·손은 영문 고정, 숫자 2종).
    delphi Pos/Copy 또는 4번의 구분자 분리로 파싱 가능. CRLF(#13#10)로 끝나
    수신기가 줄 단위로 읽어도 된다.
    """
    return "GESTURE|{}|{}|{:.2f}|{:.3f}\r\n".format(
        gesture_event.class_name,
        gesture_event.hand_side or "",
        gesture_event.conf,
        gesture_event.ts_sec,
    ).encode("ascii")


class ConsoleEventSender:
    """개발용 — 이벤트를 로그로만 기록한다 (stdout 미사용)."""

    def send(self, gesture_event):
        logger.info(
            "event_output(console): %s (conf=%.2f)",
            gesture_event.class_name,
            gesture_event.conf,
        )


class StdioEventSender:
    """파이프(stdio) 연동 — 이벤트 한 줄을 stdout에 쓰고 즉시 flush한다.

    flush가 핵심: stdout이 파이프에 물리면 블록 버퍼링으로 바뀌어, flush 없이는
    버퍼(수 KB)가 찰 때까지 델파이에 아무것도 도착하지 않는다 (이벤트는 드문드문
    발생하므로 사실상 무한 지연). 바이트로 직접 써서 인코딩 변수도 제거한다.
    """

    def send(self, gesture_event):
        sys.stdout.buffer.write(build_text_payload(gesture_event))
        sys.stdout.buffer.flush()
        logger.info("event_output(stdio): %s (conf=%.2f)",
                    gesture_event.class_name, gesture_event.conf)


def create_event_sender(config):
    """config의 event_output.mode에 맞는 Sender를 만든다."""
    mode = config["event_output"]["mode"]
    if mode == "stdio":
        return StdioEventSender()
    if mode == "console":
        return ConsoleEventSender()
    raise ValueError(f"지원하지 않는 event_output.mode: {mode}")
