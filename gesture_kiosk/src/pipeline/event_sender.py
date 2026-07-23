"""이벤트 전송 — 확정된 제스처 이벤트를 회사 키오스크 프로그램(델파이7)으로 넘기는 접점.

2026-07-23 확정(회사 요청 — "엔진은 이벤트를 print만 하면 된다") — 연동 방식은
**파이프(stdio)**: 델파이7이 이 엔진을 자식 프로세스로 실행하고, 엔진이 stdout에
찍는 텍스트 한 줄을 익명 파이프로 그대로 읽는다. 네트워크(UDP·웹소켓)는 이 결정과
함께 철회 — 포트·방화벽 이슈가 사라진다. 같은 델파이7 프로그램에 붙는 시선추적
엔진도 같은 방식(print+flush)이라 형식을 맞췄다. 자매 코드베이스 GMtech_project
(다른 팀원 커밋)의 `docs/델파이7_연동가이드.md` · `delphi_ui/`가 델파이 쪽 수신
구현 예제다 — CreateProcess로 자식 프로세스를 띄우고 ReadFile로 stdout을 줄
단위로 읽는다(Pos(#13#10, ...) 기준 — CRLF로 끝나야 한다).

**중요: 로그는 stderr·파일로만 나가야 한다** (`src/utils/logger.py`의
`StreamHandler()`는 인자 없이 만들면 표준적으로 stderr — 확인됨). stdout은
이벤트 전용 채널이라 다른 출력이 섞이면 델파이 파서가 오염된다.
"""
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def build_gesture_line(gesture_event):
    """델파이7 텍스트 규격 — `GESTURE|이벤트|손|신뢰도|시각` 한 줄(CRLF 없는 본문).

    구분자 '|'는 값에 절대 등장하지 않는다(이벤트명은 영문 고정, 손은 영문 고정 또는
    빈 문자열, 나머지 숫자 2종). 델파이 쪽은 Pos/Copy로 4번 분리해 파싱한다
    (docs/델파이7_연동가이드.md 참고).

    손(3번 필드) — 이 저장소(young)는 MediaPipe로 손 1개만 보고 좌/우 손 정체성을
    구분하지 않는다(GestureEvent.shape는 손의 좌/우가 아니라 손 모양(point/fist)이라
    다른 개념 — 이 필드에 넣으면 의미가 어긋난다). 그래서 항상 빈 문자열을 보낸다 —
    델파이 쪽 파싱 코드도 현재 이 필드를 실제로 쓰지 않는다(이벤트명만으로 분기).
    """
    return "GESTURE|{}|{}|{:.2f}|{:.3f}".format(
        gesture_event.class_name, "", gesture_event.conf, gesture_event.ts_sec,
    )


class ConsoleEventSender:
    """개발용 — 이벤트를 로그로만 기록한다 (stdout 미사용)."""

    def send(self, gesture_event):
        logger.info(
            "event_output(console): %s (conf=%.2f)",
            gesture_event.class_name,
            gesture_event.conf,
        )


class UdpEventSender:
    """레거시 예시 구현 — 이벤트를 JSON으로 UDP 전송한다 (실연동 규격 아님, 개발용으로만 유지)."""

    def __init__(self, config):
        import socket

        udp = config["event_output"]["udp"]
        self._addr = (udp["host"], udp["port"])
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, gesture_event):
        import json

        payload = {
            "class_name": gesture_event.class_name,
            "conf": round(gesture_event.conf, 4),
            "ts_sec": gesture_event.ts_sec,
        }
        if gesture_event.shape is not None:
            payload["shape"] = gesture_event.shape
        if gesture_event.data is not None:
            payload["data"] = gesture_event.data
        self._sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), self._addr)
        logger.info("event_output(udp %s:%s): %s", *self._addr, gesture_event.class_name)


class StdioEventSender:
    """실연동 구현 — stdout에 이벤트 한 줄을 print하고 즉시 flush한다 (2026-07-23 회사 확정).

    flush가 핵심: stdout이 파이프에 물리면 블록 버퍼링으로 바뀌어, flush 없이는
    버퍼가 찰 때까지 델파이에 아무것도 도착하지 않는다(이벤트가 드문드문 발생하므로
    사실상 무한 지연) — flush=True로 매번 즉시 내보낸다.

    print()의 기본 줄바꿈("\\n")은 윈도우에서 sys.stdout이 텍스트 모드일 때
    os.linesep("\\r\\n")으로 자동 변환된다 — 이 저장소는 윈도우 전용 타깃이라
    이 변환에 기대어 CRLF를 만든다(문자열에 "\\r\\n"을 직접 넣지 않는다 — 넣으면
    "\\n" 부분이 또 변환돼 "\\r\\r\\n"이 되는 이중 변환 버그가 난다).
    """

    def send(self, gesture_event):
        print(build_gesture_line(gesture_event), flush=True)
        logger.info("event_output(stdio): %s (conf=%.2f)",
                    gesture_event.class_name, gesture_event.conf)


def create_event_sender(config):
    """config의 event_output.mode에 맞는 Sender를 만든다."""
    mode = config["event_output"]["mode"]
    if mode == "udp":
        return UdpEventSender(config)
    if mode == "stdio":
        return StdioEventSender()
    if mode == "console":
        return ConsoleEventSender()
    raise ValueError(f"지원하지 않는 event_output.mode: {mode}")
