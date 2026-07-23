"""이벤트 전송 — 확정된 제스처 이벤트를 회사 키오스크 프로그램(델파이7)으로 넘기는 접점.

2026-07-23 확정(팀 결정) — 델파이7과의 실연동 전송 방식은 **네임드 파이프**로 정했다.
델파이7이 파이프 **서버**로 접속을 받고, 이 엔진이 **클라이언트**로 접속해 전송한다
(GMtech_project가 검토했던 WebSocket안과 같은 방향 — Delphi가 서버, 엔진이 클라이언트).
프로토콜은 JSON이 아니라 **평문 명령어 7개 고정**(up/down/left/right/home/back/ok) —
델파이 쪽은 파이프로 들어온 한 줄(개행 구분)을 그대로 명령어로 인식해 실행한다고 팀장이
확인. class_name -> 명령어 매핑은 PIPE_COMMAND_BY_CLASS_NAME 참고 — 우리 7개 제스처
클래스(move_up/move_down/move_left/move_right/select/go_back/go_home)가 프로토콜
7개 토큰과 1:1 대응이라 매핑 누락 걱정이 없다.

console/udp는 개발용 예시 구현으로 남겨둔다(다른 수신 규격이 필요해지면 그때 활성화).
"""
import json
import socket
import time

from src.utils.logger import get_logger

logger = get_logger("pipeline")

# 델파이7 파이프 프로토콜 — 7개 고정 명령어 (2026-07-23 팀 확정). 우리 쪽 class_name은
# 그대로 로그·디버깅에 남기고, 파이프로는 이 평문 토큰만 내보낸다
PIPE_COMMAND_BY_CLASS_NAME = {
    "move_left": "left",
    "move_right": "right",
    "move_up": "up",
    "move_down": "down",
    "select": "ok",
    "go_back": "back",
    "go_home": "home",
}


class ConsoleEventSender:
    """예시 구현 1 — 이벤트를 로그로만 기록한다."""

    def send(self, gesture_event):
        logger.info(
            "event_output(console): %s (conf=%.2f)",
            gesture_event.class_name,
            gesture_event.conf,
        )


class UdpEventSender:
    """예시 구현 2 — 이벤트를 JSON으로 UDP 전송한다 (개발·디버깅용 — 실연동은 PipeEventSender)."""

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
        if gesture_event.shape is not None:
            payload["shape"] = gesture_event.shape
        if gesture_event.data is not None:
            payload["data"] = gesture_event.data
        self._sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), self._addr)
        logger.info("event_output(udp %s:%s): %s", *self._addr, gesture_event.class_name)


class PipeEventSender:
    """실연동 구현 — 델파이7 네임드 파이프로 평문 명령어(up/down/left/right/home/back/ok)를
    전송한다 (2026-07-23 팀 결정).

    델파이7이 파이프 서버, 이 엔진이 클라이언트로 접속한다. 파이썬 표준 open()이 윈도우
    네임드 파이프 경로(\\\\.\\pipe\\이름)도 일반 파일처럼 열 수 있어(CreateFileW 경유),
    별도 라이브러리(pywin32 등) 없이 연결한다 — 팀장 확인: 델파이 쪽은 파이프에 들어온
    한 줄만 보고 알아서 인식·실행하므로 JSON 등 포맷이 필요 없다.

    연결이 끊겨도 파이프라인 자체는 죽지 않게: send() 실패 시 연결을 버리고
    reconnect_backoff_sec 간격으로만 재접속을 시도한다(끊긴 채로 매 프레임 재시도해
    추론 루프를 막지 않기 위한 최소 간격 — WebSocketEventSender와 같은 패턴,
    GMtech_project gesture_kiosk 참고).

    open_fn·clock은 단위 테스트 주입용(PersonLock.sharpness_fn과 같은 패턴) — 실제
    파이프 서버 없이 재접속 backoff·전송 실패 경로를 검증한다.
    """

    def __init__(self, config, open_fn=None, clock=time.monotonic):
        pipe = config["event_output"]["pipe"]
        self._pipe_name = pipe["name"]
        self._reconnect_backoff_sec = pipe["reconnect_backoff_sec"]
        # "r+b": 델파이 CreateNamedPipe가 기본값인 duplex(PIPE_ACCESS_DUPLEX)로 만들었다는
        # 전제 — 인바운드 전용(PIPE_ACCESS_INBOUND)으로 만들었다면 "wb"로 바꿔야 열린다.
        # ★TODO(팀 확인 필요): 델파이 쪽 CreateNamedPipe 접근 모드 확정되면 재확인할 것
        self._open_fn = open_fn or (lambda name: open(name, "r+b", buffering=0))
        self._clock = clock
        self._pipe_file = None
        self._last_connect_attempt_sec = None

    def _ensure_connected(self, now_sec):
        if self._pipe_file is not None:
            return True
        if (self._last_connect_attempt_sec is not None
                and now_sec - self._last_connect_attempt_sec < self._reconnect_backoff_sec):
            return False   # 재접속 최소 간격 미경과 — 이번 프레임은 그냥 건너뛴다
        self._last_connect_attempt_sec = now_sec
        try:
            self._pipe_file = self._open_fn(self._pipe_name)
            logger.info("event_output(pipe): 연결됨 (%s)", self._pipe_name)
            return True
        except OSError as err:
            logger.warning("event_output(pipe): 연결 실패 — %s (%s)", self._pipe_name, err)
            return False

    def send(self, gesture_event):
        if not self._ensure_connected(self._clock()):
            return
        command = PIPE_COMMAND_BY_CLASS_NAME.get(gesture_event.class_name)
        if command is None:
            # classes 설정에 새 값이 추가됐는데 매핑이 안 갱신된 경우 — 조용히 누락시키지 않는다
            logger.warning(
                "event_output(pipe): class_name '%s'에 대응하는 명령어가 없음 (프로토콜 매핑 갱신 필요)",
                gesture_event.class_name,
            )
            return
        try:
            self._pipe_file.write((command + "\n").encode("ascii"))
            self._pipe_file.flush()
            logger.info("event_output(pipe %s): %s -> %s",
                        self._pipe_name, gesture_event.class_name, command)
        except OSError as err:
            logger.warning("event_output(pipe): 전송 실패 — %s", err)
            self._pipe_file = None   # 다음 send()가 재접속을 시도하게 버린다


def create_event_sender(config):
    """config의 event_output.mode에 맞는 Sender를 만든다."""
    mode = config["event_output"]["mode"]
    if mode == "udp":
        return UdpEventSender(config)
    if mode == "pipe":
        return PipeEventSender(config)
    if mode == "console":
        return ConsoleEventSender()
    raise ValueError(f"지원하지 않는 event_output.mode: {mode}")
