"""announce 모듈 — 토크백(보이스오버): 현재 위치·기능을 음성으로 안내한다.

역할 분담(회사 프로그램 연동 계약):
- 제스처 이벤트의 공통 안내(이동·선택·처음으로 등)는 엔진이 자동으로 읽는다
  (config announce.event_templates)
- 화면 맥락 안내(지금 포커스된 버튼이 무엇인지)는 화면 구조를 아는 UI가
  POST /announce {"text": "..."}로 요청한다 — demo_ui가 시연 구현이다

백엔드:
- tts : pyttsx3 — 윈도우 SAPI(한국어 보이스) / 맥 nsss. 오프라인·설치 의존성 없음
- log : 음성 대신 로그로만 출력 (개발·테스트용)

TTS 엔진은 전용 워커 스레드에서만 만든다·부른다 (SAPI COM 제약).
안내가 밀리면 최신 것만 남긴다 — 시각장애인 사용자에게 옛 안내를 늦게
읽어 주는 것이 무(無)안내보다 나쁘기 때문이다.
"""
import queue
import threading

from src.utils.logger import get_logger

logger = get_logger("announce")

QUEUE_MAX_COUNT = 4
STOP_SENTINEL = None


class Announcer:
    def __init__(self, config):
        announce = config["announce"]
        self.enabled = announce["enabled"]
        self._backend = announce["backend"]
        self._rate_wpm = announce["rate_wpm"]
        self._volume = announce["volume"]
        self._event_templates = announce["event_templates"]

        self._queue = queue.Queue(maxsize=QUEUE_MAX_COUNT)
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()

    # ----- 외부 인터페이스 -----

    def on_event(self, gesture_event):
        """확정 이벤트를 템플릿 문구로 안내한다. 템플릿에 없는 이벤트(레거시 등)는 침묵."""
        template = self._event_templates.get(gesture_event.class_name)
        if template is not None:
            self.announce(template)

    def announce(self, text):
        """문구 1건을 안내 큐에 넣는다. 큐가 차 있으면 가장 오래된 것을 버린다."""
        if not self.enabled or not text:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            try:
                self._queue.get_nowait()  # 최신 우선 — 오래된 안내를 버린다
            except queue.Empty:
                pass
            self._queue.put_nowait(text)

    def stop(self):
        if self._thread is not None:
            self._queue.put(STOP_SENTINEL)
            self._thread.join(timeout=2.0)

    # ----- 워커 스레드 -----

    def _worker_loop(self):
        engine = self._init_tts_engine() if self._backend == "tts" else None
        while True:
            text = self._queue.get()
            if text is STOP_SENTINEL:
                break
            if engine is not None:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    logger.exception("TTS 재생 실패 — 이후 안내는 로그로만 출력합니다")
                    engine = None
            if engine is None:
                logger.info("announce(log): %s", text)

    def _init_tts_engine(self):
        """pyttsx3 엔진을 워커 스레드 안에서 초기화한다. 실패 시 로그 백엔드로 폴백."""
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate_wpm)
            engine.setProperty("volume", self._volume)
            logger.info("TTS 초기화 완료 (pyttsx3, rate=%d)", self._rate_wpm)
            return engine
        except Exception:
            logger.warning("pyttsx3 초기화 실패 — announce는 로그로만 출력합니다 (backend=log 동작)")
            return None
