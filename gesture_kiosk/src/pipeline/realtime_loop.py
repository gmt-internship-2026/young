"""pipeline 모듈 — 캡처·추론·판정·전송을 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (2026-07-29 포즈 제거 — 손 단독 추론, hand_select.py 참고):
  카메라(스레드) → 거울 반전 → 손 랜드마크(MediaPipe — 유일한 추론 모델)
  → 사용자 손 선별(hand_select: 크기+연속성, 손 실측 자)
  → 동작 판정(gesture_filter: 손 모양 래치 + 첫 선 궤적 4방향)
  → 이벤트 전송(stdio — stdout 한 줄, 델파이가 파이프로 수신)

2026-07-29 포즈(ONNX Runtime + rtmlib) 제거(사용자 결정): 제스처 판정은 손
데이터만으로 끝난다 — 포즈가 하던 잠금·자(尺)·게이트는 hand_select가 손 기반으로
대체. 추론 엔진 = MediaPipe 내장 TFLite(XNNPACK) 하나 (Apache-2.0 — 라이선스
B안 유지: 상업 허용·카피레프트 없음).

2026-07-23: 웹소켓·UDP·데모 웹 서버 제거(회사 결정 — 네트워크 철회, print 연동).
디버그는 run_demo.py --debug 로컬 창(cv2)이 담당한다.

PipelineState가 디버그 창과 공유되는 유일한 상태 저장소다.
"""
import threading
import time

from src.capture.camera_probe import select_camera
from src.capture.camera_stream import CameraStream
from src.utils.env_report import log_environment
from src.inference.hand_tracker import HandTracker
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.hand_select import HandSelector
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_debug_panel, draw_status, draw_user_hands

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5


def resolve_loop_interval_sec(model_config, is_active):
    """추론 루프의 최소 간격 — 활성(손 사용 중)일 땐 max_infer_fps, 유휴일 땐
    idle_infer_fps로 낮춰 CPU·전력을 아낀다 (2026-07-20 추론 부담 절감).
    idle_infer_fps 미설정 브랜치는 종전대로 상시 max_infer_fps."""
    max_fps = model_config["max_infer_fps"]
    idle_fps = model_config.get("idle_infer_fps", max_fps)
    return 1.0 / (max_fps if is_active else min(idle_fps, max_fps))


class PipelineState:
    """추론 결과·성능 수치를 스레드 안전하게 공유한다."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_frame = None
        self.capture_fps = 0.0
        self.infer_fps = 0.0
        self.last_event = None
        self.event_log = []
        self.is_running = False
        self.is_user_locked = False
        self.debug = {}                # 판정 계기판(gesture_filter.debug) — 실기 튜닝용
        self._viewer_count = 0         # 디버그 창 시청자 수 — 0이면 오버레이 렌더링 생략

    def add_viewer(self):
        """디버그 창 열림 — 다음 루프부터 오버레이를 그린다 (2026-07-20 최적화)."""
        with self._lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self._lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    @property
    def has_viewer(self):
        return self._viewer_count > 0

    def update_frame(self, frame):
        with self._lock:
            self._latest_frame = frame

    def get_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def append_event(self, gesture_event):
        with self._lock:
            self.last_event = gesture_event
            self.event_log.append(gesture_event)
            if len(self.event_log) > EVENT_LOG_MAX_COUNT:
                self.event_log.pop(0)


def run_pipeline(config):
    """파이프라인 전체를 조립해 시작하고 PipelineState를 돌려준다 (기획서 4.6 계약)."""
    state = PipelineState()
    log_environment(config)   # 어느 하드웨어에서 돈 기록인지 로그 첫머리에 남긴다 (2026-07-16)
    # A안: 모델을 먼저 만들고 손 인식 품질로 카메라를 프로브해 메인을 고른다
    # (2026-07-29 포즈 제거 — 프로브 채점도 손 품질 단독)
    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)   # 유일한 추론 모델 (2026-07-29 포즈 제거)
    main_device_id, main_cap = select_camera(config, hand_tracker, preprocessor)
    camera = CameraStream(config, device_id=main_device_id, cap=main_cap).start()

    first_frame = camera.capture_frame()
    frame_height_px, frame_width_px = first_frame.shape[:2]
    hand_selector = HandSelector(config, frame_width_px, frame_height_px)
    gesture_filter = GestureFilter(config)
    event_sender = create_event_sender(config)

    state.is_running = True

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        was_active = True   # 유휴↔활성 전환을 로그로 남기기 위한 직전 상태
        last_frame_seq = 0  # 새 프레임 동기화(2026-07-20) — 같은 프레임 중복 추론 방지
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame, last_frame_seq = camera.capture_new_frame(last_frame_seq)
            input_tensor = preprocessor.preprocess_frame(frame)

            hands = hand_tracker.infer(input_tensor)
            is_engaged = hand_selector.update(hands)
            state.is_user_locked = is_engaged

            # 유휴 판정 — 손이 보이거나 최근 사용 중이면 활성 (idle_infer_fps 절감)
            is_active = bool(hands) or is_engaged
            if is_active != was_active:
                logger.info("추론 %s 전환 (hands=%d)", "활성" if is_active else "유휴",
                            len(hands))
                was_active = is_active

            # 판정용 손 신호(손모양 + 손 중심) — x·y 모두 프레임 폭으로 나눈
            # 등방 좌표 (거리 자 정규화와 단위 일치, 2026-07-16)
            swipe_points_ratio = {
                side: None if info is None
                else (info[0], (info[1][0] / frame_width_px, info[1][1] / frame_width_px))
                for side, info in hand_selector.user_swipe_points().items()
            }
            gesture_event = gesture_filter.filter_signals(
                swipe_points_ratio, hand_selector.hand_scale_ratio(),   # 손 실측 자
                hand_selector.shoulder_line_y_ratio(),   # None — 하단 띠 게이트 폴백
            )
            state.debug = gesture_filter.debug

            if gesture_event is not None:
                event_sender.send(gesture_event)   # stdio: stdout 한 줄 — 델파이 파이프 수신
                state.append_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            # 오버레이(추적점·계기판·상태)는 디버그 창 시청자가 있을 때만 그린다 —
            # 실전(회사 UI는 이벤트만 수신)에서는 매 프레임 그리기·복사가 순수 낭비다
            # (2026-07-20 최적화. 판정·이벤트 경로는 위에서 이미 끝났으므로 무영향)
            if state.has_viewer:
                annotated = draw_user_hands(input_tensor, hand_selector)
                annotated = draw_debug_panel(annotated, state.debug)
                overlay_event = state.last_event
                if overlay_event is not None and (
                    time.monotonic() - overlay_event.ts_sec > EVENT_OVERLAY_HOLD_SEC
                ):
                    overlay_event = None
                annotated = draw_status(annotated, state.infer_fps, overlay_event)
                state.update_frame(annotated)

            # FPS 상한 — 개발 PC에서 200+ FPS로 도는 낭비를 막는다.
            # 유휴(손 없음)일 땐 idle_infer_fps까지 더 낮춘다 (2026-07-20)
            min_loop_interval_sec = resolve_loop_interval_sec(config["model"], is_active)
            elapsed_sec = time.monotonic() - loop_start_sec
            if elapsed_sec < min_loop_interval_sec:
                time.sleep(min_loop_interval_sec - elapsed_sec)

    threading.Thread(target=_inference_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
