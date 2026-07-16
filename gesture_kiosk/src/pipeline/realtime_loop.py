"""pipeline 모듈 — 캡처·추론·판정·안내를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (윈도우 + NVIDIA GPU 기준 — 2026-07-15 2차: 포즈 단일 엔진):
  카메라(스레드) → 거울 반전 → 사람 포즈(RTMPose) → 사용자 잠금(person_lock)
  → 동작 판정(gesture_filter: 손목 쓸기 궤적 — 토크백식 1회/2연속 분기)
  → 이벤트 전송 + 음성 안내

2026-07-16: 주민등록증 OCR 기능 제거 — 제스처 집중(사용자 결정). 개인정보
(주민등록번호) 처리 이슈가 함께 소멸했다. 백업: _before_ocr_removal/.

PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import threading
import time

from src.announce.announcer import Announcer
from src.capture.camera_stream import CameraStream
from src.inference.pose_estimator import PoseEstimator
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.person_lock import PersonLock
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_debug_panel, draw_person_lock, draw_status

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5


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
        self.announcer = None          # demo_server의 POST /announce가 사용한다

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
    camera = CameraStream(config).start()
    preprocessor = Preprocessor(config)
    pose_estimator = PoseEstimator(config)   # 유일한 추론 모델 — 모든 판정의 입력

    first_frame = camera.capture_frame()
    frame_height_px, frame_width_px = first_frame.shape[:2]
    person_lock = PersonLock(config, frame_width_px, frame_height_px)
    gesture_filter = GestureFilter(config)
    event_sender = create_event_sender(config)
    announcer = Announcer(config)
    state.announcer = announcer

    min_loop_interval_sec = 1.0 / config["model"]["max_infer_fps"]

    state.is_running = True

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame = camera.capture_frame()
            input_tensor = preprocessor.preprocess_frame(frame)

            persons = pose_estimator.infer(input_tensor)
            person_lock.update(input_tensor, persons)
            state.is_user_locked = (
                person_lock.enabled and person_lock.locked_person is not None
            )

            # 쓸기 판정용 추적점(손목 — 없으면 팔꿈치) — x·y 모두 프레임 폭으로 나눈
            # 등방 좌표 (어깨너비 정규화와 단위 일치, 2026-07-16)
            swipe_points_ratio = {
                side: None if info is None
                else (info[0], (info[1][0] / frame_width_px, info[1][1] / frame_width_px))
                for side, info in person_lock.user_swipe_points().items()
            }
            gesture_event = gesture_filter.filter_signals(
                swipe_points_ratio, person_lock.user_shoulder_width_ratio()
            )
            state.debug = gesture_filter.debug

            if gesture_event is not None:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)
                announcer.on_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_person_lock(input_tensor, person_lock)
            annotated = draw_debug_panel(annotated, state.debug)
            overlay_event = state.last_event
            if overlay_event is not None and (
                time.monotonic() - overlay_event.ts_sec > EVENT_OVERLAY_HOLD_SEC
            ):
                overlay_event = None
            annotated = draw_status(annotated, state.infer_fps, overlay_event)
            state.update_frame(annotated)

            # FPS 상한 — 개발 PC에서 200+ FPS로 도는 낭비를 막는다
            elapsed_sec = time.monotonic() - loop_start_sec
            if elapsed_sec < min_loop_interval_sec:
                time.sleep(min_loop_interval_sec - elapsed_sec)

    threading.Thread(target=_inference_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
