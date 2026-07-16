"""pipeline 모듈 — 캡처·추론·판정·안내를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (2026-07-16: 선택 판정에 손 모델 추가):
  카메라(스레드) → 거울 반전 → 사람 포즈(RTMPose) → 사용자 잠금(person_lock)
  → 잠긴 사용자 bbox 크롭 → 손 랜드마크(MediaPipe HandLandmarker)
  → 동작 판정(gesture_filter: 손목 쓸기 궤적 + 손가락 1개 인식)
  → 이벤트 전송 + 음성 안내

PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import threading
import time

from src.announce.announcer import Announcer
from src.capture.camera_stream import CameraStream
from src.inference.hand_estimator import HandEstimator, count_extended_fingers
from src.inference.pose_estimator import PoseEstimator
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.person_lock import PersonLock
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_person_lock, draw_status

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5


def _crop_bbox(frame, bbox):
    """bbox(x1, y1, x2, y2) 픽셀 좌표로 프레임을 잘라낸다 — 프레임 경계로 clamp."""
    h_px, w_px = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_px, int(x2)), min(h_px, int(y2))
    return frame[y1:y2, x1:x2]


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
    pose_estimator = PoseEstimator(config)   # 쓸기·사용자 잠금 — 프레임 전체 추론
    hand_estimator = HandEstimator(config)   # 선택(손가락 인식) — 잠긴 사용자 bbox 크롭만 추론

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
        was_user_locked = False   # 전환(잠금<->해제) 시에만 안내 — 매 프레임 반복 방지
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame = camera.capture_frame()
            input_tensor = preprocessor.preprocess_frame(frame)

            persons = pose_estimator.infer(input_tensor)
            person_lock.update(input_tensor, persons)
            state.is_user_locked = (
                person_lock.enabled and person_lock.locked_person is not None
            )
            if state.is_user_locked != was_user_locked:
                announcer.on_user_lock_change(state.is_user_locked)
                was_user_locked = state.is_user_locked

            # 쓸기 판정용 추적점(손목 — 없으면 팔꿈치) — 프레임 폭/높이 비율 좌표로 넘긴다
            swipe_points_ratio = {
                side: None if info is None
                else (info[0], (info[1][0] / frame_width_px, info[1][1] / frame_height_px))
                for side, info in person_lock.user_swipe_points().items()
            }

            # 선택 판정용 손가락 개수 — 잠긴 사용자 bbox 크롭만 봐서 다른 사람 손을 거른다
            finger_count = None
            if person_lock.locked_person is not None:
                hand_crop = _crop_bbox(input_tensor, person_lock.locked_person.bbox)
                hands = hand_estimator.infer(hand_crop)
                if hands:
                    finger_count = count_extended_fingers(hands[0])

            gesture_event = gesture_filter.filter_signals(swipe_points_ratio, finger_count)

            if gesture_event is not None:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)
                announcer.on_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_person_lock(input_tensor, person_lock, finger_count)
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
