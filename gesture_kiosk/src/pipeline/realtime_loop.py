"""pipeline 모듈 — 캡처·추론·판정·안내를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (윈도우 + NVIDIA GPU 기준 — 2026-07-15 2차: 포즈 단일 엔진):
  카메라(스레드) → 거울 반전 → 사람 포즈(RTMPose) → 사용자 잠금(person_lock)
  → 동작 판정(gesture_filter: 손목 쓸기 궤적 + 고개 꾸벅 2회)
  → 이벤트 전송 + 음성 안내

주민등록증 OCR은 별도 워커 스레드에서 돈다 — EasyOCR 1회가 수백 ms라
추론 루프(30 FPS 목표)를 막지 않게 분리한다. OCR은 UI가 요청할 때만
(state.start_ocr_mode) 원본(반전 없는) 프레임으로 동작한다.

PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import threading
import time

from src.announce.announcer import Announcer
from src.capture.camera_stream import CameraStream
from src.inference.pose_estimator import PoseEstimator
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureEvent, GestureFilter
from src.postprocess.person_lock import PersonLock
from src.utils.logger import get_logger
from src.utils.metrics import FpsMeter
from src.utils.visualize import draw_ocr_mode, draw_person_lock, draw_status

logger = get_logger("pipeline")

EVENT_LOG_MAX_COUNT = 200
EVENT_OVERLAY_HOLD_SEC = 1.5
OCR_IDLE_POLL_SEC = 0.2
ASSUMED_CAMERA_FPS = 30.0  # ocr.interval_frames를 워커의 폴링 주기로 환산할 때의 기준


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
        self._ocr_deadline_sec = None  # None이면 OCR 모드 꺼짐

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

    # ----- OCR 모드 제어 (UI -> 엔진) -----

    def start_ocr_mode(self, timeout_sec):
        with self._lock:
            self._ocr_deadline_sec = time.monotonic() + timeout_sec
        logger.info("OCR 모드 시작 (timeout=%.0f초)", timeout_sec)

    def stop_ocr_mode(self):
        with self._lock:
            self._ocr_deadline_sec = None

    def is_ocr_mode_active(self):
        with self._lock:
            if self._ocr_deadline_sec is None:
                return False
            if time.monotonic() > self._ocr_deadline_sec:
                self._ocr_deadline_sec = None
                return False
            return True


def _start_ocr_worker(state, config, camera, event_sender, announcer):
    """주민등록증 OCR 워커 — OCR 모드일 때만 원본 프레임을 주기적으로 판독한다."""
    from src.ocr.idcard_reader import IdCardReader  # easyocr 의존 — 켠 경우에만 임포트

    reader = IdCardReader(config)
    poll_interval_sec = config["ocr"]["interval_frames"] / ASSUMED_CAMERA_FPS

    def _ocr_loop():
        while state.is_running:
            if not state.is_ocr_mode_active():
                time.sleep(OCR_IDLE_POLL_SEC)
                continue
            frame = camera.capture_frame()  # 원본(반전 없음) — 글자를 읽어야 한다
            try:
                fields = reader.read(frame)
            except Exception:
                logger.exception("OCR 판독 오류 — 모드를 종료합니다")
                state.stop_ocr_mode()
                continue
            if fields is None:
                time.sleep(poll_interval_sec)
                continue
            event = GestureEvent(
                class_name="fill_id_fields",
                conf=fields["conf"],
                ts_sec=time.monotonic(),
                data={"name": fields["name"], "rrn": fields["rrn"]},
            )
            event_sender.send(event)
            state.append_event(event)
            announcer.on_event(event)
            state.stop_ocr_mode()  # 1회 인식이 목적 — 성공 즉시 종료

    threading.Thread(target=_ocr_loop, daemon=True).start()
    logger.info("OCR 워커 시작 (poll=%.2f초)", poll_interval_sec)


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
    ocr_guide_region = config["ocr"]["guide_region_ratio"]

    state.is_running = True
    if config["ocr"]["enabled"]:
        _start_ocr_worker(state, config, camera, event_sender, announcer)

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

            # 쓸기 판정용 추적점(손목 — 없으면 팔꿈치) — 프레임 폭/높이 비율 좌표로 넘긴다
            swipe_points_ratio = {
                side: None if info is None
                else (info[0], (info[1][0] / frame_width_px, info[1][1] / frame_height_px))
                for side, info in person_lock.user_swipe_points().items()
            }
            gesture_event = gesture_filter.filter_signals(
                swipe_points_ratio, person_lock.user_neck_ratio()
            )

            if gesture_event is not None:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)
                announcer.on_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_person_lock(input_tensor, person_lock)
            if state.is_ocr_mode_active():
                annotated = draw_ocr_mode(annotated, ocr_guide_region)
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
