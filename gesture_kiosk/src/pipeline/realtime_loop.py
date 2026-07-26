"""pipeline 모듈 — 캡처·추론·판정·안내를 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (2026-07-23: 동작 판정을 손 모양+이동으로 통합):
  카메라(스레드) → 거울 반전 → 사람 포즈(RTMPose — 잠금 상태면 검출 생략 패스트패스,
  아래 should_refresh 참고, 사용자 잠금 전용) → 사용자 잠금(person_lock)
  → 잠긴 사용자 bbox 크롭 → 손 랜드마크(MediaPipe HandLandmarker — 역시 매 프레임이
  아니라 should_refresh 주기로만 재인식, 그 사이는 마지막 결과 재사용)
  → 동작 판정(gesture_filter: 손 모양(point/fist) + 손 위치 이동 방향을 함께 봐서
  left/right/top/bottom·ok·back·home 확정 — gesture_filter.py 모듈 docstring 참고)
  → 이벤트 전송 + 음성 안내

2026-07-20 유휴 적응 FPS(GMtech feat/think_win_cpu 이식, resolve_loop_interval_sec 참고):
  사람이 안 보이고 잠금도 없는 유휴 상태면 추론 루프 FPS를 idle_infer_fps로 낮춰
  키오스크 대기 시간의 CPU·전력을 아낀다. 사람이 감지되면 즉시 max_infer_fps로 복귀.

PipelineState가 예시 UI 서버와 공유되는 유일한 상태 저장소다.
"""
import os
import threading
import time

from src.announce.announcer import Announcer
from src.capture.camera_stream import CameraStream
from src.inference.hand_estimator import DEFAULT_EXTENDED_RATIO, HandEstimator, count_extended_fingers
from src.inference.pose_estimator import PoseEstimator, pad_bbox
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


def _crop_bbox(frame, bbox):
    """bbox(x1, y1, x2, y2) 픽셀 좌표로 프레임을 잘라낸다 — 프레임 경계로 clamp."""
    h_px, w_px = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w_px, int(x2)), min(h_px, int(y2))
    return frame[y1:y2, x1:x2]


def _hand_point_ratio(bbox, point, frame_width_px):
    """손 랜드마크(bbox 크롭-로컬 픽셀 좌표) -> 전체 프레임 등방 비율 좌표 (gestures.hand_move용).

    hand_estimator.infer()는 _crop_bbox()로 잘라낸 영역 기준 좌표를 돌려주므로,
    원점 오프셋을 _crop_bbox와 똑같이 클램프해서 더해야 프레임 전체 기준으로 맞는다.

    2026-07-24 이식 — x·y 모두 frame_width_px로 정규화한다(예전엔 y를 frame_height_px로
    나눠, 1280x720 카메라 기준 y가 x보다 ~1.78배 민감했다 — person_lock의 어깨너비
    비율과 단위를 맞추는 등방 정규화, gesture_filter 모듈 docstring 참고).
    """
    x1, y1, _, _ = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    return ((x1 + point[0]) / frame_width_px, (y1 + point[1]) / frame_width_px)


def should_refresh(is_active, frames_since_refresh, interval_frames):
    """이번 프레임에 무거운 재계산을 할지(vs 이전 결과 재사용) 판단한다.

    "일정 간격으로만 무거운 계산을 다시 하고 그 사이는 마지막 결과를
    재사용한다"는 같은 판단을 두 곳에 재사용한다(2026-07-16 성능 최적화):
    - 포즈 검출: is_active=잠금 여부. 미잠금(찾는 중)이면 후보가 필요하니
      항상 재계산(True). 잠금 상태면 redetect_interval_frames마다만 전체
      검출(infer)로 재동기화, 나머지는 infer_at_bbox()로 검출을 건너뛴다
      (이 CPU 실측 검출 ~89ms vs bbox 재사용 ~22ms).
    - 손가락 인식: is_active=True 고정(호출부가 이미 "잠금 상태"에서만 부른다) —
      infer_interval_frames마다만 hand_estimator를 다시 돌리고, 나머지는
      마지막 finger_count를 재사용한다(이 CPU 실측: 손 인식 단독 ~32ms —
      포즈 패스트패스와 매 프레임 같이 돌리면 오히려 77.8ms/프레임까지 느려짐).

    순수 함수로 둬 rtmlib·mediapipe 없이 단위 테스트한다.
    """
    if not is_active:
        return True
    return frames_since_refresh >= interval_frames


def resolve_loop_interval_sec(model_config, is_active):
    """추론 루프의 최소 간격 — 활성(사람 감지·잠금)일 땐 max_infer_fps, 유휴일 땐
    idle_infer_fps로 낮춰 CPU·전력을 아낀다 (2026-07-20 GMtech feat/think_win_cpu 이식).
    idle_infer_fps 미설정이면 종전대로 상시 max_infer_fps."""
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
        self.announcer = None          # demo_server의 POST /announce가 사용한다
        self.debug = {}                # 판정 계기판 — gesture_filter.debug 스냅샷(2026-07-22)

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
    event_sender = create_event_sender(config)
    announcer = Announcer(config)
    state.announcer = announcer

    redetect_interval_frames = config["model"]["redetect_interval_frames"]
    bbox_pad_ratio = config["model"]["bbox_pad_ratio"]
    hand_infer_interval_frames = config["hand_model"]["infer_interval_frames"]
    finger_extended_ratio = config["gestures"]["shapes"].get(
        "extended_ratio", DEFAULT_EXTENDED_RATIO
    )
    point_finger_count = config["gestures"]["shapes"]["point_finger_count"]
    fist_finger_count = config["gestures"]["shapes"]["fist_finger_count"]
    # 학습된 손 모양 분류기(2026-07-23) — 설정돼 있으면 count_extended_fingers() 규칙
    # 대신 이걸로 point/fist를 판정한다. scripts/train_hand_shape_classifier.py 참고
    shape_classifier = None
    classifier_weights_path = config["gestures"]["shapes"].get("classifier_weights_path")
    if classifier_weights_path:
        from src.postprocess.hand_shape_classifier import HandShapeClassifier

        shape_classifier = HandShapeClassifier(
            os.path.join(config["root_dir"], classifier_weights_path)
        )
        logger.info("손 모양 학습 분류기 로딩 완료: %s", classifier_weights_path)

    # 학습된 방향 분류기(2026-07-24) — 설정돼 있으면 min_dist_*_ratio/axis_dominance
    # 임계값 비교 대신 이걸로 좌/우/상/하를 판정한다. scripts/train_direction_classifier.py 참고
    direction_classifier = None
    direction_weights_path = config["gestures"]["hand_move"].get("classifier_weights_path")
    if direction_weights_path:
        from src.postprocess.direction_classifier import DirectionClassifier

        direction_classifier = DirectionClassifier(
            os.path.join(config["root_dir"], direction_weights_path)
        )
        logger.info("방향 학습 분류기 로딩 완료: %s", direction_weights_path)

    gesture_filter = GestureFilter(config, direction_classifier=direction_classifier)

    state.is_running = True

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        was_user_locked = False   # 전환(잠금<->해제) 시에만 안내 — 매 프레임 반복 방지
        was_active = True   # 유휴↔활성 전환 로그용 직전 상태 (2026-07-20 유휴 적응 FPS)
        frames_since_redetect = 0   # 잠금 상태에서 마지막 전체 검출 이후 지난 프레임 수
        frames_since_hand_infer = 0   # 마지막 손가락 재인식 이후 지난 프레임 수
        last_finger_count = None      # 재인식을 건너뛴 프레임에 재사용할 캐시
        last_hand_point_ratio = None  # 마지막 손 위치(비율 좌표) — hand_swipe용 캐시
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame = camera.capture_frame()
            input_tensor = preprocessor.preprocess_frame(frame)

            # 잠금 상태면 재동기화 주기가 아닌 한 검출을 건너뛰고 이전 위치만 재추정
            # (2026-07-16 성능 최적화) — was_user_locked는 지난 프레임 끝 시점 상태
            if should_refresh(was_user_locked, frames_since_redetect, redetect_interval_frames):
                persons = pose_estimator.infer(input_tensor)
                frames_since_redetect = 0
            else:
                padded_bbox = pad_bbox(
                    person_lock.locked_person.bbox, bbox_pad_ratio,
                    frame_width_px, frame_height_px,
                )
                persons = pose_estimator.infer_at_bbox(input_tensor, padded_bbox)
                frames_since_redetect += 1

            person_lock.update(input_tensor, persons)
            # 어깨너비/어깨선 비율(2026-07-24 이식) — gesture_filter의 방향 판정 임계
            # 정규화(body_scale)·들어올리기 게이트에 쓴다. 미검출 시 None(마지막 값 유지)
            shoulder_width_ratio = person_lock.user_shoulder_width_ratio()
            shoulder_line_y_ratio = person_lock.user_shoulder_line_y_ratio()
            state.is_user_locked = (
                person_lock.enabled and person_lock.locked_person is not None
            )
            if state.is_user_locked != was_user_locked:
                announcer.on_user_lock_change(state.is_user_locked)
                if state.is_user_locked:
                    frames_since_hand_infer = hand_infer_interval_frames  # 잠기면 즉시 1회 인식
                else:
                    last_finger_count = None      # 해제 — 이전 사용자의 손가락 캐시는 버린다
                    last_hand_point_ratio = None  # 손 위치 캐시도 함께 버린다
                was_user_locked = state.is_user_locked

            # 유휴 판정 — 사람이 보이거나 잠금이 살아 있으면 활성 (2026-07-20 유휴 적응 FPS)
            is_active = bool(persons) or state.is_user_locked
            if is_active != was_active:
                logger.info(
                    "추론 %s 전환 (persons=%d, locked=%s)",
                    "활성" if is_active else "유휴", len(persons), state.is_user_locked,
                )
                was_active = is_active

            # 손 신호 — 손 모양(point/fist)·손 위치 이동 판정용. 잠긴 사용자 bbox 크롭만
            # 봐서 다른 사람 손을 거른다. 매 프레임 재인식하지 않고 infer_interval_frames
            # 마다만(2026-07-16 성능 최적화) — 건너뛴 프레임은 반드시 last_finger_count/
            # last_hand_point_ratio를 재사용한다(finger_count를 None으로 덮으면 손 모양이
            # 매번 끊긴 것으로 리셋돼 이동 궤적이 매번 날아간다)
            finger_count = None
            hand_point_ratio = None
            if person_lock.locked_person is not None:
                if should_refresh(True, frames_since_hand_infer, hand_infer_interval_frames):
                    hand_crop = _crop_bbox(input_tensor, person_lock.locked_person.bbox)
                    hands = hand_estimator.infer(hand_crop)
                    if hands:
                        if shape_classifier is not None:
                            shape_label = shape_classifier.classify(hands[0])
                            # count_extended_fingers()와 같은 int 규약으로 맞춘다 — gesture_filter가
                            # point_finger_count/fist_finger_count와 비교하는 로직을 그대로 재사용
                            last_finger_count = {
                                "point": point_finger_count, "fist": fist_finger_count,
                            }.get(shape_label)   # 그 외 라벨("none" 등)이면 None — 어느 것도 매치 안 됨
                        else:
                            last_finger_count = count_extended_fingers(hands[0], finger_extended_ratio)
                        # 랜드마크 0번(손목) — hand_move가 이동 추적에 쓰는 대표점
                        last_hand_point_ratio = _hand_point_ratio(
                            person_lock.locked_person.bbox, hands[0][0], frame_width_px,
                        )
                    else:
                        last_finger_count = None
                        last_hand_point_ratio = None
                    frames_since_hand_infer = 0
                else:
                    frames_since_hand_infer += 1
                finger_count = last_finger_count
                hand_point_ratio = last_hand_point_ratio

            gesture_event = gesture_filter.filter_signals(
                finger_count, hand_point_ratio, shoulder_width_ratio, shoulder_line_y_ratio,
            )
            state.debug = gesture_filter.debug

            if gesture_event is not None:
                event_sender.send(gesture_event)
                state.append_event(gesture_event)
                announcer.on_event(gesture_event)

            infer_fps_meter.update()
            state.capture_fps = camera.fps_meter.avg_fps
            state.infer_fps = infer_fps_meter.avg_fps

            annotated = draw_person_lock(input_tensor, person_lock, finger_count)
            annotated = draw_debug_panel(annotated, state.debug)
            overlay_event = state.last_event
            if overlay_event is not None and (
                time.monotonic() - overlay_event.ts_sec > EVENT_OVERLAY_HOLD_SEC
            ):
                overlay_event = None
            annotated = draw_status(annotated, state.infer_fps, overlay_event)
            state.update_frame(annotated)

            # FPS 상한 — 개발 PC에서 200+ FPS로 도는 낭비를 막는다.
            # 유휴(사람 없음·미잠금)일 땐 idle_infer_fps까지 더 낮춘다 (2026-07-20)
            min_loop_interval_sec = resolve_loop_interval_sec(config["model"], is_active)
            elapsed_sec = time.monotonic() - loop_start_sec
            if elapsed_sec < min_loop_interval_sec:
                time.sleep(min_loop_interval_sec - elapsed_sec)

    threading.Thread(target=_inference_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
