"""pipeline 모듈 — 캡처·추론·판정·전송을 연결해 실시간 루프를 구동한다 (기획서 2.2, 3.2).

프레임 흐름 (2026-07-29 포즈 제거 — 손 단독 추론, hand_select.py 참고):
  카메라(스레드) → 거울 반전 → 손 랜드마크(MediaPipe — 유일한 추론 모델)
  → 사용자 손 선별(hand_select: 크기+연속성, 손 실측 자)
  → 동작 판정(config gestures.engine으로 택1, 2026-08-06):
      swipe(종전) = gesture_filter: 손 모양 래치 + 첫 선 궤적 4방향
      pose_classifier(feat/shape_ml 신설) = pose_gesture_filter: 정지 자세
      (모양+방향 콤보) 하나를 학습 분류기가 통째로 판정 — 궤적 추적 없음
  → 이벤트 전송(stdio — stdout 한 줄, 델파이가 파이프로 수신)

2026-07-29 포즈(ONNX Runtime + rtmlib) 제거(사용자 결정): 제스처 판정은 손
데이터만으로 끝난다 — 포즈가 하던 잠금·자(尺)·게이트는 hand_select가 손 기반으로
대체. 추론 엔진 = MediaPipe 내장 TFLite(XNNPACK) 하나 (Apache-2.0 — 라이선스
B안 유지: 상업 허용·카피레프트 없음).

2026-07-23: 웹소켓·UDP·데모 웹 서버 제거(회사 결정 — 네트워크 철회, print 연동).
디버그는 main.py의 로컬 창(cv2, 2026-08-11부터 기본으로 켠 채 시작)이 담당한다.

2026-07-31 머리 앵커 추론 스레드 분리(키오스크 실기 — FPS 저하·획 끊김): 포즈
lite는 BlazeFace(수 ms)와 달리 호출당 수십 ms라 손 루프 인라인이면 10Hz마다
33ms 예산을 넘겨 프레임을 놓쳤다 — 빠른 쓸기(모션 블러 + 큰 이동폭)의 추적이
바로 그 회차에 끊긴다. 앵커는 느리게 변하는 값이라 비동기 반영으로 충분하다.

PipelineState가 디버그 창과 공유되는 유일한 상태 저장소다.
"""
import math
import threading
import time

from src.capture.camera_stream import CameraStream, init_camera
from src.utils.env_report import log_environment
from src.inference.head_detector import HeadDetector
from src.inference.hand_tracker import HandTracker
from src.inference.preprocessor import Preprocessor
from src.pipeline.event_sender import create_event_sender
from src.postprocess.gesture_filter import GestureFilter
from src.postprocess.hand_select import HandSelector
from src.postprocess.hand_shape import hand_center_point
from src.postprocess.hand_shape_classifier import HandShapeClassifier
from src.postprocess.pose_gesture_filter import PoseGestureFilter, classify_pose_combo
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


def resolve_roi_box(prev_box, anchor_box, frame_width_px, frame_height_px, roi_cfg,
                    reach_widths):
    """머리 앵커 기반 손 추론 크롭 창 -> (x1, y1, x2, y2) | None(전체 프레임).

    원거리 디지털 줌(2026-07-31 키오스크 실기 — 거리별 인식 편차): MediaPipe
    손바닥 검출기는 프레임 전체를 고정 크기(~192px)로 줄여 보므로, 먼 사용자의
    손은 캡처 해상도와 무관하게 몇 픽셀로 뭉개져 검출이 끊기고 모양(주먹/한
    손가락) 판별이 무너진다. 앵커 주변 팔 도달 반경만 잘라 넣으면 손이 모델
    입력에서 그 비율만큼 커진다 — 크롭은 원본 픽셀 그대로(리사이즈 없음)라
    좌표는 크롭 원점만 더하면 프레임 좌표와 동일하다.

    - 앵커 없음(상체 미노출)·근거리(크롭이 프레임 짧은 변을 덮음) → None.
    - 히스테리시스: 크롭 창을 매 프레임 옮기면 VIDEO 모드의 추적 ROI가 어긋나
      재검출이 반복된다 — 중심·크기가 문턱 이상 변할 때만 창을 갱신한다.
    순수 함수 — tests/test_infer_optimization.py.
    """
    if anchor_box is None or reach_widths is None:
        return None
    head_width_px = float(anchor_box[2] - anchor_box[0])
    if head_width_px <= 0.0:
        return None
    half_px = max(reach_widths * head_width_px * roi_cfg.get("pad_reach_ratio", 1.3),
                  roi_cfg.get("min_side_px", 320) / 2.0)
    if 2.0 * half_px >= min(frame_width_px, frame_height_px):
        return None   # 근거리 — 크롭이 프레임을 사실상 다 덮는다: 전체 프레임 사용
    anchor_center_x = (anchor_box[0] + anchor_box[2]) / 2.0
    anchor_center_y = (anchor_box[1] + anchor_box[3]) / 2.0
    side_px = int(2.0 * half_px)
    x1 = int(max(0.0, min(anchor_center_x - half_px, frame_width_px - side_px)))
    y1 = int(max(0.0, min(anchor_center_y - half_px, frame_height_px - side_px)))
    target = (x1, y1, x1 + side_px, y1 + side_px)
    if prev_box is not None:
        prev_center = ((prev_box[0] + prev_box[2]) / 2.0,
                       (prev_box[1] + prev_box[3]) / 2.0)
        prev_side_px = float(max(prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]))
        target_center = ((target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0)
        if (math.dist(prev_center, target_center)
                <= roi_cfg.get("move_ratio", 0.15) * prev_side_px
                and abs(side_px - prev_side_px)
                <= roi_cfg.get("resize_ratio", 0.2) * prev_side_px):
            return prev_box   # 문턱 미달 — 창 유지 (추적 ROI 안정)
    return target


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
        self.camera_device_id = None   # 현재 열린 카메라 장치 번호 — 전환 확인용(계기판·로그)
        self._cycle_camera_fn = None   # run_pipeline이 심는 콜백 — 카메라 전환 실행부

    def add_viewer(self):
        """디버그 창 열림 — 다음 루프부터 오버레이를 그린다 (2026-07-20 최적화)."""
        with self._lock:
            self._viewer_count += 1

    def remove_viewer(self):
        with self._lock:
            self._viewer_count = max(0, self._viewer_count - 1)

    def cycle_camera(self):
        """디버그 창 'c' 키(2026-08-04 신설) — 다음 후보 장치로 카메라를 전환한다.

        키오스크 현장에서 장치 번호(0/1 중 뭐가 맞는지)를 config 수정·재실행 없이
        즉석으로 비교하려는 요청 — run_pipeline이 심어둔 콜백을 호출만 한다
        (실제 스트림 교체는 그쪽 클로저가 담당, 아래 _cycle_camera 독스트링).
        """
        if self._cycle_camera_fn is not None:
            self._cycle_camera_fn()

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
    startup_sec = time.monotonic()
    state = PipelineState()
    log_environment(config)   # 어느 하드웨어에서 돈 기록인지 로그 첫머리에 남긴다 (2026-07-16)
    # 시작 병렬화(2026-08-03 — 키오스크 시작 20초 단축): 카메라 오픈(MSMF —
    # 키오스크 실측 ~11초)과 모델 로딩(~수 초)이 서로 독립이라 겹친다.
    # (2026-08-04 사용자 결정 — 카메라 자동 선별(auto_select/camera_probe) 제거:
    # 현장에서 프로브가 의도와 다른 USB 카메라를 고르는 사례가 있어 신뢰하지
    # 않기로 함. config의 device_id 고정 하나만 연다 — 프로브 갈래 자체가 소멸)
    pre_open = {"cap": None, "error": None}

    def _open_main_camera():
        try:
            pre_open["cap"] = init_camera(config)
        except RuntimeError as error:
            pre_open["error"] = error   # 메인 스레드에서 다시 던진다

    pre_open["thread"] = threading.Thread(target=_open_main_camera, daemon=True)
    pre_open["thread"].start()
    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)   # 주 추론 모델 (2026-07-29 포즈 제거)
    # 머리 앵커(2026-07-31 몸통판 — 얼굴 검출 교체, 사용자 결정): 포즈(BlazePose)가
    # 몸 실루엣으로 잡은 머리 위치의 사람 손만 인식 — 마스크·썬글라스·모자·색상
    # 무관, 옆 사람 손 난입 차단. config에 head_anchor 섹션이 없으면 비활성(종전)
    head_cfg = config.get("head_anchor") or {}
    head_detector = HeadDetector(config) if head_cfg else None
    models_elapsed_sec = time.monotonic() - startup_sec
    pre_open["thread"].join()   # 모델 로딩과 겹쳐 돌던 카메라 오픈 대기
    if pre_open["error"] is not None:
        raise pre_open["error"]
    main_device_id, main_cap = config["camera"]["device_id"], pre_open["cap"]
    camera = CameraStream(config, device_id=main_device_id, cap=main_cap).start()
    state.camera_device_id = main_device_id

    def _cycle_camera():
        """'c' 키 콜백 — camera를 다음 후보 장치로 교체(nonlocal 재바인딩).

        _inference_loop·_head_anchor_loop는 매 순회 이 스코프의 camera 변수를
        다시 읽으므로(파이썬 클로저는 셀을 공유 — 스냅샷이 아니다), 여기서
        재바인딩만 하면 두 스레드 다음 순회부터 새 스트림을 쓴다. 옛 스트림은
        교체 직후 정지 — 그 찰나에 옛 참조로 진행 중이던 호출은 stop() 후에도
        마지막 캐시 프레임을 그대로 돌려줄 뿐이라(capture_frame 독스트링) 죽지 않는다.
        """
        nonlocal camera
        candidate_count = max(2, config["camera"].get("switch_candidate_count", 2))
        next_device_id = (camera.device_id + 1) % candidate_count
        try:
            new_camera = CameraStream(config, device_id=next_device_id).start()
        except RuntimeError as error:
            logger.warning("카메라 전환 실패 (device_id=%d): %s", next_device_id, error)
            return
        old_camera = camera
        camera = new_camera
        old_camera.stop()
        state.camera_device_id = next_device_id
        logger.info("카메라 전환: device_id=%d", next_device_id)

    state._cycle_camera_fn = _cycle_camera

    first_frame = camera.capture_frame()
    logger.info("시작 소요: 모델 %.1f초 · 카메라 포함 총 %.1f초 (병렬 오픈)",
                models_elapsed_sec, time.monotonic() - startup_sec)
    frame_height_px, frame_width_px = first_frame.shape[:2]
    hand_selector = HandSelector(config, frame_width_px, frame_height_px)
    # 판정 엔진 택1(2026-08-06, config gestures.engine — 모듈 독스트링): swipe는
    # 기존 GestureFilter, pose_classifier는 정지 자세 학습 분류기(feat/shape_ml).
    # 두 엔진 다 만들지 않는다 — pose_classifier는 가중치 로딩(파일 I/O)이 있어
    # 안 쓰는 쪽을 굳이 초기화할 이유가 없다
    gesture_mode = config["gestures"].get("engine", "swipe")
    gesture_filter = None
    pose_classifier = None
    pose_gesture_filter = None
    pose_min_conf = None
    pose_max_dist_ratio = None
    pose_none_margin = None
    pose_none_neighbor_ratio = None
    pose_geometric_check = None
    pose_rest_zone_below_shoulder = None
    if gesture_mode == "pose_classifier":
        pose_cfg = config["gestures"]["pose_classifier"]
        pose_classifier = HandShapeClassifier(pose_cfg["weights_path"])
        pose_gesture_filter = PoseGestureFilter(config)
        pose_min_conf = pose_cfg.get("min_conf")   # None(키 없음)이면 종전(항상 argmax 강제)
        pose_max_dist_ratio = pose_cfg.get("max_dist_ratio")   # None이면 거리 검사 건너뜀
        pose_none_margin = pose_cfg.get("none_margin")   # None이면 none 마진 검사 건너뜀
        pose_none_neighbor_ratio = pose_cfg.get("none_neighbor_ratio")   # None이면 건너뜀
        # 손가락 개수 기하 교차검증(2026-08-07 — pose_gesture_filter.classify_pose_combo
        # 독스트링 참고): hand_select와 같은 hand_shape 임계값을 재사용 — 학습
        # 분류기와 별개로 손가락 개수가 안 맞으면(V사인 등) 데이터 없이도 거른다
        if pose_cfg.get("geometric_shape_check"):
            hand_shape_cfg = config["hand_select"]["hand_shape"]
            pose_geometric_check = (
                hand_shape_cfg["extend_ratio"], hand_shape_cfg["min_valid_fingers"],
                hand_shape_cfg.get("curl_confirm_ratio", hand_shape_cfg["extend_ratio"]),
            )
        # 휴식 존 게이트(2026-08-11 — 위 config 키 독스트링 참고): pose_classifier는
        # 손 위치를 안 봐서 허리께 휴식 자세가 손 모양만으로 오발되던 것 방어
        pose_rest_zone_below_shoulder = pose_cfg.get("rest_zone_below_shoulder")
        logger.info("판정 엔진: pose_classifier (classes=%s, min_conf=%s, "
                    "max_dist_ratio=%s, none_margin=%s, none_neighbor_ratio=%s, "
                    "geometric_shape_check=%s)",
                    pose_classifier.classes, pose_min_conf, pose_max_dist_ratio,
                    pose_none_margin, pose_none_neighbor_ratio, pose_geometric_check)
    else:
        gesture_filter = GestureFilter(config)
        logger.info("판정 엔진: swipe")
    event_sender = create_event_sender(config)

    state.is_running = True

    # 머리 앵커 스레드 ↔ 손 루프 공유 관측함 — 최신 1건, 소비 즉시 비움
    head_result_lock = threading.Lock()
    head_result = [None]

    # 원거리 디지털 줌 설정(2026-07-31) — 키(roi_zoom) 삭제 시 종전(전체 프레임)
    roi_cfg = config["hand_tracker"].get("roi_zoom")

    def _inference_loop():
        infer_fps_meter = FpsMeter()
        was_active = True   # 유휴↔활성 전환을 로그로 남기기 위한 직전 상태
        last_frame_seq = 0  # 새 프레임 동기화(2026-07-20) — 같은 프레임 중복 추론 방지
        roi_box = None      # 손 추론 크롭 창 — 히스테리시스 상태 (resolve_roi_box)
        while state.is_running:
            loop_start_sec = time.monotonic()

            frame, last_frame_seq = camera.capture_new_frame(last_frame_seq)
            input_tensor = preprocessor.preprocess_frame(frame)

            # 원거리 디지털 줌 — 앵커(머리) 주변만 잘라 손 추론: 먼 손이 모델
            # 입력에서 커진다 (resolve_roi_box 독스트링). 크롭은 원본 픽셀
            # 그대로라 크롭 원점을 더하면 프레임 좌표와 동일 (z는 px 궤적·판정에
            # 미사용 — hand_shape.hand_center_point 주석)
            if roi_cfg is not None:
                roi_box = resolve_roi_box(
                    roi_box, hand_selector.anchor_head_box,
                    frame_width_px, frame_height_px, roi_cfg,
                    head_cfg.get("reach_head_widths"),
                )
            if roi_box is not None:
                rx1, ry1, rx2, ry2 = roi_box
                hands = hand_tracker.infer(input_tensor[ry1:ry2, rx1:rx2])
                for hand in hands:
                    hand.landmarks[:, 0] += rx1
                    hand.landmarks[:, 1] += ry1
            else:
                hands = hand_tracker.infer(input_tensor)
            heads = None   # None = 새 머리 관측 없음 — 앵커 유지(hand_select)
            if head_detector is not None:
                with head_result_lock:
                    heads = head_result[0]   # 앵커 스레드의 최신 관측 — 1회만 반영
                    head_result[0] = None
            is_engaged = hand_selector.update(hands, heads)
            state.is_user_locked = is_engaged

            # 유휴 판정 — 손이 보이거나 최근 사용 중이면 활성 (idle_infer_fps 절감)
            is_active = bool(hands) or is_engaged
            if is_active != was_active:
                logger.info("추론 %s 전환 (hands=%d)", "활성" if is_active else "유휴",
                            len(hands))
                was_active = is_active

            if gesture_mode == "pose_classifier":
                # 정지 자세 판정(2026-08-06 — 모듈 독스트링): 궤적·어깨 자 불필요,
                # 추적 손의 원본 랜드마크(hand_select.HandSelector.tracked_hand)를
                # 분류기에 직접 넣는다 — 왼손이면 classify_pose_combo가 오른손
                # 기준으로 미러링·역미러링까지 알아서 처리한다(모듈 독스트링)
                tracked_hand = hand_selector.tracked_hand
                combo_label = (classify_pose_combo(pose_classifier, tracked_hand,
                                                   min_conf=pose_min_conf,
                                                   max_dist_ratio=pose_max_dist_ratio,
                                                   none_margin=pose_none_margin,
                                                   none_neighbor_ratio=pose_none_neighbor_ratio,
                                                   geometric_shape_check=pose_geometric_check)
                              if tracked_hand is not None else None)
                if (combo_label is not None and pose_rest_zone_below_shoulder is not None
                        and tracked_hand is not None):
                    # 휴식 존 게이트(2026-08-11 — config gestures.pose_classifier.
                    # rest_zone_below_shoulder 독스트링 참고): 앵커·손 실측 자
                    # 둘 다 있어야 판단 — 하나라도 없으면 관대하게 통과(게이트 생략)
                    shoulder_y_ratio = hand_selector.shoulder_line_y_ratio()
                    body_scale_ratio = hand_selector.hand_scale_ratio()
                    hand_center = hand_center_point(tracked_hand.landmarks)
                    if (shoulder_y_ratio is not None and body_scale_ratio is not None
                            and hand_center is not None):
                        hand_y_ratio = hand_center[1] / frame_width_px
                        rest_zone_top_y = (shoulder_y_ratio
                                          + pose_rest_zone_below_shoulder * body_scale_ratio)
                        if hand_y_ratio > rest_zone_top_y:
                            combo_label = None   # 허리 옆 등 휴식 존 — 모양·방향 무관 무시
                hand_side = tracked_hand.user_side if tracked_hand is not None else None
                gesture_event = pose_gesture_filter.filter_signals(combo_label, hand_side)
                state.debug = pose_gesture_filter.debug
            else:
                # 판정용 손 신호(손모양 + 손 중심 + 라벨) — 단일 손 추적(2026-07-31
                # 라벨 제거). x·y 모두 프레임 폭으로 나눈 등방 좌표 (거리 자 정규화와
                # 단위 일치, 2026-07-16)
                signal = hand_selector.user_hand_signal()
                signal_ratio = None if signal is None else (
                    signal[0],
                    (signal[1][0] / frame_width_px, signal[1][1] / frame_width_px),
                    signal[2],
                    signal[3],   # 검지 비율 — 탭 클릭 판정용 (2026-08-03, 스케일 무관)
                    signal[4],   # 기하 손모양 — 탭 클릭 모드 확인 전용, 학습 분류기
                                 #   무관(2026-08-03, hand_select._classify_shape 참고)
                )
                gesture_event = gesture_filter.filter_signals(
                    signal_ratio, hand_selector.hand_scale_ratio(),   # 손 실측 자
                    hand_selector.shoulder_line_y_ratio(),   # 앵커 어깨선 (없으면 하단 띠 폴백)
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

    # 머리 앵커 추론 스레드(2026-07-31 키오스크 실기 — 모듈 독스트링): 손 루프와
    # 분리해 포즈 비용(호출당 수십 ms)이 손 추적 프레임을 밀어내지 않게 한다.
    # 관측은 최신 1건만 유지, 손 루프가 소비하는 즉시 비운다 — hand_select의
    # "None = 관측 없음(앵커 유지)" 규약이 그대로 성립한다
    def _head_anchor_loop():
        interval_sec = 1.0 / head_cfg.get("infer_fps", 10)
        while state.is_running:
            infer_start_sec = time.monotonic()
            frame = preprocessor.preprocess_frame(camera.capture_frame())
            heads = head_detector.infer(frame)
            with head_result_lock:
                head_result[0] = heads
            elapsed_sec = time.monotonic() - infer_start_sec
            if elapsed_sec < interval_sec:
                time.sleep(interval_sec - elapsed_sec)

    threading.Thread(target=_inference_loop, daemon=True).start()
    if head_detector is not None:
        threading.Thread(target=_head_anchor_loop, daemon=True).start()
    logger.info("실시간 파이프라인 시작 (frame_width_px=%d)", frame_width_px)
    return state
