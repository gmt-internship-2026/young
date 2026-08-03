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
디버그는 main.py --debug 로컬 창(cv2)이 담당한다.

2026-07-31 머리 앵커 추론 스레드 분리(키오스크 실기 — FPS 저하·획 끊김): 포즈
lite는 BlazeFace(수 ms)와 달리 호출당 수십 ms라 손 루프 인라인이면 10Hz마다
33ms 예산을 넘겨 프레임을 놓쳤다 — 빠른 쓸기(모션 블러 + 큰 이동폭)의 추적이
바로 그 회차에 끊긴다. 앵커는 느리게 변하는 값이라 비동기 반영으로 충분하다.

PipelineState가 디버그 창과 공유되는 유일한 상태 저장소다.
"""
import math
import threading
import time

from src.capture.camera_probe import select_camera
from src.capture.camera_stream import CameraStream, init_camera
from src.utils.env_report import log_environment
from src.inference.head_detector import HeadDetector
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
    startup_sec = time.monotonic()
    state = PipelineState()
    log_environment(config)   # 어느 하드웨어에서 돈 기록인지 로그 첫머리에 남긴다 (2026-07-16)
    # 시작 병렬화(2026-08-03 — 키오스크 시작 20초 단축): 카메라 오픈(MSMF —
    # 키오스크 실측 ~11초)과 모델 로딩(~수 초)이 서로 독립이라 겹친다. 프로브
    # (auto_select)가 켜져 있으면 오픈에 모델이 필요해 종전 순차 경로 유지
    probe_cfg = config["camera"].get("auto_select") or {}
    pre_open = None
    if not probe_cfg.get("enabled"):
        pre_open = {"cap": None, "error": None}

        def _open_main_camera():
            try:
                pre_open["cap"] = init_camera(config)
            except RuntimeError as error:
                pre_open["error"] = error   # 메인 스레드에서 다시 던진다

        pre_open["thread"] = threading.Thread(target=_open_main_camera, daemon=True)
        pre_open["thread"].start()
    # A안: 모델을 먼저 만들고 손 인식 품질로 카메라를 프로브해 메인을 고른다
    # (2026-07-29 포즈 제거 — 프로브 채점도 손 품질 단독)
    preprocessor = Preprocessor(config)
    hand_tracker = HandTracker(config)   # 주 추론 모델 (2026-07-29 포즈 제거)
    # 머리 앵커(2026-07-31 몸통판 — 얼굴 검출 교체, 사용자 결정): 포즈(BlazePose)가
    # 몸 실루엣으로 잡은 머리 위치의 사람 손만 인식 — 마스크·썬글라스·모자·색상
    # 무관, 옆 사람 손 난입 차단. config에 head_anchor 섹션이 없으면 비활성(종전)
    head_cfg = config.get("head_anchor") or {}
    head_detector = HeadDetector(config) if head_cfg else None
    models_elapsed_sec = time.monotonic() - startup_sec
    if pre_open is not None:
        pre_open["thread"].join()   # 모델 로딩과 겹쳐 돌던 카메라 오픈 대기
        if pre_open["error"] is not None:
            raise pre_open["error"]
        main_device_id, main_cap = config["camera"]["device_id"], pre_open["cap"]
    else:
        main_device_id, main_cap = select_camera(config, hand_tracker, preprocessor)
    camera = CameraStream(config, device_id=main_device_id, cap=main_cap).start()

    first_frame = camera.capture_frame()
    logger.info("시작 소요: 모델 %.1f초 · 카메라 포함 총 %.1f초 (%s)",
                models_elapsed_sec, time.monotonic() - startup_sec,
                "병렬 오픈" if pre_open is not None else "프로브 경로")
    frame_height_px, frame_width_px = first_frame.shape[:2]
    hand_selector = HandSelector(config, frame_width_px, frame_height_px)
    gesture_filter = GestureFilter(config)
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

            # 판정용 손 신호(손모양 + 손 중심 + 라벨) — 단일 손 추적(2026-07-31
            # 라벨 제거). x·y 모두 프레임 폭으로 나눈 등방 좌표 (거리 자 정규화와
            # 단위 일치, 2026-07-16)
            signal = hand_selector.user_hand_signal()
            signal_ratio = None if signal is None else (
                signal[0],
                (signal[1][0] / frame_width_px, signal[1][1] / frame_width_px),
                signal[2],
                signal[3],   # 검지 비율 — 탭 클릭 판정용 (2026-08-03, 스케일 무관)
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
