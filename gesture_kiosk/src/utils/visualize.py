"""디버그 시각화 — 포즈·잠금 상태를 프레임 위에 그린다 (디버그 창에 사용)."""
import cv2

EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 머리 박스
TRACKED_COLOR = (60, 220, 60)    # 추적 손(단일 손 — 2026-07-31 라벨 제거)
CANDIDATE_COLOR = (150, 150, 150)  # 게이트 통과 후보 손 — 획득 전 상황 확인용
SHAPE_TAG = {"fist": "(F)", "finger": "(1)", "open": "(5)"}   # F=주먹, 1=한 손가락,
                                               # 5=손바닥(2026-07-31 temp 계층). 불명은 무표시


def draw_user_hands(frame, hand_selector):
    """추적 손 박스·추적점과 후보 손 점을 그린다.

    2026-07-31 단일 손 추적(라벨 제거): 좌/우 점 2개 대신 **추적 손 1점**(초록
    원 + 모양 태그)과 게이트 통과 후보(회색 점)를 표시 — 어떤 손이 잡혔고
    누가 후보로 대기 중인지 실기에서 바로 확인. 앵커 머리 상자("USER HEAD" —
    몸통판: 포즈 기반, 마스크·모자 무관)는 유지.
    """
    if getattr(hand_selector, "anchor_head_box", None) is not None:
        x1, y1, x2, y2 = hand_selector.anchor_head_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER HEAD", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    if hand_selector.locked_box is not None:
        x1, y1, x2, y2 = hand_selector.locked_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER HAND", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    for point in hand_selector.candidate_points():
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, CANDIDATE_COLOR, 1)
    signal = hand_selector.user_hand_signal()
    if signal is not None:
        # 2026-08-03 탭 클릭 신설로 검지비율이 4번째 값으로 추가됨(hand_select.py) —
        # 여기선 안 쓰므로 버린다
        shape, point, _label, _index_ratio = signal
        x_px, y_px = int(point[0]), int(point[1])
        cv2.circle(frame, (x_px, y_px), 10, TRACKED_COLOR, 2)
        cv2.putText(
            frame, "H" + SHAPE_TAG.get(shape, ""), (x_px + 12, y_px + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACKED_COLOR, 2,
        )
    return frame



def draw_debug_panel(frame, debug):
    """판정 계기판 — 좌하단에 내부값 표시 (실기 튜닝용, 2026-07-16).

    SCALE=어깨 스케일 / ARM=활성 팔+손 모양(원시 판별) / RET=복귀 삼킴 예약 방향 /
    SWIPE=진행도(±1.0 판정) / LATCH=고정 모양(F=주먹, 1=한 손가락, -=없음)과
    전환 후보(cand 모양:연속 수) — 판정은 래치만 본다 (2026-07-28 v3).
    """
    if not debug:
        return frame
    h_px = frame.shape[0]
    swallow = debug.get("swallow")
    swallow_tag = f" [RET:{swallow}]" if swallow else ""
    side = debug.get("active_side") or "-"
    shape_tag = SHAPE_TAG.get(debug.get("hand_shape"), "")
    latch_label = {"fist": "F", "finger": "1", "open": "5"}.get(
        debug.get("latched_shape"), "-")
    candidate = debug.get("latch_candidate")
    latch_tag = latch_label + (f" cand:{candidate}" if candidate else "")
    lines = [
        f"SCALE {debug.get('body_scale', 0):.2f}  ARM {side}{shape_tag}{swallow_tag}",
        f"SWIPE x{debug.get('swipe_progress_x', 0):+.2f} y{debug.get('swipe_progress_y', 0):+.2f}"
        f"  LATCH {latch_tag}",
    ]
    for line_idx, line in enumerate(lines):
        y_px = h_px - 14 - 24 * (len(lines) - 1 - line_idx)
        cv2.putText(frame, line, (10, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    return frame


def draw_status(frame, avg_fps, gesture_event=None):
    """FPS와 최근 확정 이벤트를 좌상단에 표시한다."""
    cv2.putText(
        frame, f"FPS {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2
    )
    if gesture_event is not None:
        cv2.putText(
            frame,
            f"EVENT {gesture_event.class_name}",
            (10, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            EVENT_COLOR,
            2,
        )
    return frame
