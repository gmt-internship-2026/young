"""디버그 시각화 — 포즈·잠금 상태를 프레임 위에 그린다 (예시 UI 스트림에도 사용)."""
import cv2

EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 얼굴 박스


def draw_person_lock(frame, person_lock, finger_count=None):
    """잠긴 사용자의 얼굴 박스·손가락 인식 상태를 그린다.

    finger_count: 편 손가락 개수(엄지 제외) — 손 모양(point=1/fist=0) 판정 신호.
    없으면 표시 안 함.
    """
    if person_lock.locked_face_box is not None:
        x1, y1, x2, y2 = person_lock.locked_face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER LOCK", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
        if finger_count is not None:
            cv2.putText(
                frame, f"FINGERS {finger_count}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2,
            )
    return frame


def draw_debug_panel(frame, debug):
    """판정 계기판 — 좌하단에 내부값 표시 (실기 튜닝용, 2026-07-22, GMtech_project와 같은 패턴).

    SHAPE=현재 손 모양(point/fist/None) / POINT·FIST=각 모양의 이동 진행도(±1.0 도달
    시 확정, gesture_filter._SwipeTracker) — 2026-07-23 개편 이후 지금 잡히지 않는
    쪽(모양이 다른 트래커)은 항상 0으로 보인다(리셋된 상태).
    """
    if not debug:
        return frame
    h_px = frame.shape[0]
    swallow_bits = [
        f"{label}:{value}" for label, value in
        (("P", debug.get("swallow_point")), ("F", debug.get("swallow_fist")))
        if value
    ]
    swallow_tag = f"  RET[{' '.join(swallow_bits)}]" if swallow_bits else ""
    lines = [
        f"SHAPE {debug.get('hand_shape')}  FINGERS {debug.get('finger_count')}{swallow_tag}",
        f"POINT x{debug.get('point_x', 0):+.2f} y{debug.get('point_y', 0):+.2f}"
        f"  FIST x{debug.get('fist_x', 0):+.2f} y{debug.get('fist_y', 0):+.2f}",
    ]
    for line_idx, line in enumerate(lines):
        y_px = h_px - 14 - 22 * (len(lines) - 1 - line_idx)
        cv2.putText(frame, line, (10, y_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
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
