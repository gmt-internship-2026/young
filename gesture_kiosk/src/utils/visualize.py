"""디버그 시각화 — 포즈·잠금 상태를 프레임 위에 그린다 (예시 UI 스트림에도 사용)."""
import cv2

EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 얼굴 박스
WRIST_COLOR = {"left": (255, 120, 60), "right": (60, 120, 255)}


def draw_person_lock(frame, person_lock):
    """잠긴 사용자의 얼굴 박스와 쓸기 추적점(사용자 기준 좌/우)을 그린다.

    라벨: L/R + 팔꿈치 폴백 중이면 "(E)" — 손목 미검출 상태를 화면에서 확인할 수 있게.
    """
    if person_lock.locked_face_box is not None:
        x1, y1, x2, y2 = person_lock.locked_face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER LOCK", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    for side, point_info in person_lock.user_swipe_points().items():
        if point_info is None:
            continue
        source, point = point_info
        x_px, y_px = int(point[0]), int(point[1])
        label = side[0].upper() + ("(E)" if source == "elbow" else "")
        cv2.circle(frame, (x_px, y_px), 10, WRIST_COLOR[side], 2)
        cv2.putText(
            frame, label, (x_px + 12, y_px + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WRIST_COLOR[side], 2,
        )
    return frame



def draw_debug_panel(frame, debug):
    """판정 계기판 — 좌하단에 내부값 표시 (실기 튜닝용, 2026-07-16).

    SCALE=어깨 스케일 / SWIPE=진행도(±1.0 판정)+장전 상태 / PEND=수직 1회 보류.
    """
    if not debug:
        return frame
    h_px = frame.shape[0]
    armed_tag = "" if debug.get("is_armed", True) else " [REARM]"
    side = debug.get("active_side") or "-"
    source_tag = "(E)" if debug.get("active_source") == "elbow" else ""
    lines = [
        f"SCALE {debug.get('body_scale', 0):.2f}  ARM {side}{source_tag}{armed_tag}",
        f"SWIPE x{debug.get('swipe_progress_x', 0):+.2f} y{debug.get('swipe_progress_y', 0):+.2f}",
    ]
    if debug.get("pending"):
        lines.append(f"PEND {debug['pending']} (1/2)")
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
