"""디버그 시각화 — 검출 결과와 상태를 프레임 위에 그린다 (예시 UI 스트림에도 사용)."""
import cv2

BBOX_COLOR = (0, 220, 120)
EVENT_COLOR = (0, 160, 255)
TEXT_COLOR = (255, 255, 255)
LOCK_COLOR = (255, 200, 0)       # 잠긴 사용자 얼굴 박스
WRIST_COLOR = {"left": (255, 120, 60), "right": (60, 120, 255)}
OCR_COLOR = (80, 200, 255)       # OCR 모드 안내 영역


def draw_bbox(frame, detections):
    """검출된 제스처 bbox와 이름·좌우(L/R)·conf를 그린다 — L/R은 사용자 기준."""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        side = getattr(det, "hand_side", None)
        side_tag = f"[{side[0].upper()}]" if side in ("left", "right") else ""
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BBOX_COLOR, 2)
        cv2.putText(
            frame,
            f"{det.class_name}{side_tag} {det.conf:.2f}",
            (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BBOX_COLOR,
            2,
        )
    return frame


def draw_person_lock(frame, person_lock):
    """잠긴 사용자의 얼굴 박스와 손목(사용자 기준 좌/우)을 그린다."""
    if person_lock.locked_face_box is not None:
        x1, y1, x2, y2 = person_lock.locked_face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), LOCK_COLOR, 2)
        cv2.putText(
            frame, "USER LOCK", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LOCK_COLOR, 2
        )
    for side, wrist in person_lock.user_wrists().items():
        if wrist is None:
            continue
        x_px, y_px = int(wrist[0]), int(wrist[1])
        cv2.circle(frame, (x_px, y_px), 10, WRIST_COLOR[side], 2)
        cv2.putText(
            frame, side[0].upper(), (x_px + 12, y_px + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WRIST_COLOR[side], 2,
        )
    return frame


def draw_ocr_mode(frame, guide_region_ratio):
    """OCR 모드 표시 — 주민등록증을 비출 안내 영역을 그린다."""
    h_px, w_px = frame.shape[:2]
    x1_r, y1_r, x2_r, y2_r = guide_region_ratio
    p1 = (int(w_px * x1_r), int(h_px * y1_r))
    p2 = (int(w_px * x2_r), int(h_px * y2_r))
    cv2.rectangle(frame, p1, p2, OCR_COLOR, 2)
    cv2.putText(
        frame, "ID CARD SCAN", (p1[0], p1[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, OCR_COLOR, 2,
    )
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
