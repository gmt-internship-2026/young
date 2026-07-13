"""학습 프레임 수집 — 카메라에서 제스처 프레임을 모아 라벨링 준비를 한다.

사용법:
    python gesture/collect_frames.py --out ../gesture_kiosk/data/raw/session1
    (미리보기 창에서  [스페이스] 저장 / [a] 자동 저장 토글(0.5초 간격) / [q] 종료)

수집 지침(기획서 5.4):
- 인물 단위 분할을 위해 세션 폴더명에 촬영자 구분을 남긴다 (예: session1_personA)
- 신규 스펙 위주로: 주먹(fist)·손바닥(palm)·OK — 왼손/오른손, 거리·조명 다양하게
"""
import argparse
import os
import time

import cv2

AUTO_INTERVAL_SEC = 0.5


def main():
    parser = argparse.ArgumentParser(description="학습 프레임 수집기")
    parser.add_argument("--out", required=True, help="저장 폴더 (예: data/raw/session1_personA)")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"카메라(device={args.device})를 열 수 없습니다")

    saved_count = 0
    is_auto = False
    last_saved_sec = 0.0
    print("[INFO] [스페이스] 저장 / [a] 자동 저장 토글 / [q] 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        preview = frame.copy()
        cv2.putText(preview, f"saved {saved_count}  auto={'ON' if is_auto else 'off'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 120), 2)
        cv2.imshow("collect_frames", preview)

        key = cv2.waitKey(1) & 0xFF
        now_sec = time.monotonic()
        is_save = key == ord(" ") or (is_auto and now_sec - last_saved_sec >= AUTO_INTERVAL_SEC)
        if is_save:
            path = os.path.join(args.out, f"frame_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(path, frame)
            saved_count += 1
            last_saved_sec = now_sec
        elif key == ord("a"):
            is_auto = not is_auto
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] {saved_count}장 저장: {args.out}")


if __name__ == "__main__":
    main()
