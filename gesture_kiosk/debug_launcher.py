"""배포판(exe) 카메라 창 자동 실행 런처 (2026-08-03 신설, 사용자 요청).

bat 대신 exe로 열 수 있게 — gesture_kiosk_debug.bat과 같은 일(gesture_kiosk.exe
실행)을 하는 별도의 경량 exe. mediapipe·cv2를 다시 담지 않는다(subprocess로
같은 폭의 gesture_kiosk.exe를 호출만 함) — 그래서 onefile로 가볍게 빌드된다.
빌드는 build_exe.bat이 담당(별도 pyinstaller --onefile 호출, 산출물을
dist\\gesture_kiosk\\gesture_kiosk_debug.exe로 배치).
2026-08-11: main.py의 카메라 창 기본값이 켠 채 시작으로 바뀌어(--debug 인자
소멸, 아래 참고) 플래그 없이 그냥 실행 — gesture_kiosk.exe 단독 실행과
차이가 없어졌지만, 델파이 배포판에서 "카메라 확인용 런처"라는 별도 exe
이름 자체가 운용상 의미가 있어 유지한다.
"""
import os
import subprocess
import sys


def main():
    # sys.executable = 이 런처 exe 자신의 경로(onefile이어도 임시 해제 경로가 아니라
    # 실제 실행 위치) — 같은 폭의 gesture_kiosk.exe를 그 기준으로 찾는다
    launcher_dir = os.path.dirname(os.path.abspath(sys.executable))
    engine_path = os.path.join(launcher_dir, "gesture_kiosk.exe")
    if not os.path.exists(engine_path):
        print(f"[FAIL] gesture_kiosk.exe를 찾을 수 없습니다: {engine_path}")
        input("Enter를 눌러 종료...")
        sys.exit(1)
    subprocess.run([engine_path], cwd=launcher_dir)


if __name__ == "__main__":
    main()
