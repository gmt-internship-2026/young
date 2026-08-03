"""gesture_kiosk 공식 진입점 — 실시간 엔진 구동 (2026-08-03 신설, 사용자 결정).

델파이(회사 UI)가 **이 파일을 직접 실행**한다 — bat 경유 없이:
    py main.py            # 엔진만 — 이벤트가 stdout에 한 줄씩
    py main.py --debug    # 카메라 창을 켠 채 시작(선택)

★카메라 창 런타임 토글(2026-08-03 사용자 결정 — 재실행 없는 점검): 엔진이
도는 중에 stdin에 한 줄 명령으로 카메라·계기판 창을 켜고 끈다:
    cam on   ↵    # 창 열기 (콘솔에서 직접 타이핑하거나, 델파이가 stdin 파이프로)
    cam off  ↵    # 창 닫기 (창에서 q/ESC도 동일)
    quit     ↵    # 엔진 종료 (Ctrl+C와 동일)
--debug 재실행이 필요 없어 현장 점검 중에도 연동을 끊지 않는다.

★환경 비종속(2026-08-03 가상환경 제거 — 사용자 결정): 시스템 파이썬에 직접
설치·실행한다(install.bat). 연구소 시선추적은 exe(파이썬 내장)라 환경 공유
자체가 불가능 — 별개 프로세스이며, 라이브러리 버전 정합(mediapipe 0.10.14·
numpy 1.26.4)은 만일의 통합 대비 보험으로 유지한다. 이 파일은 의존이 설치된
**아무 파이썬**으로 실행돼도 되고, 호출 위치(CWD)도 무관하다
(아래에서 엔진 폴더로 고정 — config의 상대 경로가 항상 성립).
구 scripts/run_demo.py·run.bat을 대체. 연동 상세: docs/델파이7_연동가이드.md.
"""
import argparse
import os
import sys
import threading
import time

PROCESS_START_SEC = time.monotonic()   # 시작 시간 계측 기점 — 임포트 비용 포함
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)   # 델파이가 어느 작업 디렉터리에서 띄우든 모델·로그 상대 경로 성립

from src.utils.config_loader import load_config
from src.utils.logger import get_logger, init_logging

DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")
DEBUG_WINDOW_NAME = "gesture_kiosk debug"


def disable_console_quick_edit():
    """콘솔 빠른 편집(QuickEdit) 해제 — 터치 먹통 방지 (2026-07-31 키오스크 실기).

    윈도우 콘솔은 클릭/터치로 "선택 모드"에 들어가면 프로세스 출력이 통째로
    정지한다 — stdout이 이벤트 채널인 이 엔진은 그대로 멈춘 것처럼 보였다
    (실기: 제목줄 "선택 관리자" 먹통). 터치스크린 키오스크에서는 콘솔을
    스치기만 해도 발생하므로 시작 시 꺼 둔다. 콘솔이 없으면(델파이 파이프
    실행) 조용히 통과 — 실패해도 엔진 기동을 막지 않는다.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        std_input_handle = kernel32.GetStdHandle(-10)   # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(std_input_handle, ctypes.byref(mode)):
            return   # 콘솔 없음 — 파이프/서비스 실행
        quick_edit_flag = 0x0040
        extended_flags = 0x0080   # 없이 끄면 일부 콘솔이 설정을 무시한다
        kernel32.SetConsoleMode(std_input_handle,
                                (mode.value & ~quick_edit_flag) | extended_flags)
    except Exception:   # noqa: 방어적 — 콘솔 설정 실패가 엔진 기동을 막으면 안 된다
        pass


def watch_stdin_commands(state, want_window):
    """stdin 명령 감시 스레드 — cam on/off·quit (모듈 독스트링의 런타임 토글).

    stdout은 이벤트 전용 채널이라 명령 입력은 stdin이 맡는다 — 콘솔 타이핑과
    델파이의 stdin 파이프 쓰기가 같은 경로다. stdin이 닫혀 있으면(파이프
    미연결 실행) 즉시 EOF — 스레드만 조용히 끝나고 엔진은 계속 돈다.
    """
    logger = get_logger("scripts")
    for line in sys.stdin:
        command = line.strip().lower()
        if command in ("cam on", "cam", "debug on"):
            want_window["value"] = True
            logger.info("stdin 명령: 카메라 창 켜기")
        elif command in ("cam off", "debug off"):
            want_window["value"] = False
            logger.info("stdin 명령: 카메라 창 끄기")
        elif command in ("quit", "exit"):
            logger.info("stdin 명령: 엔진 종료")
            state.is_running = False
            return
        elif command:
            logger.info("stdin 명령 무시(미지원): %r", command)


def run_window_loop(state, want_window):
    """메인 스레드 창 루프 — want_window 플래그에 따라 창을 열고 닫는다.

    cv2 창은 메인 스레드에서 돌려야 한다(macOS 제약 — 윈도우도 동일 정책).
    시청자 등록(add_viewer)으로 오버레이 렌더링이 켜진다 — 창이 없으면
    그리기 자체를 생략하는 기존 최적화 그대로.
    """
    import cv2   # 창을 한 번도 안 열어도 임포트 비용은 무시 가능 — cv2는 이미 로딩됨

    is_window_open = False
    try:
        while state.is_running:
            if want_window["value"] and not is_window_open:
                state.add_viewer()
                is_window_open = True
            elif not want_window["value"] and is_window_open:
                state.remove_viewer()
                cv2.destroyAllWindows()
                is_window_open = False
            if not is_window_open:
                time.sleep(0.2)
                continue
            frame = state.get_frame()
            if frame is not None:
                cv2.imshow(DEBUG_WINDOW_NAME, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):   # 27 = ESC — 창만 닫는다 (엔진은 계속)
                want_window["value"] = False
    except KeyboardInterrupt:
        pass
    finally:
        state.is_running = False
        if is_window_open:
            state.remove_viewer()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="gesture_kiosk 실시간 엔진")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--debug", action="store_true",
                        help="카메라·계기판 창을 켠 채 시작 (실행 중 cam on/off로도 토글)")
    args = parser.parse_args()

    disable_console_quick_edit()
    config = load_config(args.config)
    init_logging(config)
    logger = get_logger("scripts")
    logger.info("프로세스 기동 %.1f초 (인터프리터·경량 임포트)",
                time.monotonic() - PROCESS_START_SEC)

    from src.pipeline.realtime_loop import run_pipeline

    state = run_pipeline(config)
    logger.info("엔진 구동 중 — 이벤트 stdout 한 줄씩 · 카메라 창 cam on/off(+Enter) · 종료 Ctrl+C")
    # 콘솔 운영자 안내 — stdout(이벤트 채널)을 오염시키지 않도록 stderr로 한 줄만
    sys.stderr.write("[안내] 카메라 창: cam on / cam off (+Enter) · 종료: quit 또는 Ctrl+C\n")
    sys.stderr.flush()

    want_window = {"value": args.debug}
    if sys.stdin is not None:   # pythonw 등 stdin 자체가 없는 실행 — 토글 없이 구동
        threading.Thread(target=watch_stdin_commands, args=(state, want_window),
                         daemon=True).start()
    run_window_loop(state, want_window)


if __name__ == "__main__":
    main()
