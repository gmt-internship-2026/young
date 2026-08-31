"""gesture_kiosk 공식 진입점 — 실시간 엔진 구동 (2026-08-03 신설, 사용자 결정).

델파이(회사 UI)가 **이 파일을 직접 실행**한다 — bat 경유 없이:
    py main.py            # 창 없이 시작 — 카메라·계기판 창 없이 바로 제스처
                          #   인식만 돈다(2026-08-31 사용자 결정 — 기본값 재전환:
                          #   2026-08-11엔 현장 점검 편의로 "항상 켠 채 시작"이
                          #   기본이었으나, 실사용/납품 환경에서는 창이 매번 뜨는
                          #   쪽이 오히려 불필요해 다시 꺼진 채 시작으로 되돌림)
    py main.py --cam      # 카메라·계기판 창을 켠 채 시작 — 현장 점검·튜닝용

★카메라 창 런타임 토글(2026-08-03 사용자 결정 — 재실행 없는 점검): 엔진이
도는 중에 stdin에 한 줄 명령으로 카메라·계기판 창을 켜고 끈다:
    cam on   ↵    # 창 열기 (콘솔에서 직접 타이핑하거나, 델파이가 stdin 파이프로)
    cam off  ↵    # 창 닫기 (창에서 q/ESC도 동일)
    quit     ↵    # 엔진 종료 (Ctrl+C와 동일)
기본이 꺼진 채 시작이라 점검 중 필요할 때만 cam on(또는 시작 인자 --cam).

★카메라 전환(2026-08-04 신설 — 현장에서 장치 번호 즉석 확인): config 수정·
재실행 없이 다음 장치 번호로 순환 전환한다(configs/config.yaml의
camera.switch_candidate_count 범위 안). 두 경로 다 됨:
    카메라 창에서 `c` 키          # 물리 키보드
    cam next  ↵ (stdin, cam on처럼)  # 화상키보드 — 이미지창은 포커스가
                                      #   콘솔에 남으면 키를 못 받으므로 안전한 경로
어느 번호가 맞는지 화면 보며 바로 비교 가능. 확정되면 config의
camera.device_id를 그 번호로 남겨야 다음 실행에도 유지된다(전환은 런타임
한정, 저장 안 됨).

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
        elif command in ("cam next", "cam switch"):
            # 카메라 전환(2026-08-04)의 stdin 경로 — 이미지창의 'c' 키는 화상키보드로
            # 안 먹는 경우가 있다(포커스가 콘솔에 남아 창으로 키 이벤트가 안 감).
            # 콘솔 타이핑은 cam on/off와 같은 경로라 화상키보드에서도 확실히 동작
            state.cycle_camera()
            logger.info("stdin 명령: 카메라 전환")
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
            elif key == ord("c"):   # 카메라 전환(2026-08-04 — 현장 장치번호 즉석 비교)
                state.cycle_camera()
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
    parser.add_argument("--cam", action="store_true",
                        help="카메라·계기판 창을 켠 채 시작 (기본은 꺼진 채 시작 —"
                             " 2026-08-31 사용자 결정. 실행 중 cam on/off로도 토글 가능)")
    args = parser.parse_args()

    disable_console_quick_edit()
    config = load_config(args.config)
    init_logging(config)
    logger = get_logger("scripts")
    logger.info("프로세스 기동 %.1f초 (인터프리터·경량 임포트)",
                time.monotonic() - PROCESS_START_SEC)

    from src.pipeline.realtime_loop import run_pipeline

    state = run_pipeline(config)
    window_state_text = "켠 채 시작(--cam)" if args.cam else "꺼진 채 시작"
    logger.info("엔진 구동 중 — 이벤트 stdout 한 줄씩 · 카메라 창 %s, cam off/on(+Enter)으로"
               " 토글 · 카메라 전환 c키 또는 cam next(+Enter) · 종료 Ctrl+C",
               window_state_text)
    # 콘솔 운영자 안내 — stdout(이벤트 채널)을 오염시키지 않도록 stderr로 한 줄만
    sys.stderr.write(f"[안내] 카메라 창: {window_state_text} (끄려면 cam off, 다시 켜려면"
                     " cam on, +Enter) · 카메라 전환: c키 또는 cam next(+Enter) ·"
                     " 종료: quit 또는 Ctrl+C\n")
    sys.stderr.flush()

    want_window = {"value": args.cam}
    if sys.stdin is not None:   # pythonw 등 stdin 자체가 없는 실행 — 토글 없이 구동
        threading.Thread(target=watch_stdin_commands, args=(state, want_window),
                         daemon=True).start()
    run_window_loop(state, want_window)


def _run_interactive_safe():
    """탐색기에서 main.py를 직접 더블클릭해 실행하는 경우(2026-08-11 사용자
    요청 — "무조건 main.py로 열어야 한다", 우회 bat 불가)를 위한 안전판.

    콘솔이 파이프 없이 진짜 인터랙티브(예: py.exe 연결로 더블클릭 실행)면
    sys.stdin.isatty()가 True다 — 이때만 예외 발생 시 트레이스백을 찍고
    아무 키 입력을 기다린다: 안 그러면 카메라 연결 실패 등으로 시작 직후
    죽었을 때 콘솔 창이 원인도 못 읽고 바로 닫혀버린다(더블클릭 실행의
    전형적 증상). quit/Ctrl+C로 끝내는 정상 종료 경로는 안 건드린다 —
    main()이 그냥 정상 반환하면 이 함수도 그대로 반환해 창이 즉시 닫힌다.
    델파이가 자식 프로세스로 띄우는 배포 경로는 stdin이 파이프가 아니거나
    비대화식이라 isatty()가 False — 크래시해도 대기 없이 그대로 종료해
    Delphi 쪽 재기동 타이머(WM_ENGINE_EXITED)가 정상 동작한다.
    """
    is_interactive = sys.stdin is not None and sys.stdin.isatty()
    try:
        main()
    except Exception:
        if not is_interactive:
            raise
        import traceback
        traceback.print_exc()
        input("\n[오류로 종료됨] 창을 닫으려면 아무 키나 누르세요...")
        sys.exit(1)


if __name__ == "__main__":
    _run_interactive_safe()
