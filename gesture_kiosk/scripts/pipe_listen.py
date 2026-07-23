"""파이프 수신 확인 도구 — 델파이 없이 stdio 연동 규격을 검증한다 (2026-07-23).

엔진(run_demo.py)을 자식 프로세스로 실행하고, 델파이가 할 일(자식 프로세스의
stdout을 익명 파이프로 줄 단위 수신)을 그대로 흉내 내 받은 줄을 보여준다.
윈도우 배포 PC에서 델파이 연결 전에 이 도구로 규격을 먼저 확인한다.

사용법 (프로젝트 루트에서):
    python scripts/pipe_listen.py                        # 엔진 실행 + 수신 줄 출력
    python scripts/pipe_listen.py --config configs/config.yaml
종료: Ctrl+C (엔진도 함께 종료)
"""
import argparse
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def main():
    parser = argparse.ArgumentParser(description="stdio 파이프 수신 확인 (델파이 대역)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()

    cmd = [sys.executable, os.path.join(ROOT_DIR, "scripts", "run_demo.py"),
           "--config", args.config]
    print(f"[INFO] 엔진 실행: {' '.join(cmd)}")
    print("[INFO] stdout 파이프 수신 대기 — 카메라 앞에서 제스처를 해 보세요 (종료 Ctrl+C)")
    print("[INFO] 기대 규격: GESTURE|이벤트|손|신뢰도|시각  (이벤트 = left/right/top/bottom/back/home/ok)")

    # stderr는 물려받아 그대로 콘솔에 — 엔진 로그와 수신 줄이 한눈에 대조된다
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        for raw_line in process.stdout:
            line = raw_line.decode("ascii", errors="replace").rstrip("\r\n")
            print(f"[RECV] {line}")
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
