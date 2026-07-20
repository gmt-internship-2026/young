"""UDP 이벤트 수신 확인 도구 — 델파이 프로그램 없이 전송 규격을 눈으로 검증한다.

사용법 (엔진과 다른 터미널에서):
    python scripts/udp_listen.py            # config.yaml의 udp.port(9999) 수신
    python scripts/udp_listen.py --port 9999

엔진 쪽은 config event_output.mode: udp 로 켜고 실행한다.
받은 데이터그램을 그대로 출력한다 — 델파이7 텍스트 규격이면
`GESTURE|select|left|1.00|123.456` 한 줄이 보인다.
"""
import argparse
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config.yaml"
)


def main():
    parser = argparse.ArgumentParser(description="UDP 이벤트 수신 확인")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", type=int, default=None, help="미지정 시 config의 udp.port")
    args = parser.parse_args()

    port = args.port or load_config(args.config)["event_output"]["udp"]["port"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"[INFO] UDP {port} 수신 대기 — 엔진에서 제스처를 하면 여기 찍힌다 (Ctrl+C 종료)")
    try:
        while True:
            data, addr = sock.recvfrom(4096)
            print(f"{addr[0]}:{addr[1]} -> {data.decode('utf-8', errors='replace').rstrip()}")
    except KeyboardInterrupt:
        print("\n[INFO] 종료")


if __name__ == "__main__":
    main()
