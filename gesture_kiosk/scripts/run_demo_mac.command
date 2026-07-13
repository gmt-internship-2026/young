#!/bin/zsh
# 맥 실행 스크립트 — Terminal.app에서 열려 카메라 권한(TCC)이 Terminal에 귀속된다.
# Claude/IDE 등 GUI 앱 하위 셸에서는 카메라 권한 팝업이 뜨지 않아 이 경로로 실행한다.
# 사용: 더블클릭 또는 `open scripts/run_demo_mac.command`
# 상세: docs/MAC_RUN_LOG.md

cd "$(dirname "$0")/.."

echo "[gesture_kiosk] 맥 데모 시작 — 카메라 권한 팝업이 뜨면 '허용'을 누르세요"
echo "[gesture_kiosk] 브라우저: http://localhost:5001"

# 권한 허용 전에는 카메라 열기에 실패하므로, 허용될 때까지 3초 간격 자동 재시도
for i in {1..40}; do
  ./venv/bin/python scripts/run_demo.py --config configs/config_mac.yaml
  exit_code=$?
  # 정상 종료(Ctrl+C 등)면 재시도하지 않는다
  if [[ $exit_code -eq 0 || $exit_code -eq 130 ]]; then
    break
  fi
  echo "[gesture_kiosk] 실행 실패(코드 $exit_code) — 카메라 권한 허용 후 자동 재시도 ($i/40)"
  sleep 3
done
