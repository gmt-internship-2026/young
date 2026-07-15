#!/bin/zsh
# 맥 팔등 데이터 수집 스크립트 — Terminal.app에서 열려 카메라 권한(TCC)이 Terminal에 귀속된다.
# Claude/IDE 등 GUI 앱 하위 셸에서는 카메라 권한 팝업이 뜨지 않아 이 경로로 실행한다.
# 사용: 더블클릭 또는 `open scripts/collect_arm_side_mac.command`
# 사람을 바꿔 수집할 때는 터미널에서 직접: python scripts/collect_arm_side.py --person p02

cd "$(dirname "$0")/.."

if [[ ! -f venv/bin/activate ]]; then
  echo "[gesture_kiosk] 셋업이 안 되어 있습니다 — 터미널에서 bash setup_mac.sh 를 먼저 실행하세요"
  read -k 1 -s "?아무 키나 누르면 닫힙니다..."
  exit 1
fi

echo "[gesture_kiosk] 팔등 데이터 수집 시작 — 카메라 권한 팝업이 뜨면 '허용'을 누르세요"
echo "[gesture_kiosk] 키: d=등쪽 저장 · f=안쪽 저장 · q=종료"

for i in {1..40}; do
  ./venv/bin/python scripts/collect_arm_side.py --config configs/config_mac.yaml
  exit_code=$?
  if [[ $exit_code -eq 0 || $exit_code -eq 130 ]]; then
    break
  fi
  echo "[gesture_kiosk] 실행 실패(코드 $exit_code) — 카메라 권한 허용 후 자동 재시도 ($i/40)"
  sleep 3
done
