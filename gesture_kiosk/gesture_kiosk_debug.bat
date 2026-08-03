@echo off
chcp 65001 >nul
cd /d %~dp0

:: exe 배포판(dist\gesture_kiosk\)용 더블클릭 진입점 (2026-08-03 신설 — 사용자 요청:
:: cam on을 매번 타이핑하지 않고 카메라·계기판 창이 바로 뜨게). gesture_kiosk.exe
:: --debug와 동일 — 콘솔은 그대로 뜨고(이벤트 로그 확인용), 카메라 창이 같이 켜진다.
:: 종료: 카메라 창에서 q/ESC(창만 닫힘, 엔진 계속) 또는 콘솔에서 quit/Ctrl+C(엔진 종료)

gesture_kiosk.exe --debug
pause
