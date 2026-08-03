# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 빌드 레시피 (2026-08-03 신설) — 재빌드: build_exe.bat 또는
#   venv_win\Scripts\python.exe -m PyInstaller --noconfirm gesture_kiosk.spec
# onedir(단일 폴더)을 쓴다 — mediapipe·opencv가 커서(수백MB) onefile 자체압축은
# 시작마다 임시폴더 해제 비용만 늘고 이점이 없다.
# collect_all(mediapipe/cv2): 두 패키지 다 공식 pyinstaller hook이 없어(2026-08-03
# 확인 — hooks-contrib 미포함) 서브모듈·데이터 누락을 안전하게 전부 담는다.
from PyInstaller.utils.hooks import collect_all

datas = [('configs/config.yaml', 'configs'), ('models/weights/hand_landmarker.task', 'models/weights'), ('models/weights/pose_landmarker_lite.task', 'models/weights')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('mediapipe')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='gesture_kiosk',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='gesture_kiosk',
)
