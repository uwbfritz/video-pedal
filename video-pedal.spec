# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.compat import is_linux, is_win

root = Path(SPECPATH).resolve()
if is_linux:
    camera_modules = ['pyvirtualcam._native_linux_v4l2loopback']
elif is_win:
    camera_modules = [
        'pyvirtualcam._native_windows_obs',
        'pyvirtualcam._native_windows_unity_capture',
    ]
else:
    raise RuntimeError('Build natively on Linux or Windows')

# The standard cv2/NumPy hooks collect native libraries and OpenCV GUI plugins.
a = Analysis(
    [str(root / 'bundle' / 'entrypoint.py')],
    pathex=[str(root)],
    binaries=[],
    datas=[],
    hiddenimports=camera_modules,
    hookspath=[str(root / 'bundle' / 'hooks')],
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
    a.binaries,
    a.datas,
    [],
    name='video-pedal',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
