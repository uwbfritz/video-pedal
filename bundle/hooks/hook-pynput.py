"""Avoid the stock hook's display-sensitive collect_submodules('pynput')."""

from PyInstaller.compat import is_linux, is_win

if is_linux:
    hiddenimports = [
        "pynput.keyboard._xorg",
        "pynput.mouse._xorg",
        "pynput._util.xorg",
        "pynput._util.xorg_keysyms",
    ]
elif is_win:
    hiddenimports = [
        "pynput.keyboard._win32",
        "pynput.mouse._win32",
        "pynput._util.win32",
        "pynput._util.win32_vks",
    ]
else:
    raise RuntimeError("Build natively on Linux or Windows")
