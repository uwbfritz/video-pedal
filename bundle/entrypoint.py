"""Frozen launcher with a hardware-free packaging diagnostic."""

import sys

from loop_pedal import main


def check_packaged_dependencies():
    import importlib

    import cv2
    import numpy as np
    from pynput import keyboard, mouse
    from pyvirtualcam.camera import BACKENDS

    backend = "win32" if sys.platform == "win32" else "xorg"
    for module in (
        f"pynput.keyboard._{backend}",
        f"pynput.mouse._{backend}",
        f"pynput._util.{backend}",
    ):
        importlib.import_module(module)
    assert keyboard.Listener.__module__.endswith(f"_{backend}")
    assert mouse.Listener.__module__.endswith(f"_{backend}")
    expected = {"obs", "unitycapture"} if sys.platform == "win32" else {"v4l2loopback"}
    assert expected <= BACKENDS.keys(), BACKENDS
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok and cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape == frame.shape
    cv2.imshow("Packaging check", frame)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Packaged dependencies OK (native camera imports, hotkey backends, NumPy, OpenCV GUI)")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-packaged-dependencies"]:
        check_packaged_dependencies()
    else:
        sys.exit(main())
