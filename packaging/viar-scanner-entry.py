"""PyInstaller entry point for the standalone VIAR Scanner package."""

import sys
import time

import cv2
import customtkinter
import numpy
import PIL
import PIL._tkinter_finder
import tkinter

from camscan import __version__
from camscan.app import CamScanApp
from camscan.camera import Camera


def self_test():
    """Verify that every bundled runtime component can be imported."""
    assert hasattr(cv2, "ximgproc")
    assert hasattr(cv2.ximgproc, "niBlackThreshold")
    assert cv2.ximgproc.BINARIZATION_SAUVOLA == 1
    print(
        "VIAR Scanner",
        __version__,
        "self-test OK;",
        "OpenCV",
        cv2.__version__,
        "NumPy",
        numpy.__version__,
        "Pillow",
        PIL.__version__,
        "Tk",
        tkinter.TkVersion,
        "CustomTkinter",
        customtkinter.__version__,
    )


def camera_self_test():
    """Open VIAR, let its automatic controls settle, and read a real frame."""
    camera = Camera()
    try:
        deadline = time.monotonic() + camera.warmup_seconds + 4
        while time.monotonic() < deadline and not camera.is_warmed_up:
            time.sleep(0.1)
        frame = camera.capture()
        if frame is None:
            raise RuntimeError("VIAR-500A did not return a frame")
        print(
            "VIAR camera self-test OK;",
            f"{frame.shape[1]}x{frame.shape[0]}",
            f"frames={camera._frame_sequence}",
        )
    finally:
        camera.close()


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    if "--camera-self-test" in sys.argv:
        camera_self_test()
        return
    if "--version" in sys.argv:
        print(__version__)
        return

    app = CamScanApp()
    app.mainloop()


if __name__ == "__main__":
    main()
