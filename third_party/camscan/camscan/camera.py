"""
Camera access tailored for the VIAR-500A document camera.

The VIAR exposes a single uncompressed YUYV stream through V4L2.  Keeping that
stream open is important: automatic exposure and white balance only converge
while frames are actually being read.  A background reader also prevents the
2-fps full-resolution mode from blocking the Tk event loop.
"""

import logging
import platform
import threading
import time

import cv2


SYSTEM = platform.system()
if SYSTEM == "Windows":
    API_PREFERENCE = cv2.CAP_DSHOW
elif SYSTEM == "Linux":
    API_PREFERENCE = cv2.CAP_V4L2
else:
    API_PREFERENCE = cv2.CAP_ANY


class Camera:
    """
    Continuously read a camera into a latest-frame buffer.

    The defaults are the verified VIAR-500A full-resolution V4L2 mode.
    ``rotation`` is applied before frames reach document detection.
    """

    def __init__(
        self,
        index: int = 0,
        resolution: tuple[int, int] = (2592, 1944),
        target_fps: float = 2,
        rotation: int = 180,
        warmup_seconds: float = 8.0,
    ):
        self.index = index
        self.resolution = resolution
        self.target_fps = target_fps
        self.rotation = rotation
        self.warmup_seconds = warmup_seconds

        self._video_capture = None
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_sequence = 0
        self._first_frame_time = None
        self.initialize()

    @property
    def is_opened(self) -> bool:
        return bool(self._video_capture and self._video_capture.isOpened())

    @property
    def warmup_remaining(self) -> float:
        if self._first_frame_time is None:
            return self.warmup_seconds
        return max(0.0, self.warmup_seconds - (time.monotonic() - self._first_frame_time))

    @property
    def is_warmed_up(self) -> bool:
        return self.is_opened and self.warmup_remaining <= 0

    def _release(self):
        self._stop_event.set()
        thread = self._reader_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if self._video_capture is not None:
            self._video_capture.release()
        self._video_capture = None
        self._reader_thread = None

    def close(self):
        """Stop frame acquisition and release the V4L2 device."""
        self._release()

    def initialize(self):
        """Open the selected camera and start continuously acquiring frames."""
        self._release()
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._first_frame_time = None
        with self._frame_lock:
            self._latest_frame = None
            self._frame_sequence = 0

        capture = cv2.VideoCapture(self.index, API_PREFERENCE)
        self._video_capture = capture
        if not capture.isOpened():
            logging.error("Cannot open camera index %s", self.index)
            return

        # The VIAR-500A advertises only YUYV at its useful scan resolutions.
        if SYSTEM == "Linux":
            yuyv = cv2.VideoWriter_fourcc(*"YUYV")
            capture.set(cv2.CAP_PROP_FOURCC, yuyv)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual = (
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            capture.get(cv2.CAP_PROP_FPS),
        )
        logging.info(
            "Camera %s opened at %sx%s @ %.2f fps",
            self.index,
            actual[0],
            actual[1],
            actual[2],
        )

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="viar-camera-reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._first_frame_event.wait(timeout=3.0)

    def _reader_loop(self):
        while not self._stop_event.is_set():
            capture = self._video_capture
            if capture is None or not capture.isOpened():
                break
            ok, frame = capture.read()
            if not ok:
                if not self._stop_event.is_set():
                    logging.warning("Could not read a frame from camera %s", self.index)
                    time.sleep(0.1)
                continue

            frame = self._rotate(frame)
            with self._frame_lock:
                self._latest_frame = frame
                self._frame_sequence += 1
                if self._first_frame_time is None:
                    self._first_frame_time = time.monotonic()
                    self._first_frame_event.set()

    def _rotate(self, frame: cv2.Mat) -> cv2.Mat:
        if self.rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def set_index(self, index: int):
        self.index = index
        self.initialize()

    def set_resolution(self, resolution: tuple[int, int]):
        self.resolution = resolution
        # Pick the verified VIAR rate for each advertised resolution.
        rates = {
            (2592, 1944): 2,
            (2048, 1536): 3,
            (1600, 1200): 5,
            (1280, 1024): 7.5,
            (640, 480): 30,
        }
        self.target_fps = rates.get(resolution, self.target_fps)
        self.initialize()

    def set_rotation(self, rotation: int):
        if rotation not in (0, 90, 180, 270):
            raise ValueError("rotation must be one of 0, 90, 180, 270")
        self.rotation = rotation

    def show_settings(self):
        self._video_capture.set(cv2.CAP_PROP_SETTINGS, 1)

    def capture(self) -> cv2.Mat:
        """Return a copy of the newest continuously acquired frame."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def capture_with_sequence(self) -> tuple[cv2.Mat, int]:
        """Return the latest frame and its monotonically increasing sequence."""
        with self._frame_lock:
            if self._latest_frame is None:
                return None, self._frame_sequence
            return self._latest_frame.copy(), self._frame_sequence

    def get_available_device_indices(self) -> list[int]:
        """
        Probe camera indices.  Temporarily release VIAR because V4L2 devices
        generally cannot be opened a second time while already streaming.
        """
        self._release()
        found_camera_indices = []
        for index in range(10):
            dummy_capture = cv2.VideoCapture(index, API_PREFERENCE)
            if dummy_capture.isOpened():
                found_camera_indices.append(index)
            dummy_capture.release()
        self.initialize()
        return found_camera_indices

    def __del__(self):
        self._release()
