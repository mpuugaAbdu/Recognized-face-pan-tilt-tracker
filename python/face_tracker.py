"""Pan/tilt target generation for a recognized face.

This module deliberately contains no recognition logic.  It receives the box of
an already-recognized face and converts its offset from the image centre into
small, bounded corrections for the two PCA9685 servo channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from arduino.app_utils import Bridge


# Calibrated PCA9685 pulse limits supplied with the original servo test.
PAN_MIN = 110
PAN_MAX = 500
TILT_MIN = 110
TILT_MAX = 200

# Change either sign if that axis follows in the opposite direction after the
# camera bracket is assembled.  The defaults match the common pan/tilt layout.
PAN_DIRECTION = -1
TILT_DIRECTION = 1

# A small centre region prevents constant corrections caused by detector noise.
PAN_DEAD_ZONE = 0.035
TILT_DEAD_ZONE = 0.045

# Maximum target change for one recognition update.  The MCU applies its own
# 20 ms slew limiter as well, so motion remains smooth even after a large jump.
PAN_CORRECTION_MAX = 36
TILT_CORRECTION_MAX = 11
COMMAND_INTERVAL_SECONDS = 0.022
ERROR_SMOOTHING = 0.56

# Lost-face search motion. The scan remains inside the calibrated servo limits
# and changes tilt only when pan reaches an edge.
SEARCH_COMMAND_INTERVAL_SECONDS = 0.045
SEARCH_PAN_STEP = 20
SEARCH_TILT_LEVELS = (
    (TILT_MIN + TILT_MAX) // 2,
    min(TILT_MAX, (TILT_MIN + TILT_MAX) // 2 + 22),
    max(TILT_MIN, (TILT_MIN + TILT_MAX) // 2 - 22),
)


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _correction(error: float, dead_zone: float, maximum: int) -> int:
    """Return a proportional correction after removing a symmetric dead zone."""
    magnitude = abs(error)
    if magnitude <= dead_zone:
        return 0
    scaled = min(1.0, (magnitude - dead_zone) / (1.0 - dead_zone))
    correction = max(1, round(scaled * maximum))
    return correction if error > 0 else -correction


@dataclass
class TrackingCommand:
    pan: int
    tilt: int
    horizontal_error: float
    vertical_error: float


class FaceTracker:
    """Maintain servo targets and send them to the UNO Q microcontroller."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pan = (PAN_MIN + PAN_MAX) // 2
        self._tilt = (TILT_MIN + TILT_MAX) // 2
        self._filtered_x = 0.0
        self._filtered_y = 0.0
        self._last_command_at = 0.0
        self._search_direction = 1
        self._search_tilt_index = 0
        self._last_search_at = 0.0

    def update(
        self,
        box: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> TrackingCommand:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")

        x1, y1, x2, y2 = (int(value) for value in box)
        face_x = (x1 + x2) / 2.0
        face_y = (y1 + y2) / 2.0
        horizontal_error = (face_x - frame_width / 2.0) / (frame_width / 2.0)
        vertical_error = (face_y - frame_height / 2.0) / (frame_height / 2.0)

        with self._lock:
            now = time.monotonic()
            self._filtered_x += ERROR_SMOOTHING * (horizontal_error - self._filtered_x)
            self._filtered_y += ERROR_SMOOTHING * (vertical_error - self._filtered_y)
            horizontal_error, vertical_error = self._filtered_x, self._filtered_y

            # Detection callbacks may arrive much faster than a hobby servo can
            # react.  Rate limiting makes speed independent of camera FPS and
            # prevents many queued bridge calls from causing overshoot.
            if now - self._last_command_at < COMMAND_INTERVAL_SECONDS:
                return TrackingCommand(
                    self._pan, self._tilt, horizontal_error, vertical_error
                )
            self._last_command_at = now

            pan_delta = PAN_DIRECTION * _correction(
                horizontal_error, PAN_DEAD_ZONE, PAN_CORRECTION_MAX
            )
            tilt_delta = TILT_DIRECTION * _correction(
                vertical_error, TILT_DEAD_ZONE, TILT_CORRECTION_MAX
            )
            self._pan = _clamp(self._pan + pan_delta, PAN_MIN, PAN_MAX)
            self._tilt = _clamp(self._tilt + tilt_delta, TILT_MIN, TILT_MAX)
            pan, tilt = self._pan, self._tilt

        # The MCU callback only updates targets; actual motion is performed by
        # its non-blocking 20 ms control loop.
        Bridge.call("set_face_target", pan, tilt)
        return TrackingCommand(pan, tilt, horizontal_error, vertical_error)

    def search_step(self) -> TrackingCommand:
        """Sweep through the safe pan range and several tilt levels."""
        with self._lock:
            now = time.monotonic()
            if now - self._last_search_at < SEARCH_COMMAND_INTERVAL_SECONDS:
                return TrackingCommand(self._pan, self._tilt, 0.0, 0.0)
            self._last_search_at = now

            next_pan = self._pan + self._search_direction * SEARCH_PAN_STEP
            if next_pan >= PAN_MAX or next_pan <= PAN_MIN:
                self._pan = _clamp(next_pan, PAN_MIN, PAN_MAX)
                self._search_direction *= -1
                self._search_tilt_index = (
                    self._search_tilt_index + 1
                ) % len(SEARCH_TILT_LEVELS)
                self._tilt = SEARCH_TILT_LEVELS[self._search_tilt_index]
            else:
                self._pan = next_pan
            pan, tilt = self._pan, self._tilt

        Bridge.call("set_face_target", pan, tilt)
        return TrackingCommand(pan, tilt, 0.0, 0.0)

    def stop_search(self) -> None:
        """Reset search timing when normal recognized-face tracking resumes."""
        with self._lock:
            self._last_search_at = 0.0


face_tracker = FaceTracker()
