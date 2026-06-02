"""
lane_detector.py
----------------
Detects lane lines from a single BGR camera frame using Canny edge
detection + Hough transform.  No simulator data is used.
"""

from collections import deque
import cv2
import numpy as np


class LaneDetector:
    """
    Detects left / right lane lines and estimates lateral steering error.

    Uses a temporal history buffer (deque) to smooth jitter between frames.
    """

    def __init__(self, history_len: int = 8):
        self.left_history:  deque = deque(maxlen=history_len)
        self.right_history: deque = deque(maxlen=history_len)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _roi(self, img: np.ndarray) -> np.ndarray:
        """Mask out everything outside the trapezoidal road region."""
        h, w = img.shape[:2]
        pts = np.array([
            [(int(w * 0.05), int(h * 0.95)),
             (int(w * 0.40), int(h * 0.55)),
             (int(w * 0.60), int(h * 0.55)),
             (int(w * 0.95), int(h * 0.95))]
        ], np.int32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, pts, 255)
        return cv2.bitwise_and(img, mask)

    def _fit_line(self, lines: list, h: int):
        """Fit a single representative line through a list of (x1,y1,x2,y2)."""
        if not lines:
            return None
        xs, ys = [], []
        for x1, y1, x2, y2 in lines:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if len(xs) < 2:
            return None
        try:
            p = np.poly1d(np.polyfit(ys, xs, 1))
            yb, yt = int(h * 0.93), int(h * 0.55)
            xb, xt = int(p(yb)), int(p(yt))
            # Sanity check: discard wildly extrapolated lines
            if abs(xb) > h * 3 or abs(xt) > h * 3:
                return None
            return (xb, yb, xt, yt)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect lanes in *frame* (BGR, HxWx3).

        Returns
        -------
        dict with keys:
          overlay       – BGR image same size as frame with lane visualisation
          edges         – single-channel edge / ROI image
          steering      – float in [-1, 1]; negative = steer left
          lines_found   – int, raw Hough line count
          left          – (xb, yb, xt, yt) or None
          right         – (xb, yb, xt, yt) or None
        """
        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi   = self._roi(edges)
        lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 50,
                                minLineLength=30, maxLineGap=120)

        left_l, right_l = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < 15:
                    continue
                if -2.0 < slope < -0.3:
                    left_l.append((x1, y1, x2, y2))
                elif 0.3 < slope < 2.0:
                    right_l.append((x1, y1, x2, y2))

        ll = self._fit_line(left_l,  h)
        rl = self._fit_line(right_l, h)
        if ll: self.left_history.append(ll)
        if rl: self.right_history.append(rl)

        # Smoothed lines
        sl = (tuple(np.mean(self.left_history,  axis=0).astype(int))
              if self.left_history  else None)
        sr = (tuple(np.mean(self.right_history, axis=0).astype(int))
              if self.right_history else None)

        # Build overlay
        overlay = np.zeros_like(frame)
        if sl and sr:
            pts = np.array(
                [[sl[0], sl[1]], [sl[2], sl[3]],
                 [sr[2], sr[3]], [sr[0], sr[1]]], np.int32)
            cv2.fillPoly(overlay, [pts], (0, 140, 0))
        if sl: cv2.line(overlay, (sl[0], sl[1]), (sl[2], sl[3]), (0, 0, 255), 4)
        if sr: cv2.line(overlay, (sr[0], sr[1]), (sr[2], sr[3]), (0, 0, 255), 4)

        # Lateral error → steering signal
        steering = 0.0
        if sl and sr:
            lane_center = (sl[0] + sr[0]) // 2
            steering = (w // 2 - lane_center) / (w // 2)

        return {
            'overlay':     overlay,
            'edges':       roi,
            'steering':    steering,
            'lines_found': len(left_l) + len(right_l),
            'left':        sl,
            'right':       sr,
        }
