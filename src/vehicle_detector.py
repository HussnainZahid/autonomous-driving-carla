"""
vehicle_detector.py
-------------------
Detects vehicles, pedestrians and traffic signs from a single BGR frame
using YOLOv8 (ultralytics).  No simulator data is used.
"""

import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    Wraps YOLOv8 with distance estimation and threat classification.

    Distance is estimated via the pinhole camera formula:
        distance = (real_width * focal_length) / pixel_width
    """

    # Known approximate real-world widths (metres)
    REAL_WIDTHS: dict = {
        'car':        1.8,
        'truck':      2.5,
        'bus':        2.5,
        'motorcycle': 0.8,
        'bicycle':    0.6,
        'person':     0.5,
    }

    # Focal length tuned for CARLA's 100° FOV @ 960 px wide
    FOCAL: int = 500

    # COCO class IDs we care about
    CLASSES: dict = {
        0:  'person',
        1:  'bicycle',
        2:  'car',
        3:  'motorcycle',
        5:  'bus',
        7:  'truck',
        9:  'traffic light',
        11: 'stop sign',
    }

    def __init__(self, model: YOLO):
        self.model = model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_lane(cx: int, cy: int, w: int, h: int,
                 dist: float, name: str) -> bool:
        """
        Ultra-tight lane-occupancy check.

        Only triggers for objects very close to the image centre-line so
        that adjacent-lane vehicles do not cause false braking.
        """
        if name in ('stop sign', 'traffic light'):
            return False
        if cy < h * 0.3:           # sky / far horizon – skip
            return False

        frame_center   = w // 2
        off_center_pct = abs(cx - frame_center) / (w // 2)   # 0=centre, 1=edge

        if dist < 8:    return off_center_pct < 0.08
        if dist < 15:   return off_center_pct < 0.06
        return              off_center_pct < 0.04

    @staticmethod
    def _threat(name: str, dist: float, in_lane: bool) -> str:
        if name not in ('car', 'truck', 'bus', 'motorcycle', 'person', 'bicycle'):
            return 'SAFE'
        if dist < 8  and in_lane: return 'DANGER'
        if dist < 18 and in_lane: return 'WARNING'
        return 'SAFE'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 on *frame* (BGR, HxWx3).

        Returns a list of detection dicts sorted by ascending distance.
        Each dict contains:
          bbox, class, confidence, distance, threat,
          center, in_lane, side, relative_x, relative_y
        """
        h, w = frame.shape[:2]
        results = self.model(frame, conf=0.40, verbose=False, device='cpu')

        dets = []
        for r in results:
            for box in r.boxes:
                cid  = int(box.cls[0])
                if cid not in self.CLASSES:
                    continue
                name = self.CLASSES[cid]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw   = x2 - x1
                bh   = y2 - y1
                cx   = (x1 + x2) // 2
                cy   = (y1 + y2) // 2

                if bw < 12 or bh < 12:  continue   # noise
                if bw > w * 0.7:        continue   # own bonnet / artefact

                dist    = round((self.REAL_WIDTHS.get(name, 1.8) * self.FOCAL)
                                / max(bw, 1), 1)
                in_lane = self._in_lane(cx, cy, w, h, dist, name)
                threat  = self._threat(name, dist, in_lane)

                dets.append({
                    'bbox':       (x1, y1, x2, y2),
                    'class':      name,
                    'confidence': conf,
                    'distance':   dist,
                    'threat':     threat,
                    'center':     (cx, cy),
                    'in_lane':    in_lane,
                    'side':       'left' if cx < w // 2 else 'right',
                    'relative_x': (cx - w // 2) / (w // 2),
                    'relative_y': 1.0 - (cy / h),
                })

        dets.sort(key=lambda d: d['distance'])
        return dets

    def draw(self, frame: np.ndarray, dets: list[dict]) -> np.ndarray:
        """Annotate *frame* with bounding boxes, labels and distance."""
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = d['bbox']
            colour = (
                (0, 0, 255)   if d['threat'] == 'DANGER'  else
                (0, 165, 255) if d['threat'] == 'WARNING' else
                (0, 255, 0)
            )
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            label  = f"{d['class']} {d['distance']}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return vis
