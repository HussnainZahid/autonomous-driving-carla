"""
cockpit_renderer.py
-------------------
Renders a Tesla-style cockpit HUD alongside the camera feed.

Layout (1280 x 540):
  LEFT  960 px  – annotated camera frame
  RIGHT 320 px  – bird's-eye view, AI decision panel,
                  steering wheel, speedometer, status bar
"""

import math
import time

import cv2
import numpy as np


class CockpitRenderer:
    """Assembles the full cockpit canvas for display and recording."""

    def __init__(self, frame_w: int = 960, frame_h: int = 540):
        self.fw  = frame_w
        self.fh  = frame_h
        self.cw  = frame_w + 320   # total canvas width
        self.ch  = frame_h         # total canvas height
        self.px  = frame_w         # x-start of right panel
        self.frame_count = 0
        self.t0  = time.time()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, cam_frame: np.ndarray,
               dets: list,
               lane_data: dict,
               decision: dict) -> np.ndarray:
        """
        Build and return the full cockpit canvas (BGR, ch × cw × 3).

        Parameters
        ----------
        cam_frame : annotated camera frame (BGR)
        dets      : list of detection dicts
        lane_data : lane detection result dict
        decision  : AI driver decision dict
        """
        self.frame_count += 1
        canvas = np.full((self.ch, self.cw, 3), 25, dtype=np.uint8)
        canvas[0:self.fh, 0:self.fw] = cam_frame

        px, pw = self.px + 8, 304

        # ---- Camera overlay bars ----
        self._draw_camera_overlay(canvas, decision, dets)

        # ---- Right panel sections ----
        self._draw_bev(canvas,      px, 5,              pw, 210, dets, decision)
        self._draw_ai(canvas,       px, 220,            pw, 120, decision)
        self._draw_steering(canvas, px + 10,  348, 90, decision['steer'])
        self._draw_speed(canvas,    px + 160, 348, 90, decision['speed'])
        self._draw_status(canvas,   px, self.fh - 80,  pw, 72,  decision)

        return canvas

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_camera_overlay(self, canvas, decision, dets):
        """Top status bar drawn directly on the camera portion."""
        cv2.rectangle(canvas, (0, 0), (self.fw, 36), (15, 15, 15), -1)
        cv2.putText(canvas, "VISION-ONLY AUTONOMOUS DRIVING",
                    (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        fps = self.frame_count / max(time.time() - self.t0, 0.001)
        cv2.putText(canvas,
                    f"FPS: {fps:.1f} | Speed: {decision['speed']:.0f} km/h",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        nv = sum(1 for d in dets if d['class'] in ('car', 'truck', 'bus'))
        np_ = sum(1 for d in dets if d['class'] == 'person')
        cv2.putText(canvas, f"Vehicles: {nv}  Persons: {np_}",
                    (self.fw - 220, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Flashing red border + warning banner for CRITICAL urgency
        if decision['urgency'] == 'CRITICAL':
            if self.frame_count % 4 < 2:
                cv2.rectangle(canvas, (0, 0), (self.fw, self.fh), (0, 0, 255), 5)
            mx = self.fw // 2
            cv2.rectangle(canvas, (mx - 150, 40), (mx + 150, 68), (0, 0, 200), -1)
            cv2.putText(canvas, "!! COLLISION WARNING !!",
                        (mx - 140, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def _draw_bev(self, canvas, x, y, w, h, dets, dec):
        """Bird's-eye-view mini-map."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 35), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), 1)
        cv2.putText(canvas, "BIRD'S EYE VIEW",
                    (x + 80, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        rcx = x + w // 2
        rw  = 120
        rl, rr = rcx - rw // 2, rcx + rw // 2
        cv2.rectangle(canvas, (rl, y + 20), (rr, y + h - 5), (50, 50, 50), -1)
        cv2.line(canvas, (rl, y + 20), (rl, y + h - 5), (160, 160, 160), 2)
        cv2.line(canvas, (rr, y + 20), (rr, y + h - 5), (160, 160, 160), 2)
        lw = rw // 3
        for lane in range(1, 3):
            lx = rl + lane * lw
            for dy in range(20, h - 5, 16):
                cv2.line(canvas, (lx, y + dy), (lx, y + dy + 8), (100, 100, 100), 1)

        # Ego car
        ecy = y + h - 30
        cv2.rectangle(canvas, (rcx - 9, ecy - 16), (rcx + 9, ecy + 16), (0, 220, 60), -1)
        cv2.rectangle(canvas, (rcx - 6, ecy - 16), (rcx + 6, ecy - 8), (80, 160, 200), -1)

        for d in dets:
            if d['class'] not in ('car', 'truck', 'bus', 'motorcycle'):
                continue
            cx_map = int(np.clip(rcx + d['relative_x'] * rw * 0.4, rl + 8, rr - 8))
            cy_map = int(np.clip(ecy - 20 - d['relative_y'] * (h - 70), y + 25, ecy - 20))
            c = ((0, 0, 255)   if d['threat'] == 'DANGER'  else
                 (0, 165, 255) if d['threat'] == 'WARNING' else
                 (0, 200, 255))
            cv2.rectangle(canvas, (cx_map - 7, cy_map - 12), (cx_map + 7, cy_map + 12), c, -1)
            cv2.putText(canvas, f"{d['distance']:.0f}m",
                        (cx_map - 10, cy_map - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.22, c, 1)

    def _draw_ai(self, canvas, x, y, w, h, dec):
        """AI decision panel."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 35), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), 1)
        cv2.putText(canvas, "AI DECISION",
                    (x + 90, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        urgency_colours = {
            'CRITICAL': (0, 0, 200), 'HIGH': (0, 100, 220),
            'MEDIUM':   (0, 160, 220), 'LOW': (0, 140, 60),
        }
        bc = urgency_colours.get(dec['urgency'], (80, 80, 80))
        cv2.rectangle(canvas, (x + 8, y + 22), (x + w - 8, y + 50), bc, -1)
        (tw, _), _ = cv2.getTextSize(dec['action'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(canvas, dec['action'],
                    (x + (w - tw) // 2, y + 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        reason = dec['reason']
        for i, start in enumerate(range(0, len(reason), 35)):
            cv2.putText(canvas, reason[start:start + 35],
                        (x + 10, y + 68 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 220, 160), 1)

        cv2.putText(canvas, f"Throttle: {dec['throttle']:.1f}",
                    (x + 10,  y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
        cv2.putText(canvas, f"Brake: {dec['brake']:.1f}",
                    (x + 120, y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
        cv2.putText(canvas, f"Steer: {dec['steer']:+.2f}",
                    (x + 210, y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 100), 1)

    def _draw_steering(self, canvas, x, y, sz, steer):
        """Steering wheel widget."""
        cx, cy = x + sz // 2, y + sz // 2 + 12
        r = sz // 2 - 6
        cv2.putText(canvas, "STEERING", (x + 12, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 255), 1)
        cv2.circle(canvas, (cx, cy), r, (70, 70, 70), 2)
        a  = -steer * 0.8
        ex = int(cx + (r - 4) * math.sin(a))
        ey = int(cy - (r - 4) * math.cos(a))
        cv2.line(canvas, (cx, cy), (ex, ey), (0, 255, 100), 2)
        cv2.circle(canvas, (cx, cy), 4, (0, 200, 255), -1)
        direction = "LEFT" if steer < -0.05 else "RIGHT" if steer > 0.05 else "CENTER"
        colour = (0, 255, 255) if direction != "CENTER" else (0, 200, 0)
        cv2.putText(canvas, direction, (cx - 18, cy + r + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, colour, 1)

    def _draw_speed(self, canvas, x, y, sz, speed):
        """Speedometer widget."""
        cx, cy = x + sz // 2, y + sz // 2 + 12
        r = sz // 2 - 6
        cv2.putText(canvas, "SPEED", (x + 22, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 255), 1)
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 135, 405, (50, 50, 50), 2)
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 135, 255, (0, 140, 0), 2)
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 255, 345, (0, 180, 200), 2)
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, 345, 405, (0, 0, 200), 2)
        na = 135 + (np.clip(speed, 0, 80) / 80) * 270
        nr = math.radians(na)
        nx = int(cx + (r - 8) * math.cos(nr))
        ny = int(cy + (r - 8) * math.sin(nr))
        cv2.line(canvas, (cx, cy), (nx, ny), (255, 255, 255), 2)
        cv2.circle(canvas, (cx, cy), 3, (0, 200, 255), -1)
        cv2.putText(canvas, f"{speed:.0f}",   (cx - 12, cy +  8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(canvas, "km/h",           (cx - 14, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (130, 130, 130), 1)

    def _draw_status(self, canvas, x, y, w, h, dec):
        """Bottom status bar."""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (35, 35, 35), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), 1)
        cv2.putText(canvas, "SYSTEM STATUS",
                    (x + 85, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        info = [
            f"Detections: {dec['n_detections']}",
            f"Dangers: {dec['n_dangers']}    Warnings: {dec['n_warnings']}",
            f"Lane Steer: {dec['steering_cv']:+.3f}",
            f"Frame: {dec['frame']}",
        ]
        for i, text in enumerate(info):
            colour = ((255, 100, 100) if 'Danger' in text and dec['n_dangers'] > 0
                      else (180, 180, 180))
            cv2.putText(canvas, text, (x + 10, y + 28 + i * 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
