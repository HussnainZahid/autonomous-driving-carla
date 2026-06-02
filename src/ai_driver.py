"""
ai_driver.py
------------
Waypoint-following + vision-based safety decision engine.

Steering comes from CARLA waypoints; all *safety* decisions
(braking, stop signs, traffic lights) are derived from camera
detections and the CARLA traffic-light API.
"""

import math

import carla
import numpy as np


class AIDriver:
    """
    Autonomous driving controller.

    Priority order (highest → lowest):
        1. Red / Yellow traffic light  (via CARLA V2X-style API)
        2. Stop sign                   (detected by YOLOv8 from camera)
        3. Emergency obstacle in lane  (YOLOv8 DANGER)
        4. Warning obstacle in lane    (YOLOv8 WARNING)
        5. Speed management            (target cruise speed)
    """

    def __init__(self, vehicle: carla.Vehicle,
                 carla_map: carla.Map,
                 target_speed: float = 35.0):
        self.vehicle      = vehicle
        self.map          = carla_map
        self.target_speed = target_speed   # km/h
        self.speed        = 0.0
        self.log:   list  = []
        self.steer_kp     = 1.2
        self.prev_steer   = 0.0

        # Stop-sign state machine
        # States: NONE → APPROACHING → STOPPING → STOPPED → DONE
        self.stop_sign_state:    str = 'NONE'
        self.stop_sign_timer:    int = 0
        self.stop_sign_cooldown: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_waypoint_steer(self):
        """
        Calculate steering needed to stay in the current CARLA lane.

        Returns (steer: float, on_road: bool).
        """
        transform = self.vehicle.get_transform()
        location  = transform.location
        forward   = transform.get_forward_vector()

        wp = self.map.get_waypoint(location, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0, False

        next_wps = wp.next(8.0)
        if not next_wps:
            return 0.0, False
        target_loc = next_wps[0].transform.location

        dx = target_loc.x - location.x
        dy = target_loc.y - location.y

        target_angle  = math.atan2(dy, dx)
        forward_angle = math.atan2(forward.y, forward.x)
        angle_diff    = target_angle - forward_angle

        # Normalise to [-π, π]
        while angle_diff >  math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi

        steer   = float(np.clip(self.steer_kp * angle_diff, -1.0, 1.0))
        on_road = wp.lane_type == carla.LaneType.Driving
        return steer, on_road

    def _check_traffic_light(self) -> str:
        """Return 'RED', 'YELLOW', or 'GREEN'."""
        if self.vehicle.is_at_traffic_light():
            light = self.vehicle.get_traffic_light()
            if light is not None:
                state = light.get_state()
                if state == carla.TrafficLightState.Red:    return 'RED'
                if state == carla.TrafficLightState.Yellow: return 'YELLOW'
        return 'GREEN'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_and_control(self, detections: list[dict],
                           lane_data: dict,
                           frame_num: int) -> dict:
        """
        Compute and apply VehicleControl; return a decision summary dict.

        Parameters
        ----------
        detections : list of dicts from VehicleDetector.detect()
        lane_data  : dict from LaneDetector.detect()
        frame_num  : current frame index (for logging)
        """
        velocity   = self.vehicle.get_velocity()
        self.speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        wp_steer,  on_road    = self._get_waypoint_steer()
        lane_steer             = lane_data.get('steering',    0)
        lanes_found            = lane_data.get('lines_found', 0)
        light_state            = self._check_traffic_light()

        dangers   = [d for d in detections if d['threat'] == 'DANGER']
        warnings  = [d for d in detections if d['threat'] == 'WARNING']
        stop_signs = [d for d in detections if d['class'] == 'stop sign']

        # -- Stop-sign cooldown tick --
        if self.stop_sign_cooldown > 0:
            self.stop_sign_cooldown -= 1

        # Trigger stop-sign approach
        if (stop_signs and
                self.stop_sign_state   == 'NONE' and
                self.stop_sign_cooldown == 0 and
                stop_signs[0]['distance'] < 25):
            self.stop_sign_state = 'APPROACHING'

        # ----------------------------------------------------------------
        # DECISION TREE
        # ----------------------------------------------------------------
        throttle = 0.0
        brake    = 0.0
        steer    = wp_steer
        action   = 'CRUISE'
        reason   = 'Road clear'

        if light_state == 'RED':
            brake, action = 0.8, 'STOP — RED LIGHT'
            reason = 'Red traffic light — stopping'

        elif light_state == 'YELLOW':
            if self.speed > 15:
                brake, action = 0.5, 'SLOW — YELLOW'
                reason = 'Yellow light — slowing down'
            else:
                throttle, action = 0.3, 'PROCEED'
                reason = 'Yellow light — proceeding'

        elif self.stop_sign_state == 'APPROACHING':
            brake, action = 0.4, 'SLOW — STOP SIGN'
            reason = 'Stop sign ahead — slowing down'
            if stop_signs and stop_signs[0]['distance'] < 12:
                self.stop_sign_state = 'STOPPING'

        elif self.stop_sign_state == 'STOPPING':
            brake, action = 0.8, 'STOP — SIGN'
            reason = 'Stopping at stop sign'
            if self.speed < 2.0:
                self.stop_sign_state = 'STOPPED'
                self.stop_sign_timer = 40      # 2 s at 20 fps

        elif self.stop_sign_state == 'STOPPED':
            brake, action = 1.0, 'WAITING — STOP'
            self.stop_sign_timer -= 1
            reason = f'Stopped at sign ({self.stop_sign_timer / 20:.1f}s left)'
            if self.stop_sign_timer <= 0:
                self.stop_sign_state = 'DONE'

        elif self.stop_sign_state == 'DONE':
            throttle, action = 0.5, 'DEPARTING STOP'
            reason = 'Leaving stop sign — accelerating'
            if self.speed > 15:
                self.stop_sign_state    = 'NONE'
                self.stop_sign_cooldown = 100  # ~5 s

        elif dangers:
            closest  = dangers[0]
            brake, action = 1.0, 'EMERGENCY BRAKE'
            reason = f"DANGER: {closest['class']} at {closest['distance']}m!"

        elif warnings:
            closest = warnings[0]
            if closest['distance'] < 10:
                brake, action = 0.6, 'BRAKE'
                reason = f"Braking: {closest['class']} at {closest['distance']}m"
            else:
                throttle, action = 0.25, 'CAUTION'
                reason = f"Watching: {closest['class']} at {closest['distance']}m"

        elif self.speed < self.target_speed - 5:
            throttle, action = 0.55, 'ACCELERATE'
            reason = f'Speeding up to {self.target_speed:.0f} km/h'

        elif self.speed > self.target_speed + 5:
            brake, action = 0.25, 'SLOW DOWN'
            reason = f'Reducing from {self.speed:.0f} km/h'

        else:
            throttle, action = 0.35, 'CRUISE'
            reason = f'Cruising at {self.speed:.0f} km/h — road clear'

        # Blend lane-detection steering if confident
        if lanes_found >= 3 and abs(lane_steer) < 0.5:
            steer = 0.8 * wp_steer + 0.2 * lane_steer

        # Temporal smoothing
        steer = float(np.clip(0.7 * steer + 0.3 * self.prev_steer, -1.0, 1.0))
        self.prev_steer = steer

        # Apply control to CARLA vehicle
        control          = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0, 1))
        control.brake    = float(np.clip(brake,    0, 1))
        control.steer    = steer
        control.hand_brake = False
        self.vehicle.apply_control(control)

        # Urgency label for HUD
        urgency = 'LOW'
        if 'EMERGENCY' in action or 'RED' in action:            urgency = 'CRITICAL'
        elif 'BRAKE' in action  or 'STOP' in action:            urgency = 'HIGH'
        elif 'CAUTION' in action or 'YELLOW' in action or \
             'SLOW' in action:                                   urgency = 'MEDIUM'

        decision = {
            'action':      action,   'reason':       reason,
            'urgency':     urgency,  'speed':        self.speed,
            'throttle':    throttle, 'brake':        brake,
            'steer':       steer,    'steering_cv':  lane_steer,
            'wp_steer':    wp_steer, 'n_detections': len(detections),
            'n_dangers':   len(dangers), 'n_warnings': len(warnings),
            'light':       light_state, 'on_road':   on_road,
            'stop_sign':   self.stop_sign_state, 'frame': frame_num,
        }
        self.log.append(decision)
        return decision
