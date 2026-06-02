"""
visualize_pipeline.py
---------------------
Standalone script: connect to CARLA, grab one frame, and display the
complete 6-panel CV pipeline.

Run:
    python visualize_pipeline.py
    python visualize_pipeline.py --save-png outputs/pipeline.png
"""

import argparse
import sys
import time
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import carla
except ImportError:
    sys.exit("[ERROR] carla package not found. See docs/SETUP.md.")

from ultralytics import YOLO
from src.lane_detector    import LaneDetector
from src.vehicle_detector import VehicleDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host",       default="localhost")
    p.add_argument("--port",       default=2000, type=int)
    p.add_argument("--save-png",   default=None,
                   help="Optional path to save the figure as PNG")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    return p.parse_args()


def main():
    args = parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(15.0)
    world  = client.get_world()

    # Sync mode
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0, sun_altitude_angle=60.0))

    library = world.get_blueprint_library()

    # Spawn a temporary ego vehicle
    bp  = library.filter('vehicle.tesla.model3')[0]
    sps = world.get_map().get_spawn_points()
    ego = None
    for sp in random.sample(sps, len(sps)):
        ego = world.try_spawn_actor(bp, sp)
        if ego: break
    assert ego, "Could not spawn vehicle"
    ego.set_autopilot(False)

    # Camera
    cam_bp = library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '960')
    cam_bp.set_attribute('image_size_y', '540')
    cam_bp.set_attribute('fov',          '100')
    cam     = world.spawn_actor(cam_bp,
                                carla.Transform(carla.Location(x=1.8, z=1.4),
                                                carla.Rotation(pitch=-5)),
                                attach_to=ego)
    latest  = {'img': None}
    cam.listen(lambda img: latest.update(
        {'img': np.frombuffer(img.raw_data, np.uint8)
                .reshape((img.height, img.width, 4))[:, :, :3].copy()}))

    # Warm-up
    for _ in range(10):
        world.tick()
    time.sleep(0.5)

    frame = latest['img']
    assert frame is not None, "No camera frame received"

    # ---- CV pipeline ----
    lane_det = LaneDetector()
    veh_det  = VehicleDetector(YOLO(args.yolo_model))

    ld   = lane_det.detect(frame)
    dets = veh_det.detect(frame)

    gray = cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    panels = [
        (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
         "1. RAW CAMERA FRAME\n(Our ONLY input — pixels)"),
        (gray, "2. GRAYSCALE + BLUR\n(Noise reduction)"),
        (ld['edges'], "3. CANNY EDGES + ROI\n(Road features only)"),
        (cv2.cvtColor(
            cv2.addWeighted(frame, 1.0, ld['overlay'], 0.5, 0),
            cv2.COLOR_BGR2RGB),
         f"4. LANE DETECTION\n(Hough — {ld['lines_found']} lines)"),
        (cv2.cvtColor(veh_det.draw(frame, dets), cv2.COLOR_BGR2RGB),
         f"5. YOLOv8 DETECTION\n({len(dets)} objects + distance)"),
        (cv2.cvtColor(
            veh_det.draw(cv2.addWeighted(frame, 1.0, ld['overlay'], 0.4, 0),
                         dets),
            cv2.COLOR_BGR2RGB),
         "6. COMBINED OUTPUT\n(This drives the car)"),
    ]

    cmaps = [None, 'gray', 'gray', None, None, None]
    for ax, (img, title), cmap in zip(axes.flat, panels, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(
        "VISION-ONLY CV PIPELINE — Camera Pixels → Driving Decisions\n"
        "No simulator data used! Same approach as Tesla Autopilot.",
        fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if args.save_png:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.save_png)), exist_ok=True)
        plt.savefig(args.save_png, bbox_inches='tight', dpi=120)
        print(f"Saved: {args.save_png}")

    plt.show()

    # Cleanup
    cam.stop(); cam.destroy(); ego.destroy()
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


if __name__ == "__main__":
    main()
