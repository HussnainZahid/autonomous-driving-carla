"""
main.py
-------
Vision-only autonomous driving in CARLA.

Run with:
    python main.py                        # default settings
    python main.py --frames 600           # 30-second run
    python main.py --no-pygame            # headless (server / SSH)
    python main.py --output ./my_video.mp4
"""

import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Graceful import checks
# ---------------------------------------------------------------------------
try:
    import carla
except ImportError:
    sys.exit(
        "[ERROR] carla package not found.\n"
        "Make sure CARLA is installed and its Python API is on PYTHONPATH.\n"
        "See docs/SETUP.md for instructions."
    )

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics not found. Run: pip install ultralytics")

from src.lane_detector    import LaneDetector
from src.vehicle_detector import VehicleDetector
from src.ai_driver        import AIDriver
from src.cockpit_renderer import CockpitRenderer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Vision-only CARLA autonomous driving")
    p.add_argument("--host",       default="localhost",  help="CARLA server host")
    p.add_argument("--port",       default=2000, type=int)
    p.add_argument("--frames",     default=1200, type=int, help="Number of frames to record")
    p.add_argument("--fps",        default=20,   type=int, help="Simulation FPS")
    p.add_argument("--speed",      default=35.0, type=float, help="Target cruise speed km/h")
    p.add_argument("--output",     default="outputs/autonomous_driving_carla.mp4")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--no-traffic", action="store_true", help="Skip traffic spawning")
    p.add_argument("--no-pygame",  action="store_true", help="Headless mode (no live window)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# CARLA helpers
# ---------------------------------------------------------------------------
def connect_to_carla(host: str, port: int, timeout: float = 15.0) -> carla.World:
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()
    return client, world


def configure_world(world: carla.World, fps: int) -> None:
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)
    world.set_weather(carla.WeatherParameters(
        cloudiness=10.0, precipitation=0.0,
        sun_altitude_angle=60.0, fog_density=0.0, wetness=0.0
    ))


def spawn_ego_vehicle(world: carla.World, blueprint_library) -> carla.Vehicle:
    for bp_filter in ('vehicle.tesla.model3', 'vehicle.audi.a2', 'vehicle.*'):
        bps = blueprint_library.filter(bp_filter)
        if bps:
            bp = bps[0]
            break

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for sp in spawn_points:
        actor = world.try_spawn_actor(bp, sp)
        if actor:
            actor.set_autopilot(False)
            return actor
    raise RuntimeError("Could not spawn ego vehicle at any spawn point.")


def attach_camera(world: carla.World,
                  vehicle: carla.Vehicle,
                  blueprint_library,
                  width: int = 960,
                  height: int = 540) -> tuple:
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(width))
    cam_bp.set_attribute('image_size_y', str(height))
    cam_bp.set_attribute('fov',          '100')
    cam_bp.set_attribute('sensor_tick',  '0.05')

    transform = carla.Transform(
        carla.Location(x=1.8, z=1.4),
        carla.Rotation(pitch=-5)
    )
    camera      = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    latest_frame: dict = {'image': None}

    def _callback(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        latest_frame['image'] = arr[:, :, :3].copy()   # BGR

    camera.listen(_callback)
    return camera, latest_frame


def spawn_traffic(client: carla.Client,
                  world:  carla.World,
                  ego_location,
                  blueprint_library,
                  n_vehicles: int = 12,
                  n_walkers:  int = 6) -> tuple:
    """Spawn NPC vehicles and pedestrians."""
    # Temporarily go async for spawning (avoids TM timeouts)
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    time.sleep(1.0)
    client.set_timeout(60.0)

    spawn_points = world.get_map().get_spawn_points()
    available    = [sp for sp in spawn_points
                    if sp.location.distance(ego_location) > 25]
    random.shuffle(available)

    vehicle_bps = blueprint_library.filter('vehicle.*')
    tm          = client.get_trafficmanager(8000)

    vehicles = []
    for sp in available[:n_vehicles]:
        bp = random.choice(vehicle_bps)
        if any(k in bp.id for k in ('bike', 'motorcycle', 'isetta')):
            continue
        v = world.try_spawn_actor(bp, sp)
        if v:
            vehicles.append(v)

    for v in vehicles:
        try:
            v.set_autopilot(True, tm.get_port())
        except Exception as e:
            print(f"   [warn] autopilot: {e}")

    walker_bps      = blueprint_library.filter('walker.pedestrian.*')
    controller_bp   = blueprint_library.find('controller.ai.walker')
    walkers         = []

    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        bp     = random.choice(walker_bps)
        walker = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker:
            ctrl = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
            if ctrl:
                walkers.append((walker, ctrl))

    time.sleep(3.0)
    for walker, ctrl in walkers:
        try:
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(1.0 + random.random())
        except Exception:
            pass

    # Switch back to sync
    settings = world.get_settings()
    settings.synchronous_mode   = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)
    for _ in range(10):
        world.tick()

    return vehicles, walkers


def cleanup(world: carla.World,
            camera,
            ego_vehicle,
            traffic_vehicles: list,
            traffic_walkers:  list) -> None:
    camera.stop()
    camera.destroy()
    ego_vehicle.destroy()
    for v in traffic_vehicles:
        try: v.destroy()
        except Exception: pass
    for walker, ctrl in traffic_walkers:
        try:
            ctrl.stop(); ctrl.destroy(); walker.destroy()
        except Exception: pass
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 65)
    print("  AUTONOMOUS DRIVING — VISION-ONLY CONTROL IN CARLA")
    print("=" * 65)

    # ---- Connect & configure ----
    client, world = connect_to_carla(args.host, args.port)
    configure_world(world, args.fps)
    carla_map     = world.get_map()
    print(f"✅ Connected | Map: {carla_map.name} | {args.fps} FPS sync mode")

    blueprint_library = world.get_blueprint_library()

    # ---- Ego vehicle + camera ----
    ego_vehicle = spawn_ego_vehicle(world, blueprint_library)
    camera, latest_frame = attach_camera(world, ego_vehicle, blueprint_library)
    print(f"✅ Spawned: {ego_vehicle.type_id}")
    print(f"✅ Camera: 960×540 @ 100° FOV (ONLY sensor input)")

    # Warm-up
    for _ in range(10):
        world.tick()
    time.sleep(0.5)

    # ---- Traffic ----
    traffic_vehicles, traffic_walkers = [], []
    if not args.no_traffic:
        traffic_vehicles, traffic_walkers = spawn_traffic(
            client, world, ego_vehicle.get_location(), blueprint_library)
        print(f"✅ Traffic: {len(traffic_vehicles)} vehicles, {len(traffic_walkers)} pedestrians")

    # ---- CV + AI modules ----
    yolo_model = YOLO(args.yolo_model)
    lane_det   = LaneDetector()
    veh_det    = VehicleDetector(yolo_model)
    ai_driver  = AIDriver(ego_vehicle, carla_map, target_speed=args.speed)
    cockpit    = CockpitRenderer(960, 540)
    print("✅ CV modules ready (LaneDetector + YOLOv8 + AIDriver)")

    # ---- Output video ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(args.output, fourcc, float(args.fps),
                             (cockpit.cw, cockpit.ch))

    # ---- Optional pygame window ----
    screen = None
    if not args.no_pygame:
        try:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((1280, 540))
            pygame.display.set_caption("Autonomous Driving — Live Cockpit")
            clock  = pygame.time.Clock()
        except Exception as e:
            print(f"[warn] pygame unavailable ({e}); running headless")
            screen = None

    # ---- Main loop ----
    print(f"\n   Recording {args.frames} frames (~{args.frames/args.fps:.0f}s)…")
    print("   Press ESC / Q to stop early.\n")

    running   = True
    frame_num = 0
    t0        = time.time()

    try:
        for frame_num in range(1, args.frames + 1):
            # Quit event
            if screen:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN and \
                            event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
            if not running:
                print("   Stopped by user.")
                break

            world.tick()
            frame = latest_frame['image']
            if frame is None:
                continue

            # ---- CV pipeline ----
            lane_data  = lane_det.detect(frame)
            detections = veh_det.detect(frame)
            decision   = ai_driver.decide_and_control(detections, lane_data, frame_num)

            # ---- Render ----
            cam_vis       = cv2.addWeighted(frame, 1.0, lane_data['overlay'], 0.35, 0)
            cam_vis       = veh_det.draw(cam_vis, detections)
            cockpit_frame = cockpit.render(cam_vis, detections, lane_data, decision)

            out.write(cockpit_frame)

            if screen:
                import pygame
                rgb   = cv2.cvtColor(cockpit_frame, cv2.COLOR_BGR2RGB)
                rgb   = cv2.resize(rgb, (1280, 540))
                surf  = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
                clock.tick(args.fps)

            if frame_num % 100 == 0:
                fps_actual = frame_num / max(time.time() - t0, 0.001)
                print(f"   [{frame_num:4d}/{args.frames}] "
                      f"{fps_actual:.1f} fps | "
                      f"{decision['speed']:.0f} km/h | "
                      f"{decision['action']}")

    except KeyboardInterrupt:
        print("\n   Stopped by keyboard.")
    except Exception as exc:
        print(f"\n   Error at frame {frame_num}: {exc}")
        import traceback; traceback.print_exc()
    finally:
        out.release()
        if screen:
            import pygame; pygame.quit()
        cleanup(world, camera, ego_vehicle, traffic_vehicles, traffic_walkers)

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"  ✅ DONE  |  {frame_num} frames  |  {elapsed:.1f}s")
    if os.path.exists(args.output):
        print(f"  Output:  {args.output}  "
              f"({os.path.getsize(args.output) / 1_048_576:.1f} MB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
