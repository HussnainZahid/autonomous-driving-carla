# SETUP GUIDE

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8 – 3.11 | 3.12+ not yet tested with CARLA |
| CARLA Simulator | 0.9.15 / 0.9.16 | Download from carla.org |
| CUDA (optional) | 11.x / 12.x | YOLOv8 GPU inference |

---

## 1 — Install CARLA

Download **CARLA 0.9.16** from https://github.com/carla-simulator/carla/releases

### Windows
```
CarlaUE4.exe
```

### Linux
```bash
./CarlaUE4.sh
# or headless:
./CarlaUE4.sh -RenderOffScreen
```

---

## 2 — Install the CARLA Python API

The CARLA Python wheel is **not** on PyPI. Find it inside your CARLA folder:

```bash
# Linux/macOS
pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-0.9.16-cp3*-linux-x86_64.whl

# Windows
pip install <CARLA_ROOT>/PythonAPI/carla/dist/carla-0.9.16-cp3*-win_amd64.whl
```

Or add the egg to your path instead:
```bash
export PYTHONPATH=$PYTHONPATH:<CARLA_ROOT>/PythonAPI/carla
```

---

## 3 — Install Python dependencies

```bash
cd autonomous_driving_carla
pip install -r requirements.txt
```

---

## 4 — Verify setup

```bash
python -c "import carla; print('carla OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## 5 — Run commands

### ▶ Full autonomous driving (default — 60 s, live window)
```bash
python main.py
```

### ▶ Headless / server mode (no pygame window)
```bash
python main.py --no-pygame
```

### ▶ Custom duration and output
```bash
python main.py --frames 400 --output outputs/my_run.mp4
```

### ▶ Slower target speed, no NPC traffic
```bash
python main.py --speed 20 --no-traffic
```

### ▶ Visualise the CV pipeline (6-panel figure)
```bash
python visualize_pipeline.py
# save PNG:
python visualize_pipeline.py --save-png outputs/pipeline.png
```

### ▶ Remote CARLA server
```bash
python main.py --host 192.168.1.100 --port 2000
```

### ▶ Full help
```bash
python main.py --help
```

---

## Project structure

```
autonomous_driving_carla/
├── main.py                    # entry point — full driving run
├── visualize_pipeline.py      # 6-panel CV pipeline visualisation
├── requirements.txt
├── configs/
│   └── config.yaml            # default parameters
├── src/
│   ├── lane_detector.py       # Canny + Hough lane detection
│   ├── vehicle_detector.py    # YOLOv8 object detection + distance
│   ├── ai_driver.py           # waypoint + vision decision engine
│   └── cockpit_renderer.py    # Tesla-style HUD renderer
├── notebooks/
│   └── Untitled.ipynb         # original development notebook
├── outputs/                   # video + PNG output (auto-created)
└── docs/
    └── SETUP.md               # this file
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: carla` | Re-check the wheel install or PYTHONPATH |
| `Connection refused (2000)` | Start CarlaUE4 first |
| `Failed to spawn vehicle` | Try a different map or restart CARLA |
| Black camera frames | Increase the warm-up tick count in `main.py` |
| Slow FPS in pygame | Use `--no-pygame` or a lighter YOLO model (`yolov8n`) |
| GPU out-of-memory | Set `device='cpu'` in `VehicleDetector.detect()` |
