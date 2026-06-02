# 🚗 Vision-Only Autonomous Driving in CARLA Simulator

![Python](https://img.shields.io/badge/Python-3.12-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.16-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete **vision-only autonomous driving system** built inside the 
[CARLA](https://carla.org) simulator. The car navigates entirely from 
a single RGB camera feed — no GPS, no LiDAR, no radar, no ground-truth 
position data. Just pixels → decisions, the same philosophy behind 
**Tesla Autopilot**.

---

## 📽️ Demo

> Car navigating traffic, stopping at red lights and stop signs,
> all from camera input only.

```
outputs/autonomous_driving_carla.mp4
```

---

## 🧠 How It Works

```
Camera Frame (960×540 RGB)
        │
        ├──► Lane Detector       Canny edges + Hough lines → steering signal
        │
        ├──► Vehicle Detector    YOLOv8n → objects, distances, threat level
        │
        └──► AI Driver           Priority decision tree → throttle/brake/steer
                │
                └──► Cockpit Renderer   Tesla-style HUD + MP4 recording
```

### Safety Priority Order
| Priority | Trigger | Source |
|---|---|---|
| 1 🔴 | Red / Yellow traffic light | CARLA traffic light API |
| 2 🛑 | Stop sign detected | YOLOv8 camera detection |
| 3 💥 | Emergency obstacle < 8m in lane | YOLOv8 camera detection |
| 4 ⚠️ | Warning obstacle < 18m in lane | YOLOv8 camera detection |
| 5 🚀 | Speed management | Cruise control logic |

---

## ✨ Features

- **Vision-only perception** — single RGB camera, no other sensors
- **YOLOv8 object detection** — detects cars, trucks, pedestrians,
  stop signs, traffic lights with real-time distance estimation
- **Lane detection** — Canny edge detection + Hough transform with
  temporal smoothing across frames
- **Smart lane-occupancy check** — ultra-tight centre-line filter
  prevents false braking on adjacent-lane vehicles
- **Stop sign state machine** — approaches, stops, waits 2 seconds,
  departs — all triggered from camera pixels
- **Tesla-style cockpit HUD** — bird's-eye view, speedometer,
  steering wheel, AI decision panel
- **Full video recording** — cockpit view saved as MP4
- **CPU friendly** — runs without a GPU using `-RenderOffScreen`

---

## 🗂️ Project Structure

```
autonomous_driving_carla/
├── main.py                    ← entry point
├── visualize_pipeline.py      ← 6-panel CV pipeline figure
├── requirements.txt
├── configs/
│   └── config.yaml            ← all tunable parameters
├── src/
│   ├── lane_detector.py       ← Canny + Hough lane detection
│   ├── vehicle_detector.py    ← YOLOv8 + distance + threat
│   ├── ai_driver.py           ← decision engine
│   └── cockpit_renderer.py    ← HUD renderer
├── notebooks/
│   └── autonomous_driving_carla.ipynb
├── outputs/                   ← video saved here
└── docs/
    └── SETUP.md
```

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.12
- CARLA Simulator 0.9.16
- CPU or GPU (runs fine on CPU only)

### 1 — Clone the repo
```bash
git clone https://github.com/YourUsername/autonomous-driving-carla.git
cd autonomous-driving-carla
```

### 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS
```

### 3 — Install CARLA Python API
```bash
pip install E:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl
```

### 4 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run Commands

```bash
# Start CARLA first (no GPU required)
E:\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -quality-level=Low

# Run autonomous driving
python main.py --no-pygame

# Custom duration
python main.py --frames 400 --no-pygame

# Visualise CV pipeline (6-panel figure)
python visualize_pipeline.py --save-png outputs/pipeline.png

# All options
python main.py --help
```

---

## 🖥️ CV Pipeline

The full perception pipeline from raw pixels to driving decisions:

| Step | Method | Output |
|---|---|---|
| 1. Raw frame | RGB camera 960×540 | Input image |
| 2. Preprocessing | Grayscale + Gaussian blur | Noise reduction |
| 3. Edge detection | Canny (50, 150) | Road edges |
| 4. Lane detection | Hough transform + ROI mask | Lane lines + steering |
| 5. Object detection | YOLOv8n + distance formula | Threats + distances |
| 6. Decision | Priority decision tree | Throttle / brake / steer |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 object detection |
| `opencv-python` | Lane detection, image processing |
| `numpy` | Array operations |
| `pygame` | Live display window |
| `matplotlib` | Pipeline visualisation |
| `imageio-ffmpeg` | Video encoding |
| `carla` | Simulator API |

---

## 🙏 Acknowledgements

- [CARLA Simulator](https://carla.org) — Open-source autonomous driving simulator
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Real-time object detection
- Inspired by Tesla Autopilot's camera-first autonomous driving approach

---

## 📄 License

This project is licensed under the MIT License.