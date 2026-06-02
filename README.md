# 🚗 Vision-Only Autonomous Driving in CARLA Simulator

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CARLA](https://img.shields.io/badge/CARLA-0.9.16-FF6B35?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-7B2FBE?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

<br/>

**A complete vision-only autonomous driving system built inside the CARLA simulator.**  
The car navigates entirely from a **single RGB camera feed** — no GPS, no LiDAR, no radar,  
no ground-truth position data. Just pixels → decisions.

*The same philosophy behind Tesla Autopilot.*

<br/>

 • [⚙️ Installation](#️-installation) • [▶️ Run](#️-run-commands) • [🧠 How It Works](#-how-it-works) • [📊 Results](#-results)

</div>



## 🖼️ Screenshots

<table>
  <tr>
    <td align="center">
      <img src="assets/screenshot1.png" width="100%"/>
      <b>🚦 Traffic Light Detection</b><br/>
      <sub>4 traffic lights detected at 36–50m · ACCELERATE decision · Lane centred</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/screenshot2.png" width="100%"/>
      <b>🚶 Pedestrian & Vehicle Tracking</b><br/>
      <sub>Person at 11.9m · Car at 29m · CRUISE · Steering left · Bird's-eye map active</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/screenshot3.png" width="100%"/>
      <b>🚗 Multi-Vehicle Detection</b><br/>
      <sub>Cars at 13.4m & 39.1m · 2 traffic lights · Lane tracking · CRUISE at 30 km/h</sub>
    </td>
  </tr>
</table>

---

## 🧠 How It Works

The system processes every camera frame through a 3-stage pipeline:

```
📷 Camera Frame (960×540 RGB)  ←  ONLY input to the system
         │
         ├──► 🛣️  Lane Detector
         │         Canny edge detection + Hough transform
         │         → lane lines, steering error signal
         │
         ├──► 🎯  Vehicle Detector (YOLOv8n)
         │         80-class object detection + pinhole distance formula
         │         → objects, distances, threat levels
         │
         └──► 🧠  AI Driver
                   Priority decision tree
                   → throttle / brake / steer commands
                            │
                            └──► 🖥️  Cockpit Renderer
                                       Tesla-style HUD
                                       → MP4 video recording
```

### 🔐 Safety Priority Order

| Priority | Condition | Data Source | Action |
|:---:|---|---|---|
| 1 🔴 | Red traffic light | CARLA traffic light API | Full brake |
| 2 🟡 | Yellow traffic light | CARLA traffic light API | Slow down |
| 3 🛑 | Stop sign detected | YOLOv8 camera pixels | Approach → stop → wait → go |
| 4 💥 | Obstacle < 8m in lane | YOLOv8 camera pixels | Emergency brake |
| 5 ⚠️ | Obstacle < 18m in lane | YOLOv8 camera pixels | Cautious braking |
| 6 🚀 | Road clear | — | Cruise at target speed |

---

## ✨ Features

- 📷 **Vision-only perception** — single RGB camera, zero other sensors
- 🎯 **YOLOv8 object detection** — detects 8 classes: cars, trucks, buses, motorcycles, bicycles, pedestrians, traffic lights, stop signs
- 📏 **Real-time distance estimation** — pinhole camera formula, no depth sensor needed
- 🛣️ **Lane detection** — Canny edges + Hough transform with temporal smoothing (8-frame history buffer)
- 🎯 **Ultra-tight lane-occupancy check** — prevents false braking on adjacent-lane vehicles
- 🛑 **Stop sign state machine** — APPROACHING → STOPPING → STOPPED (2s wait) → DEPARTING
- 🗺️ **Bird's-eye view mini-map** — real-time vehicle positions visualised
- 🖥️ **Tesla-style cockpit HUD** — speedometer, steering wheel, AI decision panel, system status
- 🎬 **Full video recording** — complete cockpit view saved as MP4
- 💻 **CPU friendly** — runs without any GPU using `-RenderOffScreen`
- 🚦 **NPC traffic** — 12 vehicles + 6 pedestrians with CARLA AI autopilot

---

## 🗂️ Project Structure

```
autonomous_driving_carla/
│
├── 📄 main.py                      ← Entry point — full autonomous driving run
├── 📄 visualize_pipeline.py        ← 6-panel CV pipeline visualisation
├── 📄 requirements.txt             ← Python dependencies
│
├── 📁 src/                         ← Core modules
│   ├── 🐍 lane_detector.py         ← Canny + Hough lane detection
│   ├── 🐍 vehicle_detector.py      ← YOLOv8 + distance estimation + threat logic
│   ├── 🐍 ai_driver.py             ← Priority decision engine + waypoint steering
│   ├── 🐍 cockpit_renderer.py      ← Tesla-style HUD renderer
│   └── 🐍 __init__.py
│
├── 📁 configs/
│   └── ⚙️  config.yaml             ← All tunable parameters
│
├── 📁 assets/                      ← Screenshots for README
│   ├── 🖼️  screenshot1.png
│   ├── 🖼️  screenshot2.png
│   └── 🖼️  screenshot3.png
│
├── 📁 notebooks/
│   └── 📓 autonomous_driving_carla.ipynb   ← Original development notebook
│
├── 📁 outputs/                     ← Generated video saved here
│   └── 🎬 autonomous_driving_carla.mp4
│
└── 📁 docs/
    └── 📖 SETUP.md                 ← Detailed setup instructions
```

---

## 🖥️ CV Pipeline — Step by Step

| Step | Operation | Method | Output |
|:---:|---|---|---|
| 1 | Raw camera frame | RGB sensor 960×540 px | Input image |
| 2 | Noise reduction | Grayscale + Gaussian blur (5×5) | Smoothed edges |
| 3 | Edge detection | Canny (threshold: 50, 150) | Edge map |
| 4 | ROI masking | Trapezoidal road region mask | Focused edges |
| 5 | Lane lines | Hough transform + temporal average | Left/right lanes + steering |
| 6 | Object detection | YOLOv8n (conf ≥ 0.40, CPU) | Bounding boxes + classes |
| 7 | Distance estimation | `dist = (real_width × focal) / pixel_width` | Distance per object (metres) |
| 8 | Threat classification | Lane-occupancy + distance thresholds | SAFE / WARNING / DANGER |
| 9 | Decision | Priority tree → VehicleControl | Throttle / brake / steer |
| 10 | HUD rendering | OpenCV drawing | Cockpit frame → MP4 |

---

## ⚙️ Installation

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12 |
| CARLA Simulator | 0.9.16 |
| OS | Windows 10/11 |
| RAM | 8GB+ recommended |
| GPU | Optional (CPU mode supported) |

### Step 1 — Download & launch CARLA

Download **CARLA 0.9.16** from [github.com/carla-simulator/carla/releases](https://github.com/carla-simulator/carla/releases)

Extract to `E:\CARLA_0.9.16\`

### Step 2 — Clone the repository

```bash
git clone https://github.com/HussnainZahid/autonomous-driving-carla.git
cd autonomous-driving-carla
```

### Step 3 — Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 4 — Install CARLA Python API

```bash
pip install E:\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp312-cp312-win_amd64.whl
```

### Step 5 — Install all dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Verify installation

```bash
python -c "import carla; print('CARLA OK')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import cv2; print('OpenCV OK')"
```

All three should print `OK` ✅

---

## ▶️ Run Commands

### Standard run (with live window)
```bash
# Terminal 1 — Start CARLA
E:\CARLA_0.9.16\CarlaUE4.exe

# Terminal 2 — Run the project
python main.py
```

### CPU-only / headless run (recommended for low-end PCs)
```bash
# Terminal 1 — CARLA with no GPU rendering
E:\CARLA_0.9.16\CarlaUE4.exe -RenderOffScreen -quality-level=Low

# Terminal 2 — Run headless
python main.py --no-pygame
```

### Other useful commands
```bash
# Custom duration (400 frames = 20 seconds)
python main.py --frames 400 --no-pygame

# Custom speed and no NPC traffic
python main.py --speed 20 --no-traffic

# Save the 6-panel CV pipeline figure
python visualize_pipeline.py --save-png outputs/pipeline.png

# Connect to remote CARLA server
python main.py --host 192.168.1.100 --port 2000

# Full help
python main.py --help
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Processing speed (CPU) | 15–17 FPS |
| Camera resolution | 960 × 540 px |
| Detection confidence threshold | 0.40 |
| Object classes detected | 8 |
| Distance estimation range | 5m – 60m |
| Simulation FPS | 20 FPS (synchronous) |
| Output video duration | 60 seconds (1200 frames) |
| NPC vehicles | 12 |
| NPC pedestrians | 6 |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `carla` | 0.9.16 | Simulator API |
| `ultralytics` | ≥ 8.0 | YOLOv8 object detection |
| `opencv-python` | ≥ 4.8 | Lane detection, image processing, rendering |
| `numpy` | ≥ 1.24 | Array operations |
| `torch` | ≥ 2.0 | Deep learning backend |
| `pygame` | ≥ 2.5 | Live display window |
| `matplotlib` | ≥ 3.7 | Pipeline visualisation |
| `imageio-ffmpeg` | ≥ 0.4 | Video encoding |
| `Pillow` | ≥ 9.0 | Image utilities |

---

## 🔧 Configuration

All parameters can be tuned in `configs/config.yaml`:

```yaml
driver:
  target_speed: 35.0      # cruise speed in km/h
  steer_kp: 1.2           # steering sensitivity

cv:
  yolo_model: yolov8n.pt  # swap for yolov8s.pt for better accuracy
  lane_history: 8         # temporal smoothing window

simulation:
  fps: 20
  frames: 1200            # total frames to record
```

---

## 🚧 Known Limitations

- Distance estimation uses a fixed focal length — accuracy decreases beyond 60m
- Lane detection struggles in sharp turns and low-contrast roads
- Stop sign detection depends on YOLOv8 confidence — may miss signs in rain/fog
- No path planning — follows CARLA waypoints rather than planned routes

---

## 🔮 Future Improvements

- [ ] Add depth estimation using monocular depth networks (MiDaS)
- [ ] Replace waypoint following with full path planning (A*)
- [ ] Train custom YOLO model on CARLA-specific scenarios
- [ ] Add lane change behaviour
- [ ] Night driving support with image enhancement
- [ ] Multi-camera setup (front + side cameras)

---

## 🙏 Acknowledgements

- [CARLA Simulator](https://carla.org) — Open-source autonomous driving simulator by Intel & Toyota Research
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — State-of-the-art real-time object detection
- [OpenCV](https://opencv.org) — Computer vision library
- Inspired by **Tesla Autopilot's** camera-first autonomous driving philosophy

---

## 👨‍💻 Author

**Hussnain Zahid**  
[![GitHub](https://img.shields.io/badge/GitHub-HussnainZahid-181717?style=flat&logo=github)](https://github.com/HussnainZahid)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify and distribute.

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

</div>
