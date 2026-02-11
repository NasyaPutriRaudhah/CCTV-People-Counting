<div align="center">

# 🎯 CCTV People Counting System

### Real-time People Counting & Tracking using YOLO26 Computer Vision

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![YOLO26](https://img.shields.io/badge/YOLO26-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)


[Features](#-features) •
[Tech Stack](#-tech-stack) •
[Installation](#-installation) •
[Usage](#-usage) •
[Documentation](#-documentation) •
[Contributing](#-contributing)

---

</div>

## 📖 Overview

A production-ready, modular **people counting and tracking system** designed for smart building applications. This system processes CCTV footage in real-time to count people using two complementary methods:

1. **🔲 Region of Interest (ROI) Detection** - Monitors people within specific zones
2. **📏 Line Crossing Detection** - Tracks people crossing virtual boundaries with directional entry/exit counting

Perfect for retail analytics, building management, occupancy monitoring, and smart city applications.

### 🎥 Demo

```
┌─────────────────────────┬─────────────────────────┐
│   Video 1: ROI Count    │  Video 2: Line Cross    │
│   [Zone monitoring]     │  [Entry/Exit tracking]  │
│                         │                         │
│   ┌───────────┐         │      ← EXIT LINE        │
│   │  ROI Zone │ 👤👤    │         👤👤            │
│   │    👤👤👤  │         │      ← ENTER LINE       │
│   └───────────┘         │                         │
│   Count: 5              │   Current: 12           │
└─────────────────────────┴─────────────────────────┘
           Live Statistics & Analytics
```

---

## ✨ Features

### Core Capabilities
- **Dual Detection Modes** - ROI-based and line-crossing methods
- **Real-time Processing** - GPU-accelerated YOLO inference (30+ FPS)
- **RTSP Support** - Live camera feed processing with auto-reconnection
- **Data Persistence** - Automatic saving of counting data to JSON
- **Live Visualization** - Multi-panel display with overlays and statistics

### Advanced Features
- **Performance Monitoring** - Detailed inference timing and FPS tracking
- **Configurable Thresholds** - Fine-tune detection sensitivity
- **Duplicate Filtering** - Intelligent filtering of overlapping detections
- **Event Logging** - Complete history of all crossing events
- **Frame Buffering** - Prevents frame drops from RTSP streams
- **Smoothed Tracking** - Temporal smoothing for stable counts

### Production-Ready
- **Modular Architecture** - Clean separation of concerns
- **Centralized Config** - Single source for all settings
- **Error Handling** - Graceful recovery from connection issues
- **Scalable Design** - Easy to add more cameras/zones
- **Well Documented** - Comprehensive guides and examples

---

## 🛠️ Tech Stack

### Computer Vision & AI
| Technology | Version | Purpose |
|------------|---------|---------|
| **YOLO26** | Latest | Real-time object detection |
| **Ultralytics** | 8.0+ | YOLO framework |
| **OpenCV** | 4.8+ | Video processing & visualization |
| **PyTorch** | 2.0+ | Deep learning backend |
| **CUDA** | 11.8+ | GPU acceleration (optional) |

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **NumPy** | 1.24+ | Numerical computations |
| **cvzone** | 1.6+ | Computer vision utilities |
| **Python** | 3.8+ | Programming language |

### Architecture
- **Threading** - Concurrent RTSP capture and processing
- **JSON** - Data persistence and configuration
- **FFMPEG** - RTSP stream handling
- **ByteTrack** - Multi-object tracking algorithm

### System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2 CPU cores
- CPU-only mode: ~15-20 FPS

**Recommended:**
- Python 3.10+
- 8GB RAM
- 4 CPU cores
- NVIDIA GPU (4GB+ VRAM)
- CUDA 11.8+
- GPU mode: 30-60 FPS

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────┐
│                   Main System                   │
│                   (main.py)                     │
└────────┬────────────────────────────────────┬───┘
         │                                    │
    ┌────▼────┐                         ┌────▼────┐
    │ Video 1 │                         │ Video 2 │
    │   ROI   │                         │  Line   │
    └────┬────┘                         └────┬────┘
         │                                    │
    ┌────▼─────────────┐         ┌───────────▼────────┐
    │  YOLO Detection  │         │   RTSP Capture     │
    │  (models.py)     │         │   Thread           │
    └────┬─────────────┘         └────┬───────────────┘
         │                             │
    ┌────▼─────────────┐         ┌────▼───────────────┐
    │  ROI Processor   │         │  Frame Buffer      │
    │  (processors.py) │         │  (buffer.py)       │
    └────┬─────────────┘         └────┬───────────────┘
         │                             │
         │                        ┌────▼───────────────┐
         │                        │  Line Tracker      │
         │                        │  (trackers.py)     │
         │                        └────┬───────────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
            ┌───────▼──────────┐
            │  Data Manager    │
            │  (persistence)   │
            └───────┬──────────┘
                    │
            ┌───────▼──────────┐
            │  Display Output  │
            │  (display.py)    │
            └──────────────────┘
```

### Module Overview

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `main.py` | 170 | Orchestration & main loop |
| `config.py` | 82 | Configuration management |
| `models.py` | 14 | YOLO model loading |
| `processors.py` | 189 | Video processing logic |
| `trackers.py` | 145 | Line crossing detection |
| `buffer.py` | 88 | Frame buffering |
| `data_persistence.py` | 127 | Data storage |
| `rtsp_handler.py` | 92 | RTSP connection |
| `display.py` | 163 | Visualization |
| `timers.py` | 48 | Performance tracking |
| `utils.py` | 49 | Helper functions |

**Total:** ~1,170 lines of clean, modular code

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ferbos/cctv-people-counting.git
cd cctv-people-counting
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model

```bash
# The system will automatically download YOLOv8 models on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo26n.pt
```

### 5. Verify Installation

```bash
python -c "import cv2, torch; print(f'OpenCV: {cv2.__version__}, PyTorch: {torch.__version__}')"
```

---

## 💻 Usage

### Basic Usage

```bash
python main.py
```

Press **ESC** to exit gracefully.

### Configuration

All settings are in `config.py`:

```python
# Video Sources
VIDEO1_PATH = 'path/to/video.mp4'
VIDEO2_PATH = 'rtsp://username:password@ip:port/stream'

# YOLO Model
MODEL_PATH = 'yolo26n.pt'

# Detection Settings
CONF_THRESHOLD = 0.3  # Confidence threshold (0.0-1.0)
IOU_THRESHOLD = 0.65  # IoU threshold for NMS

# Line Crossing Positions
LINE2_ENTER_Y = 340  # Entry line Y-coordinate
LINE2_EXIT_Y = 390   # Exit line Y-coordinate

# Starting Count
STARTING_COUNT = 12  # Initial people count
```

### ROI Definition

Define your Region of Interest polygon:

```python
ROI1_POINTS = np.array([
    [396, 295],  # Top-left
    [717, 290],  # Top-right
    [740, 344],  # Bottom-right
    [451, 354],  # Bottom-left
    # ... more points for complex shapes
], dtype=np.int32)
```

### Example Workflows

**1. Monitor Store Entrance (Line Crossing)**

```python
# config.py
VIDEO2_PATH = 'rtsp://admin:pass@192.168.1.100:554/stream'
LINE2_ENTER_Y = 300  # Horizontal line at entrance
LINE2_EXIT_Y = 350
STARTING_COUNT = 0
```

**2. Monitor Conference Room (ROI)**

```python
# config.py
VIDEO1_PATH = 'conference_room.mp4'
ROI1_POINTS = np.array([
    # Define room area coordinates
])
```

**3. Multi-Camera Setup**

Add more video sources by extending the architecture - see `ARCHITECTURE.md` for guidance.

---

## 📊 Output & Data

### Real-time Display

The system shows:
- ✅ Live video feeds with detection overlays
- 📊 Current count statistics
- 📈 FPS and performance metrics
- 🔄 Buffer status
- 📉 Entry/exit tracking

### JSON Data File

`line_crossing_data.json` contains:

```json
{
  "starting_count": 12,
  "current_count": 15,
  "total_entries": 25,
  "total_exits": 22,
  "session_start": "2024-02-11T10:30:00",
  "last_update": "2024-02-11T14:45:00",
  "history": [
    {
      "timestamp": "2024-02-11T10:35:12",
      "type": "entry",
      "delta": 1,
      "count_after": 13
    }
  ]
}
```

### Performance Report

On exit, the system prints:

```
FINAL REPORT - SIMPLIFIED TRACKING SYSTEM
================================================================================
⏱️  SESSION INFORMATION
   Started:      2024-02-11T10:30:00
   Runtime:      4.25 hours

👥 LINE CROSSING SUMMARY
   Starting count:       12 people
   Current count:        15 people
   Total entries:        +25 crossings
   Total exits:          -22 crossings
   Net change:           +3 people

📊 INFERENCE PERFORMANCE
   Video 1: 15,300 frames, 25.4 ms avg
   Video 2: 15,280 frames, 27.1 ms avg
   ✅ Can handle real-time 30 FPS
================================================================================
```

---

## 🎯 How It Works

### 1. ROI Detection (Video 1)

```python
1. Load video frame
2. Run YOLO detection → Get bounding boxes
3. Calculate person centroid (center point)
4. Check if centroid is inside ROI polygon
5. Count people inside ROI
6. Display with color-coded overlays
```

**Color Coding:**
- 🟢 Green: Person inside ROI
- ⚪ Gray: Person outside ROI

### 2. Line Crossing Detection (Video 2)

```python
1. Capture RTSP frame → Store in buffer
2. Retrieve frame from buffer
3. Run YOLO detection → Get bounding boxes
4. Track each person frame-by-frame
5. Detect line crossing:
   - Moving DOWN through ENTER line → +1 count
   - Moving UP through EXIT line → -1 count
6. Stop tracking after crossing (prevent double-count)
7. Update persistent data
```

**Color Coding:**
- 🔵 Blue: Above ENTER line (outside)
- 🟢 Green: Below EXIT line (inside)
- 🟡 Yellow: Between lines (transitioning)

### 3. Tracking Algorithm

The system uses a **simplified tracking** approach:

1. **Frame-by-frame matching** - Match detections to existing tracks
2. **Distance-based** - Use Euclidean distance for matching
3. **Event-triggered** - Only track until crossing detected
4. **Automatic cleanup** - Stop tracking after crossing or timeout

This approach is:
- ✅ Simpler than full multi-object tracking
- ✅ More reliable for line crossing
- ✅ Lower computational overhead
- ✅ Prevents double-counting

---

## 📁 Project Structure

```
cctv-people-counting/
│
├── 📄 main.py                    # Main entry point
├── ⚙️ config.py                  # Configuration
├── 🤖 models.py                  # YOLO models
├── 🎬 processors.py              # Video processing
├── 🎯 trackers.py                # Line crossing tracker
├── 💾 buffer.py                  # Frame buffering
├── 💿 data_persistence.py        # Data storage
├── 📡 rtsp_handler.py            # RTSP handling
├── 🖥️ display.py                 # Visualization
├── ⏱️ timers.py                  # Performance tracking
├── 🔧 utils.py                   # Utilities
│
├── 📋 requirements.txt           # Dependencies
├── 📖 README.md                  # This file
├── 🏗️ ARCHITECTURE.md            # System architecture
├── 🔄 MIGRATION_GUIDE.md         # Migration guide
└── 📊 PROJECT_STRUCTURE.txt      # Project overview
```

---

## 🔧 Advanced Configuration

### GPU Acceleration

CUDA is automatically detected and used if available:

```python
# config.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

To force CPU mode:

```python
DEVICE = "cpu"
```

### Buffer Settings

Adjust frame buffer for RTSP streams:

```python
BUFFER_SIZE = 300              # Max frames (10 sec @ 30fps)
SAVE_BUFFER_TO_DISK = False   # True: disk, False: memory
BUFFER_DIR = './rtsp_buffer'   # Directory for disk mode
```

### Detection Tuning

Fine-tune detection parameters:

```python
CONF_THRESHOLD = 0.3           # Lower = more detections
IOU_THRESHOLD = 0.65           # Higher = stricter filtering
MIN_DETECTION_DISTANCE = 50    # Min pixels between detections
SMOOTH_WINDOW = 3              # Frames for position smoothing
```

### Tracking Tuning

Adjust line crossing behavior:

```python
CROSSING_THRESHOLD = 10        # Pixels past line to confirm
MAX_TRACKING_DISTANCE = 100    # Max pixels for track matching
TRACK_TIMEOUT = 1.5            # Seconds before track expires
```

### RTSP Settings

Configure RTSP connection:

```python
MAX_RECONNECT_ATTEMPTS = 5     # Retry attempts
RECONNECT_DELAY = 2            # Seconds between retries
```

---

## 📚 Documentation

- **[README.md](README.md)** - This file
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design patterns
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Guide for migrating from single file
- **[PROJECT_STRUCTURE.txt](PROJECT_STRUCTURE.txt)** - Visual project structure

---

## 🐛 Troubleshooting

### Common Issues

**1. RTSP Connection Failed**

```bash
✗ Failed to connect to RTSP stream
```

**Solutions:**
- Verify RTSP URL format: `rtsp://user:pass@ip:port/path`
- Check network connectivity
- Ensure camera is accessible
- Try increasing `MAX_RECONNECT_ATTEMPTS`

**2. Low FPS / Performance**

```bash
❌ May struggle with real-time
```

**Solutions:**
- Use GPU if available
- Lower `CONF_THRESHOLD` to reduce detections
- Reduce `BUFFER_SIZE`
- Process every Nth frame

**3. Inaccurate Counts**

**Solutions:**
- Adjust line positions (`LINE2_ENTER_Y`, `LINE2_EXIT_Y`)
- Increase `CROSSING_THRESHOLD`
- Fine-tune `MIN_DETECTION_DISTANCE`
- Verify ROI polygon coordinates
- Check lighting conditions

**4. Module Import Errors**

```bash
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**
```bash
pip install -r requirements.txt
# Or individually:
pip install opencv-python ultralytics torch numpy cvzone
```

**5. CUDA Out of Memory**

```bash
RuntimeError: CUDA out of memory
```

**Solutions:**
- Use smaller model
- Reduce input resolution
- Process fewer streams simultaneously
- Switch to CPU mode

---

## 🔮 Roadmap

### Planned Features

- [ ] **Multi-camera support** - Process 4+ cameras simultaneously
- [ ] **Web dashboard** - Browser-based monitoring
- [ ] **REST API** - External integration
- [ ] **Database integration** - PostgreSQL/MongoDB support
- [ ] **Heat maps** - Visualize traffic patterns
- [ ] **Alert system** - Email/SMS notifications
- [ ] **Cloud export** - AWS S3/Azure Blob integration
- [ ] **Advanced analytics** - Dwell time, peak hours
- [ ] **Mobile app** - iOS/Android monitoring
- [ ] **Docker support** - Containerized deployment

### Future Enhancements

- [ ] Person re-identification across cameras
- [ ] Age/gender estimation
- [ ] Social distancing monitoring
- [ ] Mask detection
- [ ] Queue length estimation
- [] Zone occupancy limits

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

### Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO26 framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyTorch](https://pytorch.org/) - Deep learning framework


---

<div align="center">

**[⬆ back to top](#-cctv-people-counting-system)**


</div>
