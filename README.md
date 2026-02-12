# 🎥 CCTV People Counting System


A production-ready, modular **people counting and tracking system** for CCTV footage analysis. This system processes video streams in real-time to count and track people using advanced computer vision techniques.

## 📖 Overview

This system provides two complementary counting methods:

1. **🔲 Region of Interest (ROI) Detection** - Monitors people within specific zones
2. **📏 Line Crossing Detection** - Tracks people crossing virtual boundaries with directional entry/exit counting



## 🎥 Demo Visualization

```
┌─────────────────────────┬─────────────────────────┐
│   Video 1: ROI Count    │  Video 2: Line Cross    │
│   [Zone monitoring]     │  [Entry/Exit tracking]  │
│                         │                         │
│   ┌───────────┐         │      ← EXIT LINE        │
│   │  ROI Zone │ 👤👤    │         👤👤            │
│   │    👤👤👤  │         │      ← ENTER LINE       │
│   └───────────┘         │                         │
│   Count: 5              │   Enter: 12 | Exit: 8   │
└─────────────────────────┴─────────────────────────┘
           Live Statistics & Analytics
```

## 📁 Project Structure

```
cctv-people-counting/
│
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings (paths, coordinates, etc.)
├── requirements.txt           # Python dependencies
│
├── preprocess.py              # Video preprocessing (resize/crop)
├── redefine_coordinates.py    # Interactive ROI/line coordinate definition
│
├── models.py                  # YOLO model loading and management
├── trackers.py                # Detection and tracking logic
├── processors.py              # ROI and line crossing processors
│
├── rtsp_handler.py            # RTSP stream handling
├── buffer.py                  # Video buffer management
│
├── display.py                 # Visualization and overlay functions
├── timers.py                  # Frame processing time measurement
├── data_persistence.py        # JSON data logging and statistics
│
└── model_export.py            # Model format conversion utility
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NasyaPutriRaudhah/CCTV-People-Counting
cd CCTV-People-Counting
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Step-by-Step Execution

#### **Step 1: Preprocess Video**
Resize and crop your video to the desired dimensions:

```bash
python preprocess.py
```

This will:
- Load your input video
- Allow you to resize/crop the frame
- Save preprocessing parameters to JSON

#### **Step 2: Define Coordinates**
Set up your ROI zones and crossing lines interactively:

```bash
python redefine_coordinates.py
```

This will:
- Display the video frame
- Allow you to draw ROI polygons
- Allow you to define crossing lines
- Save coordinates for use in main application

#### **Step 3: Update Configuration**
Edit `config.py` with the coordinates from Step 2:

```python
# Example configuration
ROI_COORDINATES = [(100, 200), (400, 200), (400, 500), (100, 500)]
LINE_COORDINATES = [(0, 300), (640, 300)]
```

**Don't forget to save the file!**

#### **Step 4: Run the Application**
Start the people counting system:

```bash
python main.py
```

## 📝 File Documentation

| File | Description |
|------|-------------|
| `main.py` | Main program entry point - orchestrates the entire counting pipeline |
| `config.py` | Central configuration file for paths, coordinates, and system settings |
| `preprocess.py` | Video preprocessing utility - resize/crop videos and save to JSON |
| `redefine_coordinates.py` | Interactive tool for defining ROI zones and crossing lines |
| `processors.py` | Core processors for ROI detection and line crossing logic |
| `models.py` | YOLO model loader and inference handler |
| `trackers.py` | Detection and tracking functions for people counting |
| `timers.py` | Performance monitoring - frame processing time measurement |
| `rtsp_handler.py` | RTSP stream connection and management |
| `buffer.py` | Video buffer for smooth frame loading from RTSP streams |
| `display.py` | Visualization functions for overlays, counters, and statistics |
| `data_persistence.py` | JSON data logging, statistics, and analytics persistence |
| `model_export.py` | Utility to convert YOLO models to different formats |

## Configuration

### Basic Settings (`config.py`)

```python
# Video source
VIDEO_PATH = "path/to/your/video.mp4"
# or for RTSP
RTSP_URL = "rtsp://username:password@ip:port/stream"

# Model settings
MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5

# ROI coordinates (polygon points)
ROI_COORDINATES = [
    (100, 200),
    (400, 200),
    (400, 500),
    (100, 500)
]

# Line crossing coordinates
LINE_START = (0, 300)
LINE_END = (640, 300)

# Output settings
OUTPUT_PATH = "output/"
SAVE_VIDEO = True
SAVE_STATISTICS = True
```

## 🔄 Model Export

Convert your YOLO model to different formats for deployment:

```bash
python model_export.py
```

Supported formats (see [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/)):
- ONNX
- TensorRT
- CoreML
- TFLite
- OpenVINO
- And more...

## Output Data

The system generates comprehensive analytics:

### JSON Output Structure
```json
{
  "session_id": "20240212_143022",
  "total_frames_processed": 1500,
  "average_inference_time": 0.045,
  "roi_count": 15,
  "line_crossing": {
    "entries": 28,
    "exits": 13,
    "current_occupancy": 15
  },
  "performance": {
    "fps": 22.3,
    "processing_time_per_frame": 0.045
  }
}
```

## Troubleshooting

### Common Issues

**Issue**: Low FPS / Slow processing
- **Solution**: Reduce video resolution in preprocessing
- **Solution**: Use a smaller YOLO model (e.g., `yolo11n.pt`)
- **Solution**: Enable GPU acceleration

**Issue**: Inaccurate counts
- **Solution**: Adjust confidence threshold in `config.py`
- **Solution**: Recalibrate ROI/line coordinates
- **Solution**: Ensure proper lighting in video

**Issue**: RTSP connection fails
- **Solution**: Verify network connectivity
- **Solution**: Check credentials and stream URL
- **Solution**: Increase buffer size in `rtsp_handler.py`


## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the tracking algorithm
- OpenCV community for computer vision tools

