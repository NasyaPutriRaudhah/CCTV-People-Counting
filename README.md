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
│   │  ROI Zone │         │         👤👤           │
│   │ 👤👤👤  │          |   ← ENTER LINE          │
│   └───────────┘         │                         │
│   Count: 3              │   Enter: 12 | Exit: 8   │
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
#Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Defining path
VIDEO1_PATH = '"path/to/your/video.mp4'  
VIDEO2_PATH = "rtsp://username:password@ip:port/stream"
MODEL_PATH = 'yolo26n.pt'

#Detection parameters
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.65
SMOOTH_WINDOW = 3
MIN_DETECTION_DISTANCE = 50  # pixels

#RTSP Reconnection settings 
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2  # seconds

#Buffer settings
BUFFER_SIZE = 300  # Maximum frames to store (10 seconds at 30fps)
BUFFER_DIR = './rtsp_buffer'
SAVE_BUFFER_TO_DISK = False  # Set True to save to disk, False for memory
PROCESS_ALL_FRAMES = True  # Process every single frame

#Line crossing data persistence
STARTING_COUNT = 12  # Initial people count
DATA_SAVE_FILE = 'line_crossing_data.json'
DATA_SAVE_INTERVAL = 60  # Save data every N seconds

#Video 
ROI1_POINTS = np.array([
    [41, 40],
    [804, 25],
    [960, 130],
    [953, 328],
    [10, 315],
], dtype=np.int32)

#Video 2 line coordinates
LINE2_X1, LINE2_X2 = 303, 455
LINE2_ENTER_Y = 90
LINE2_EXIT_Y = 140

#Crossing tracker settings
CROSSING_THRESHOLD = 10  # Pixels past the line to confirm crossing
MAX_TRACKING_DISTANCE = 100  # Maximum distance to match person between frames
TRACK_TIMEOUT = 1.5  # Seconds to keep tracking if person disappears

#Display settings
VIDEO_RESIZE_WIDTH = 1020
VIDEO_RESIZE_HEIGHT = 600
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

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

