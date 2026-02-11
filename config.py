import numpy as np
import torch

# =========================
# DEVICE CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# VIDEO PATHS
# =========================
VIDEO1_PATH = 'Video from Nasya Putri (1).mp4'  # ROI Detection
VIDEO2_PATH = 'rtsp://admin:Ferbos2024!@192.168.68.169:554/Streaming/Channels/102?tcp'
MODEL_PATH = 'yolo26n.pt'

# =========================
# DETECTION PARAMETERS
# =========================
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.65
SMOOTH_WINDOW = 3
MIN_DETECTION_DISTANCE = 50  # pixels

# =========================
# RTSP SETTINGS
# =========================
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2  # seconds

# =========================
# BUFFER SETTINGS
# =========================
BUFFER_SIZE = 300  # Maximum frames to store (10 seconds at 30fps)
BUFFER_DIR = './rtsp_buffer'
SAVE_BUFFER_TO_DISK = False  # Set True to save to disk, False for memory
PROCESS_ALL_FRAMES = True  # Process every single frame

# =========================
# LINE CROSSING SETTINGS
# =========================
STARTING_COUNT = 12  # Initial people count
DATA_SAVE_FILE = 'line_crossing_data.json'
DATA_SAVE_INTERVAL = 60  # Save data every N seconds

# =========================
# VIDEO 1 - ROI CONFIG
# =========================
ROI1_POINTS = np.array([
    [396, 295],
    [717, 290],
    [740, 344],
    [451, 354],
    [442, 370],
    [445, 382],
    [447, 390],
    [447, 407],
    [445, 409],
    [438, 414],
    [434, 415],
    [428, 416],
    [414, 417],
    [391, 418],
    [381, 417]
], dtype=np.int32)

# =========================
# VIDEO 2 - LINE CROSSING CONFIG
# =========================
LINE2_X1, LINE2_X2 = 500, 760
LINE2_ENTER_Y = 340
LINE2_EXIT_Y = 390

# =========================
# CROSSING DETECTION SETTINGS
# =========================
CROSSING_THRESHOLD = 10  # Pixels past the line to confirm crossing
MAX_TRACKING_DISTANCE = 100  # Maximum distance to match person between frames
TRACK_TIMEOUT = 1.5  # Seconds to keep tracking if person disappears

# =========================
# DISPLAY SETTINGS
# =========================
VIDEO_RESIZE_WIDTH = 1020
VIDEO_RESIZE_HEIGHT = 600
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480