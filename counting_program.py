import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import cvzone
import os
import time
from datetime import datetime
import threading
import queue
import json
import torch

# =========================
# DEVICE CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Capability:", torch.cuda.get_device_capability(0))
else:
    print("CUDA not available, running on CPU")



# =========================
# CONFIG
# =========================
VIDEO1_PATH = 'Video from Nasya Putri (1).mp4'  # ROI Detection
VIDEO2_PATH = 'rtsp://admin:Ferbos2024!@192.168.68.169:554/Streaming/Channels/102?tcp'
MODEL_PATH = 'yolo26n.pt'

CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.65
SMOOTH_WINDOW = 3

# Filter untuk deteksi yang terlalu berdekatan
MIN_DETECTION_DISTANCE = 50  # pixels

# RTSP reconnection settings
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 2  # seconds

# BUFFER SETTINGS
BUFFER_SIZE = 300  # Maximum frames to store in buffer (10 seconds at 30fps)
BUFFER_DIR = './rtsp_buffer'  # Directory to store buffered frames
SAVE_BUFFER_TO_DISK = False  # Set True to save frames to disk, False to keep in memory
PROCESS_ALL_FRAMES = True  # Process every single frame without skipping

# LINE CROSSING SETTINGS
STARTING_COUNT = 12  # Set your starting point here (number of people initially)
DATA_SAVE_FILE = 'line_crossing_data.json'  # File to save counting data
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

# Crossing detection settings
CROSSING_THRESHOLD = 10  # Pixels past the line to confirm crossing
MAX_TRACKING_DISTANCE = 100  # Maximum distance to match person between frames
TRACK_TIMEOUT = 1.5  # Seconds to keep tracking if person disappears

# =========================
# LOAD MODELS
# =========================
model1 = YOLO(MODEL_PATH).to(DEVICE)
model2 = YOLO(MODEL_PATH).to(DEVICE)

# =========================
# SIMPLIFIED LINE CROSSING TRACKER
# =========================
class SimplifiedLineCrossingTracker:
    """
    Simplified tracker - Only track for crossing detection
    
    Logic:
    1. Track person frame-by-frame to detect movement
    2. When crossing ENTER line (down): +1 → STOP TRACKING
    3. When crossing EXIT line (up): -1 → STOP TRACKING
    4. No "entered" or "exited" flags - just detect crossing and count
    """
    def __init__(self, enter_y, exit_y, threshold=CROSSING_THRESHOLD):
        self.enter_y = enter_y
        self.exit_y = exit_y
        self.threshold = threshold
        
        # Only track people temporarily for crossing detection
        self.active_tracks = {}  # {track_id: {cx, cy, last_y, last_seen}}
        self.next_id = 0
        
        # Recently crossed positions (to prevent double counting)
        self.recent_crossings = deque(maxlen=50)
    
    def update(self, detections):
        """
        Update tracker with new detections
        Returns: (entries, exits)
        """
        entries = 0
        exits = 0
        current_time = time.time()
        
        # Match detections with active tracks
        matched_ids = set()
        new_tracks = {}
        
        for det in detections:
            cx, cy = det['cx'], det['cy']
            
            # Find best matching track
            best_match_id = None
            best_match_dist = MAX_TRACKING_DISTANCE
            
            for track_id, track in self.active_tracks.items():
                dist = np.sqrt((cx - track['cx'])**2 + (cy - track['cy'])**2)
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_id = track_id
            
            if best_match_id is not None:
                # Matched with existing track - check for crossing
                track = self.active_tracks[best_match_id]
                prev_y = track['cy']
                
                # ═══════════════════════════════════════════════════════
                # ENTER CROSSING: Moving DOWN, crossing ENTER line
                # ═══════════════════════════════════════════════════════
                if (prev_y > self.enter_y and cy < self.enter_y):
                    # Crossed ENTER line going DOWN
                    if not self._is_recent_crossing(cx, cy, 'entry'):
                        entries += 1
                        self._add_crossing(cx, cy, 'entry', current_time)
                        print(f"✓ [ENTRY +1] Person crossed ENTER (y: {prev_y:.0f}→{cy:.0f})")
                        # STOP TRACKING - don't add to new_tracks
                        matched_ids.add(best_match_id)
                        continue  # Skip adding to new_tracks
                
                # ═══════════════════════════════════════════════════════
                # EXIT CROSSING: Moving UP, crossing EXIT line
                # ═══════════════════════════════════════════════════════
                elif (prev_y < self.exit_y and cy > self.exit_y):
                    # Crossed EXIT line going UP
                    if not self._is_recent_crossing(cx, cy, 'exit'):
                        exits += 1
                        self._add_crossing(cx, cy, 'exit', current_time)
                        print(f"✓ [EXIT -1] Person crossed EXIT (y: {prev_y:.0f}→{cy:.0f})")
                        # STOP TRACKING - don't add to new_tracks
                        matched_ids.add(best_match_id)
                        continue  # Skip adding to new_tracks
                
                # Still tracking (no crossing detected) - update position
                new_tracks[best_match_id] = {
                    'cx': cx,
                    'cy': cy,
                    'last_seen': current_time
                }
                matched_ids.add(best_match_id)
            
            else:
                # New detection - start tracking
                new_id = self.next_id
                self.next_id += 1
                
                new_tracks[new_id] = {
                    'cx': cx,
                    'cy': cy,
                    'last_seen': current_time
                }
        
        # Keep tracks that were not matched but seen recently
        for track_id, track in self.active_tracks.items():
            if track_id not in matched_ids:
                if current_time - track['last_seen'] < TRACK_TIMEOUT:
                    new_tracks[track_id] = track
                # else: timeout - stop tracking
        
        self.active_tracks = new_tracks
        
        return entries, exits
    
    def _is_recent_crossing(self, cx, cy, crossing_type):
        """Check if this position recently had a crossing (prevent double count)"""
        current_time = time.time()
        for crossing in self.recent_crossings:
            if crossing['type'] == crossing_type:
                dist = np.sqrt((cx - crossing['cx'])**2 + (cy - crossing['cy'])**2)
                time_diff = current_time - crossing['time']
                # If same position within 1 second, it's a duplicate
                if dist < 60 and time_diff < 1.0:
                    return True
        return False
    
    def _add_crossing(self, cx, cy, crossing_type, current_time):
        """Record a crossing event"""
        self.recent_crossings.append({
            'cx': cx,
            'cy': cy,
            'type': crossing_type,
            'time': current_time
        })
    
    def get_status_info(self):
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.active_tracks)
        }

# =========================
# DATA PERSISTENCE CLASS
# =========================
class DataPersistence:
    """Handle saving and loading of counting data"""
    def __init__(self, filename=DATA_SAVE_FILE):
        self.filename = filename
        self.data = {
            'starting_count': STARTING_COUNT,
            'current_count': STARTING_COUNT,
            'total_entries': 0,
            'total_exits': 0,
            'session_start': None,
            'last_update': None,
            'history': []  # Log of all crossing events
        }
        self.load_data()
    
    def load_data(self):
        """Load existing data if available"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    loaded_data = json.load(f)
                    # Keep the loaded data but update starting count if changed in config
                    if loaded_data.get('starting_count') == STARTING_COUNT:
                        self.data = loaded_data
                        print(f"✓ Loaded existing data from {self.filename}")
                        print(f"  Current count: {self.data['current_count']}")
                        print(f"  Total entries: {self.data['total_entries']}")
                        print(f"  Total exits: {self.data['total_exits']}")
                    else:
                        print(f"⚠ Starting count changed, resetting data")
                        self.data['starting_count'] = STARTING_COUNT
                        self.data['current_count'] = STARTING_COUNT
                        self.save_data()
            except Exception as e:
                print(f"⚠ Error loading data: {e}")
                self.save_data()
        else:
            print(f"✓ Starting fresh with count: {STARTING_COUNT}")
            self.data['session_start'] = datetime.now().isoformat()
            self.save_data()
    
    def save_data(self):
        """Save current data to file"""
        self.data['last_update'] = datetime.now().isoformat()
        try:
            with open(self.filename, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving data: {e}")
    
    def add_entries(self, count):
        """Add entry events"""
        self.data['current_count'] += count
        self.data['total_entries'] += count
        
        # Log events
        for _ in range(count):
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'entry',
                'delta': 1,
                'count_after': self.data['current_count']
            }
            self.data['history'].append(event)
        
        # Keep only last 1000 events
        if len(self.data['history']) > 1000:
            self.data['history'] = self.data['history'][-1000:]
    
    def add_exits(self, count):
        """Add exit events"""
        self.data['current_count'] -= count
        self.data['total_exits'] += count
        
        # Log events
        for _ in range(count):
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'exit',
                'delta': -1,
                'count_after': self.data['current_count']
            }
            self.data['history'].append(event)
        
        # Keep only last 1000 events
        if len(self.data['history']) > 1000:
            self.data['history'] = self.data['history'][-1000:]
    
    def get_current_count(self):
        """Get current count"""
        return self.data['current_count']
    
    def get_summary(self):
        """Get summary statistics"""
        return {
            'starting_count': self.data['starting_count'],
            'current_count': self.data['current_count'],
            'total_entries': self.data['total_entries'],
            'total_exits': self.data['total_exits'],
            'net_change': self.data['current_count'] - self.data['starting_count'],
            'session_start': self.data['session_start'],
            'last_update': self.data['last_update']
        }

# =========================
# TIMING TRACKER
# =========================
class InferenceTimer:
    """Track inference timing statistics"""
    def __init__(self, name=""):
        self.name = name
        self.times = []
        self.total_frames = 0
        self.total_time = 0
        
    def add_time(self, inference_ms):
        """Add inference time in milliseconds"""
        self.times.append(inference_ms)
        self.total_frames += 1
        self.total_time += inference_ms
    
    def get_statistics(self):
        """Get complete statistics"""
        if len(self.times) == 0:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'p95': 0,
                'p99': 0,
                'max_fps': 0
            }
        
        times_array = np.array(self.times)
        mean_time = np.mean(times_array)
        
        return {
            'count': len(self.times),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'mean': mean_time,
            'median': np.median(times_array),
            'std': np.std(times_array),
            'p95': np.percentile(times_array, 95),
            'p99': np.percentile(times_array, 99),
            'max_fps': 1000 / mean_time if mean_time > 0 else 0
        }

# =========================
# BUFFER MANAGEMENT
# =========================
class FrameBuffer:
    """Buffer to store frames from RTSP before processing"""
    def __init__(self, max_size=BUFFER_SIZE, save_to_disk=False, buffer_dir=BUFFER_DIR):
        self.max_size = max_size
        self.save_to_disk = save_to_disk
        self.buffer_dir = buffer_dir
        self.buffer = deque(maxlen=max_size)
        self.frame_count = 0
        self.buffer_full_warned = False
        
        # Create buffer directory if saving to disk
        if self.save_to_disk and not os.path.exists(self.buffer_dir):
            os.makedirs(self.buffer_dir)
            print(f"✓ Created buffer directory: {self.buffer_dir}")
    
    def add_frame(self, frame):
        """Add frame to buffer"""
        if self.save_to_disk:
            # Save to disk
            filename = os.path.join(self.buffer_dir, f"frame_{self.frame_count:06d}.jpg")
            cv2.imwrite(filename, frame)
            self.buffer.append(filename)
        else:
            # Keep in memory
            self.buffer.append(frame.copy())
        
        self.frame_count += 1
        
        # Warn if buffer is getting full
        if len(self.buffer) >= self.max_size * 0.9 and not self.buffer_full_warned:
            print(f"⚠ Warning: Buffer is 90% full ({len(self.buffer)}/{self.max_size})")
            self.buffer_full_warned = True
    
    def get_frame(self):
        """Get frame from buffer (FIFO)"""
        if len(self.buffer) > 0:
            frame_data = self.buffer.popleft()
            
            if self.save_to_disk:
                # Read from disk
                frame = cv2.imread(frame_data)
                # Delete file after reading
                try:
                    os.remove(frame_data)
                except:
                    pass
                return frame
            else:
                # Return from memory
                return frame_data
        return None
    
    def size(self):
        """Get current buffer size"""
        return len(self.buffer)
    
    def is_empty(self):
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def clear(self):
        """Clear all buffered frames"""
        if self.save_to_disk:
            # Delete all files
            for filename in self.buffer:
                try:
                    os.remove(filename)
                except:
                    pass
        self.buffer.clear()
        self.frame_count = 0
        self.buffer_full_warned = False

# =========================
# RTSP CONNECTION
# =========================
def connect_rtsp(url, max_attempts=MAX_RECONNECT_ATTEMPTS):
    """Connect to RTSP stream with retry logic"""
    for attempt in range(max_attempts):
        print(f"Connecting to RTSP (attempt {attempt + 1}/{max_attempts})...")
        
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✓ RTSP connected successfully!")
                return cap
            else:
                cap.release()
        
        if attempt < max_attempts - 1:
            print(f"Connection failed, retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)
    
    print("✗ Failed to connect to RTSP stream")
    return None

# =========================
# HELPER FUNCTIONS
# =========================
def point_in_roi(point, roi):
    return cv2.pointPolygonTest(roi, point, False) >= 0

def filter_close_detections(ids, boxes, confs, min_distance=MIN_DETECTION_DISTANCE):
    """Filter deteksi yang terlalu berdekatan"""
    if len(ids) == 0:
        return ids, boxes, confs
    
    centroids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append([cx, cy])
    centroids = np.array(centroids)
    
    sorted_indices = np.argsort(confs)[::-1]
    keep_indices = []
    
    for i in sorted_indices:
        too_close = False
        for kept_idx in keep_indices:
            distance = np.sqrt(
                (centroids[i][0] - centroids[kept_idx][0])**2 + 
                (centroids[i][1] - centroids[kept_idx][1])**2
            )
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            keep_indices.append(i)
    
    keep_indices = sorted(keep_indices)
    return ids[keep_indices], boxes[keep_indices], confs[keep_indices]

# =========================
# PROCESSING FUNCTIONS
# =========================
def process_video1_roi(frame, id_history_v1, timer):
    """Process Video 1 with ROI detection"""
    frame = cv2.resize(frame, (1020, 600))
    
    # Draw ROI
    overlay = frame.copy()
    cv2.polylines(overlay, [ROI1_POINTS], True, (0, 255, 0), 2)
    cv2.fillPoly(overlay, [ROI1_POINTS], (0, 255, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # YOLO Detection with timing
    inference_start = time.time()
    
    results = model1.track(
        frame,
        device=DEVICE,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    inference_time = (time.time() - inference_start) * 1000  # Convert to ms
    timer.add_time(inference_time)
    
    current_in_roi = 0
    
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        ids, boxes, confs = filter_close_detections(ids, boxes, confs)
        
        for track_id, box, conf in zip(ids, boxes, confs):
            x1, y1, x2, y2 = box
            h = y2 - y1
            head_y2 = y1 + int(h * 0.6)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + head_y2) / 2)
            
            # Smoothing
            if track_id not in id_history_v1:
                id_history_v1[track_id] = deque(maxlen=SMOOTH_WINDOW)
            id_history_v1[track_id].append(cy)
            cy_smooth = int(np.mean(id_history_v1[track_id]))
            
            inside_roi = point_in_roi((cx, cy_smooth), ROI1_POINTS)
            
            if inside_roi:
                current_in_roi += 1
                color = (0, 255, 0)
            else:
                color = (128, 128, 128)
            
            cv2.rectangle(frame, (x1, y1), (x2, head_y2), color, 2)
            cvzone.putTextRect(frame, f'ID {track_id} ({conf:.2f})', (x1, y1-10),
                             scale=0.5, thickness=1, colorR=color)
            cv2.circle(frame, (cx, cy_smooth), 5, color, -1)
    
    return frame, current_in_roi

def process_video2_line(frame, timer, line_tracker, data_persistence):
    """
    Process Video 2 with simplified line crossing detection
    Track only for crossing detection, not for status tracking
    """
    frame = cv2.resize(frame, (1020, 600))
    
    # Draw crossing lines
    cv2.line(frame, (LINE2_X1, LINE2_EXIT_Y), (LINE2_X2, LINE2_EXIT_Y), (0, 0, 255), 3)
    cv2.putText(frame, "EXIT LINE (cross UP = -1)", (LINE2_X1, LINE2_EXIT_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.line(frame, (LINE2_X1, LINE2_ENTER_Y), (LINE2_X2, LINE2_ENTER_Y), (255, 0, 0), 3)
    cv2.putText(frame, "ENTER LINE (cross DOWN = +1)", (LINE2_X1, LINE2_ENTER_Y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # YOLO Detection with timing
    inference_start = time.time()
    
    results = model2(
        frame,
        device = DEVICE,
        classes=[0],
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )
    
    inference_time = (time.time() - inference_start) * 1000  # Convert to ms
    timer.add_time(inference_time)
    
    # Get detections
    current_detections = []
    
    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        
        dummy_ids = np.arange(len(boxes))
        dummy_ids, boxes, confs = filter_close_detections(dummy_ids, boxes, confs)
        
        for idx, (box, conf) in enumerate(zip(boxes, confs)):
            x1, y1, x2, y2 = box
            h = y2 - y1
            head_y2 = y1 + int(h * 0.5)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + head_y2) / 2)
            
            current_detections.append({
                'cx': cx,
                'cy': cy,
                'box': (x1, y1, x2, head_y2),
                'conf': conf
            })
            
            # Simple color coding based on position
            if cy < LINE2_ENTER_Y:
                color = (255, 0, 0)  # Blue - Above ENTER
                status = "OUT"
            elif cy > LINE2_EXIT_Y:
                color = (0, 255, 0)  # Green - Below EXIT
                status = "IN"
            else:
                color = (0, 255, 255)  # Yellow - Between lines
                status = "MID"
            
            cv2.rectangle(frame, (x1, y1), (x2, head_y2), color, 2)
            cvzone.putTextRect(frame, f'{status} ({conf:.2f})', (x1, y1-10),
                             scale=0.5, thickness=1, colorR=color)
            cv2.circle(frame, (cx, cy), 5, color, -1)
    
    # Update line tracker and get crossing events
    entries, exits = line_tracker.update(current_detections)
    
    # Update persistent data
    if entries > 0:
        data_persistence.add_entries(entries)
        print(f">>> ENTRY: +{entries}, Count: {data_persistence.get_current_count()}")
    
    if exits > 0:
        data_persistence.add_exits(exits)
        print(f">>> EXIT: -{exits}, Count: {data_persistence.get_current_count()}")
    
    # Get tracker status
    tracker_status = line_tracker.get_status_info()
    
    # Display info on frame
    cv2.putText(frame, f"Active tracks: {tracker_status['active_tracks']}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame, data_persistence.get_current_count()

def create_combined_view(frame1, frame2, count_v1, count_v2, buffer_size, 
                        fps_capture, fps_process, data_summary):
    """Combine both videos with statistics"""
    frame1_display = cv2.resize(frame1, (640, 480))
    frame2_display = cv2.resize(frame2, (640, 480))
    
    combined_view = np.hstack([frame1_display, frame2_display])
    
    # Statistics overlay
    overlay = combined_view.copy()
    stats_box_height = 300
    stats_box_width = 520
    
    cv2.rectangle(overlay, (10, 10), (stats_box_width, stats_box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, combined_view, 0.25, 0, combined_view)
    cv2.rectangle(combined_view, (10, 10), (stats_box_width, stats_box_height), (255, 255, 255), 2)
    
    y_offset = 35
    line_height = 30
    
    cv2.putText(combined_view, 'STATISTICS', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_offset += line_height
    cv2.putText(combined_view, f'Video 1 (ROI): {count_v1}', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
    # Line crossing details
    y_offset += line_height + 5
    cv2.putText(combined_view, 'VIDEO 2 - SIMPLIFIED TRACKING:', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    
    y_offset += line_height
    cv2.putText(combined_view, f'Current: {count_v2}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset += line_height - 5
    cv2.putText(combined_view, f'  Start: {data_summary["starting_count"]}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += line_height - 10
    cv2.putText(combined_view, f'  +Entry: {data_summary["total_entries"]}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    y_offset += line_height - 10
    cv2.putText(combined_view, f'  -Exit: {data_summary["total_exits"]}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    y_offset += line_height - 10
    net_change = data_summary["net_change"]
    net_color = (0, 255, 0) if net_change >= 0 else (0, 0, 255)
    cv2.putText(combined_view, f'  Net: {net_change:+d}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, net_color, 1)
    
    combined = count_v1 + count_v2
    y_offset += line_height
    cv2.putText(combined_view, f'TOTAL: {combined}', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Note
    y_offset += line_height
    cv2.putText(combined_view, 'Track only for crossing, stop after crossing', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    # FPS info
    y_offset += line_height - 5
    cv2.putText(combined_view, f'Buffer: {buffer_size} | Cap: {fps_capture:.1f} | Proc: {fps_process:.1f} fps', 
                (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return combined_view

def print_final_report(timer_v1, timer_v2, total_runtime, data_persistence):
    """Print comprehensive final report"""
    stats_v1 = timer_v1.get_statistics()
    stats_v2 = timer_v2.get_statistics()
    summary = data_persistence.get_summary()
    
    print("\n" + "="*80)
    print("FINAL REPORT - SIMPLIFIED TRACKING SYSTEM")
    print("="*80)
    
    print(f"\n⏱️  SESSION INFORMATION")
    print(f"   Started:      {summary['session_start']}")
    print(f"   Ended:        {summary['last_update']}")
    print(f"   Runtime:      {total_runtime:.2f} seconds ({total_runtime/3600:.2f} hours)")
    
    print("\n" + "-"*80)
    print("👥 LINE CROSSING SUMMARY")
    print("-"*80)
    print(f"   Starting count:       {summary['starting_count']} people")
    print(f"   Current count:        {summary['current_count']} people")
    print(f"   Total entries:        +{summary['total_entries']} crossings")
    print(f"   Total exits:          -{summary['total_exits']} crossings")
    print(f"   Net change:           {summary['net_change']:+d} people")
    
    print("\n" + "-"*80)
    print("📊 INFERENCE PERFORMANCE")
    print("-"*80)
    print(f"Video 1: {stats_v1['count']} frames, {stats_v1['mean']:.2f} ms avg")
    print(f"Video 2: {stats_v2['count']} frames, {stats_v2['mean']:.2f} ms avg")
    
    avg_inference = (stats_v1['mean'] + stats_v2['mean']) / 2
    print(f"\nAverage: {avg_inference:.2f} ms/frame")
    
    if avg_inference < 33.33:
        print(f"✅ Can handle real-time 30 FPS")
    elif avg_inference < 66.67:
        print(f"⚠️  Can handle ~15 FPS")
    else:
        print(f"❌ May struggle with real-time")
    
    print("\n" + "="*80)
    print(f"Data saved to: {DATA_SAVE_FILE}")
    print("="*80 + "\n")

# =========================
# RTSP CAPTURE THREAD
# =========================
class RTSPCaptureThread(threading.Thread):
    """Thread to continuously capture RTSP frames and store in buffer"""
    def __init__(self, rtsp_url, frame_buffer):
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.frame_buffer = frame_buffer
        self.stopped = False
        self.daemon = True
        self.fps = 0
        
    def run(self):
        print("Starting RTSP capture thread...")
        
        cap = connect_rtsp(self.rtsp_url)
        if cap is None:
            print("Failed to start RTSP capture")
            return
        
        consecutive_errors = 0
        max_errors = 20
        
        # FPS tracking
        frame_count = 0
        start_time = time.time()
        
        while not self.stopped:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                consecutive_errors = 0
                
                # Add to buffer
                self.frame_buffer.add_frame(frame)
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
                
            else:
                consecutive_errors += 1
                
                if consecutive_errors >= max_errors:
                    print(f"\nRTSP: Too many errors, reconnecting...")
                    cap.release()
                    cap = connect_rtsp(self.rtsp_url)
                    consecutive_errors = 0
                    
                    if cap is None:
                        print("RTSP: Reconnection failed")
                        break
                
                time.sleep(0.01)
        
        cap.release()
        print("RTSP capture thread stopped")
    
    def stop(self):
        self.stopped = True

# =========================
# MAIN FUNCTION
# =========================
def main():
    print("\n" + "="*70)
    print("SIMPLIFIED LINE CROSSING SYSTEM")
    print("="*70)
    print(f"Starting count: {STARTING_COUNT} people")
    print(f"Data file: {DATA_SAVE_FILE}")
    print("\n🔹 SIMPLIFIED TRACKING LOGIC:")
    print("  • Track person frame-by-frame ONLY for crossing detection")
    print("  • Cross ENTER line (down) → +1 → STOP tracking")
    print("  • Cross EXIT line (up) → -1 → STOP tracking")
    print("  • No status flags, just detect crossing and count")
    print("\nPress ESC to exit")
    print("="*70 + "\n")
    
    # Initialize
    data_persistence = DataPersistence()
    line_tracker = SimplifiedLineCrossingTracker(LINE2_ENTER_Y, LINE2_EXIT_Y)
    print("✓ Simplified tracker initialized")
    
    cap1 = cv2.VideoCapture(VIDEO1_PATH)
    assert cap1.isOpened(), "Video 1 tidak bisa dibuka"
    print("✓ Video 1 loaded")
    
    frame_buffer_v2 = FrameBuffer(
        max_size=BUFFER_SIZE,
        save_to_disk=SAVE_BUFFER_TO_DISK,
        buffer_dir=BUFFER_DIR
    )
    print(f"✓ Frame buffer created")
    
    timer_v1 = InferenceTimer(name="Video 1")
    timer_v2 = InferenceTimer(name="Video 2")
    
    rtsp_thread = RTSPCaptureThread(VIDEO2_PATH, frame_buffer_v2)
    rtsp_thread.start()
    
    print("\nFilling buffer...")
    time.sleep(3)
    print(f"✓ Buffer ready with {frame_buffer_v2.size()} frames\n")
    print("Processing started...\n")
    
    # State tracking
    id_history_v1 = {}
    count_v1 = 0
    count_v2 = data_persistence.get_current_count()
    
    process_frame_count = 0
    process_start_time = time.time()
    fps_process = 0
    
    runtime_start = time.time()
    last_save_time = time.time()
    
    cv2.namedWindow("Simplified Line Crossing System", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Process Video 1
            ret1, frame1 = cap1.read()
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            processed_frame1, c1 = process_video1_roi(frame1, id_history_v1, timer_v1)
            count_v1 = c1
            
            # Process Video 2
            if not frame_buffer_v2.is_empty():
                frame2 = frame_buffer_v2.get_frame()
                
                if frame2 is not None:
                    processed_frame2, count_v2 = process_video2_line(
                        frame2, timer_v2, line_tracker, data_persistence
                    )
                else:
                    processed_frame2 = np.zeros((600, 1020, 3), dtype=np.uint8)
                    cv2.putText(processed_frame2, "Buffer read error", (300, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                processed_frame2 = np.zeros((600, 1020, 3), dtype=np.uint8)
                cv2.putText(processed_frame2, "Waiting for frames...", (300, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Calculate FPS
            process_frame_count += 1
            elapsed = time.time() - process_start_time
            if elapsed >= 1.0:
                fps_process = process_frame_count / elapsed
                process_frame_count = 0
                process_start_time = time.time()
            
            # Auto-save
            current_time = time.time()
            if current_time - last_save_time >= DATA_SAVE_INTERVAL:
                data_persistence.save_data()
                last_save_time = current_time
            
            # Display
            data_summary = data_persistence.get_summary()
            combined_view = create_combined_view(
                processed_frame1, processed_frame2,
                count_v1, count_v2,
                frame_buffer_v2.size(),
                rtsp_thread.fps,
                fps_process,
                data_summary
            )
            
            cv2.imshow("Simplified Line Crossing System", combined_view)
            
            # Exit on ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\nExiting...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        total_runtime = time.time() - runtime_start
        
        print("\nCleaning up...")
        data_persistence.save_data()
        print(f"✓ Data saved")
        
        rtsp_thread.stop()
        rtsp_thread.join(timeout=2)
        cap1.release()
        frame_buffer_v2.clear()
        cv2.destroyAllWindows()
        
        print_final_report(timer_v1, timer_v2, total_runtime, data_persistence)
        
        print("Processing completed!")

if __name__ == "__main__":
    main()