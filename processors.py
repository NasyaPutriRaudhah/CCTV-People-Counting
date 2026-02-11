"""
Video Processing Functions
"""
import cv2
import time
import numpy as np
from collections import deque
import cvzone
from config import (
    DEVICE, CONF_THRESHOLD, IOU_THRESHOLD, SMOOTH_WINDOW,
    ROI1_POINTS, LINE2_X1, LINE2_X2, LINE2_ENTER_Y, LINE2_EXIT_Y,
    VIDEO_RESIZE_WIDTH, VIDEO_RESIZE_HEIGHT
)
from utils import point_in_roi, filter_close_detections


def process_video1_roi(frame, model, id_history, timer):
    """
    Process Video 1 with ROI detection
    
    Args:
        frame: Input frame
        model: YOLO model
        id_history: Dictionary to store tracking history
        timer: InferenceTimer object
    
    Returns:
        (processed_frame, count_in_roi)
    """
    frame = cv2.resize(frame, (VIDEO_RESIZE_WIDTH, VIDEO_RESIZE_HEIGHT))
    
    # Draw ROI
    overlay = frame.copy()
    cv2.polylines(overlay, [ROI1_POINTS], True, (0, 255, 0), 2)
    cv2.fillPoly(overlay, [ROI1_POINTS], (0, 255, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # YOLO Detection with timing
    inference_start = time.time()
    
    results = model.track(
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
            if track_id not in id_history:
                id_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
            id_history[track_id].append(cy)
            cy_smooth = int(np.mean(id_history[track_id]))
            
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


def process_video2_line(frame, model, timer, line_tracker, data_persistence):
    """
    Process Video 2 with simplified line crossing detection
    Track only for crossing detection, not for status tracking
    
    Args:
        frame: Input frame
        model: YOLO model
        timer: InferenceTimer object
        line_tracker: SimplifiedLineCrossingTracker object
        data_persistence: DataPersistence object
    
    Returns:
        (processed_frame, current_count)
    """
    frame = cv2.resize(frame, (VIDEO_RESIZE_WIDTH, VIDEO_RESIZE_HEIGHT))
    
    # Draw crossing lines
    cv2.line(frame, (LINE2_X1, LINE2_EXIT_Y), (LINE2_X2, LINE2_EXIT_Y), (0, 0, 255), 3)
    cv2.putText(frame, "EXIT LINE (cross UP = -1)", (LINE2_X1, LINE2_EXIT_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.line(frame, (LINE2_X1, LINE2_ENTER_Y), (LINE2_X2, LINE2_ENTER_Y), (255, 0, 0), 3)
    cv2.putText(frame, "ENTER LINE (cross DOWN = +1)", (LINE2_X1, LINE2_ENTER_Y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # YOLO Detection with timing
    inference_start = time.time()
    
    results = model(
        frame,
        device=DEVICE,
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