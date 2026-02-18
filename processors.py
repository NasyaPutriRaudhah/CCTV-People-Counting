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


def process_video1_roi(frame, model, id_history, timer, roi_points=None):
    # Use provided ROI points or default from config
    if roi_points is None:
        roi_points = ROI1_POINTS
    
    # Store original frame size before resize
    orig_h, orig_w = frame.shape[:2]
    
    # Resize frame
    frame = cv2.resize(frame, (VIDEO_RESIZE_WIDTH, VIDEO_RESIZE_HEIGHT))
    
    # Scale ROI points to match resized frame
    scale_x = VIDEO_RESIZE_WIDTH / orig_w
    scale_y = VIDEO_RESIZE_HEIGHT / orig_h
    
    roi_points_scaled = roi_points.copy()
    roi_points_scaled[:, 0] = (roi_points[:, 0] * scale_x).astype(np.int32)
    roi_points_scaled[:, 1] = (roi_points[:, 1] * scale_y).astype(np.int32)
    
    # Draw ROI
    overlay = frame.copy()
    cv2.polylines(overlay, [roi_points_scaled], True, (0, 255, 0), 2)
    cv2.fillPoly(overlay, [roi_points_scaled], (0, 255, 0))
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
            
            inside_roi = point_in_roi((cx, cy_smooth), roi_points_scaled)
            
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


def process_video2_line(frame, model, timer, line_tracker, data_persistence, 
                        line_x1=None, line_x2=None, line_enter_y=None, line_exit_y=None):
    # Use provided line coordinates or default from config
    if line_x1 is None:
        line_x1 = LINE2_X1
    if line_x2 is None:
        line_x2 = LINE2_X2
    if line_enter_y is None:
        line_enter_y = LINE2_ENTER_Y
    if line_exit_y is None:
        line_exit_y = LINE2_EXIT_Y
    
    # Store original frame size before resize
    orig_h, orig_w = frame.shape[:2]
    
    # Resize frame
    frame = cv2.resize(frame, (VIDEO_RESIZE_WIDTH, VIDEO_RESIZE_HEIGHT))
    
    # Scale line coordinates to match resized frame
    scale_x = VIDEO_RESIZE_WIDTH / orig_w
    scale_y = VIDEO_RESIZE_HEIGHT / orig_h
    
    line_x1_scaled = int(line_x1 * scale_x)
    line_x2_scaled = int(line_x2 * scale_x)
    line_enter_y_scaled = int(line_enter_y * scale_y)
    line_exit_y_scaled = int(line_exit_y * scale_y)
    
    # Clamp values to frame boundaries to prevent drawing errors
    line_x1_scaled = max(0, min(line_x1_scaled, VIDEO_RESIZE_WIDTH - 1))
    line_x2_scaled = max(0, min(line_x2_scaled, VIDEO_RESIZE_WIDTH - 1))
    line_enter_y_scaled = max(0, min(line_enter_y_scaled, VIDEO_RESIZE_HEIGHT - 1))
    line_exit_y_scaled = max(0, min(line_exit_y_scaled, VIDEO_RESIZE_HEIGHT - 1))
    
    # Draw crossing lines with scaled coordinates
    cv2.line(frame, (line_x1_scaled, line_exit_y_scaled), (line_x2_scaled, line_exit_y_scaled), (0, 0, 255), 3)
    cv2.putText(frame, "EXIT LINE (cross UP = -1)", (line_x1_scaled, max(15, line_exit_y_scaled - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.line(frame, (line_x1_scaled, line_enter_y_scaled), (line_x2_scaled, line_enter_y_scaled), (255, 0, 0), 3)
    cv2.putText(frame, "ENTER LINE (cross DOWN = +1)", (line_x1_scaled, min(VIDEO_RESIZE_HEIGHT - 10, line_enter_y_scaled + 25)),
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
            
            # Simple color coding based on position (use scaled coordinates)
            if cy < line_enter_y_scaled:
                color = (255, 0, 0)  # Blue - Above ENTER
                status = "OUT"
            elif cy > line_exit_y_scaled:
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
