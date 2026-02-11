import cv2
import numpy as np
from config import MIN_DETECTION_DISTANCE


def point_in_roi(point, roi):
    return cv2.pointPolygonTest(roi, point, False) >= 0


def filter_close_detections(ids, boxes, confs, min_distance=MIN_DETECTION_DISTANCE):
    if len(ids) == 0:
        return ids, boxes, confs
    
    # Calculate centroids
    centroids = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centroids.append([cx, cy])
    centroids = np.array(centroids)
    
    # Sort by confidence (highest first)
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


def print_device_info(device):
    print(f"Using device: {device}")
    
    if device == "cuda":
        import torch
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Capability:", torch.cuda.get_device_capability(0))
    else:
        print("CUDA not available, running on CPU")