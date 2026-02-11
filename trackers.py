"""
Line Crossing Tracker
"""
import time
import numpy as np
from collections import deque
from config import CROSSING_THRESHOLD, MAX_TRACKING_DISTANCE, TRACK_TIMEOUT


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
                
                # ENTER CROSSING: Moving DOWN, crossing ENTER line
                if (prev_y > self.enter_y and cy < self.enter_y):
                    # Crossed ENTER line going DOWN
                    if not self._is_recent_crossing(cx, cy, 'entry'):
                        entries += 1
                        self._add_crossing(cx, cy, 'entry', current_time)
                        print(f"✓ [ENTRY +1] Person crossed ENTER (y: {prev_y:.0f}→{cy:.0f})")
                        # STOP TRACKING - don't add to new_tracks
                        matched_ids.add(best_match_id)
                        continue  # Skip adding to new_tracks
                
                # EXIT CROSSING: Moving UP, crossing EXIT line
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