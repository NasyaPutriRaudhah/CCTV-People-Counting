"""
Line Crossing Tracker
"""
import time
import numpy as np
from collections import deque
from config import CROSSING_THRESHOLD, MAX_TRACKING_DISTANCE, TRACK_TIMEOUT


class SimplifiedLineCrossingTracker:
    def __init__(self, enter_y, exit_y, threshold=CROSSING_THRESHOLD):
        self.enter_y = enter_y
        self.exit_y = exit_y
        self.threshold = threshold
        
        # Only track people temporarily for crossing detection
        self.active_tracks = {}  # {track_id: {cx, cy, last_y, last_seen, crossed_enter, crossed_exit}}
        self.next_id = 0
        
        # Recently crossed positions (to prevent double counting)
        self.recent_crossings = deque(maxlen=50)
    
    def update(self, detections):
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
                crossed_enter = track.get('crossed_enter', False)
                crossed_exit = track.get('crossed_exit', False)
                
                # ENTER CROSSING: Moving DOWN, crossing ENTER line
                if (prev_y > self.enter_y and cy < self.enter_y):
                    # Only count if they haven't already crossed the ENTER line
                    if not crossed_enter:
                        # Crossed ENTER line going DOWN
                        if not self._is_recent_crossing(cx, cy, 'entry'):
                            entries += 1
                            self._add_crossing(cx, cy, 'entry', current_time)
                            print(f"✓ [ENTRY +1] Person crossed ENTER (y: {prev_y:.0f}→{cy:.0f})")
                            # STOP TRACKING - don't add to new_tracks
                            matched_ids.add(best_match_id)
                            continue  # Skip adding to new_tracks
                    else:
                        # Already counted, they're re-entering from between the lines
                        print(f"⊘ [SKIP ENTRY] Person re-crossed ENTER but already counted (y: {prev_y:.0f}→{cy:.0f})")
                        # Continue tracking but mark they crossed enter again
                        new_tracks[best_match_id] = {
                            'cx': cx,
                            'cy': cy,
                            'last_seen': current_time,
                            'crossed_enter': True,
                            'crossed_exit': False  # Reset exit flag
                        }
                        matched_ids.add(best_match_id)
                        continue
                
                # EXIT CROSSING: Moving UP, crossing EXIT line
                elif (prev_y < self.exit_y and cy > self.exit_y):
                    # Only count if they haven't already crossed the EXIT line
                    if not crossed_exit:
                        # Crossed EXIT line going UP
                        if not self._is_recent_crossing(cx, cy, 'exit'):
                            exits += 1
                            self._add_crossing(cx, cy, 'exit', current_time)
                            print(f"✓ [EXIT -1] Person crossed EXIT (y: {prev_y:.0f}→{cy:.0f})")
                            # STOP TRACKING - don't add to new_tracks
                            matched_ids.add(best_match_id)
                            continue  # Skip adding to new_tracks
                    else:
                        # Already counted, they're re-exiting from between the lines
                        print(f"⊘ [SKIP EXIT] Person re-crossed EXIT but already counted (y: {prev_y:.0f}→{cy:.0f})")
                        # Continue tracking but mark they crossed exit again
                        new_tracks[best_match_id] = {
                            'cx': cx,
                            'cy': cy,
                            'last_seen': current_time,
                            'crossed_enter': False,  # Reset enter flag
                            'crossed_exit': True
                        }
                        matched_ids.add(best_match_id)
                        continue
                
                # Still tracking (no crossing detected) - update position
                new_tracks[best_match_id] = {
                    'cx': cx,
                    'cy': cy,
                    'last_seen': current_time,
                    'crossed_enter': crossed_enter,
                    'crossed_exit': crossed_exit
                }
                matched_ids.add(best_match_id)
            
            else:
                # New detection - start tracking
                new_id = self.next_id
                self.next_id += 1
                
                new_tracks[new_id] = {
                    'cx': cx,
                    'cy': cy,
                    'last_seen': current_time,
                    'crossed_enter': False,
                    'crossed_exit': False
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