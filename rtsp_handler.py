import cv2
import time
import threading
from config import MAX_RECONNECT_ATTEMPTS, RECONNECT_DELAY


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