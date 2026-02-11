"""
Frame Buffer Management for RTSP Streams
"""
import os
import cv2
from collections import deque
from config import BUFFER_SIZE, SAVE_BUFFER_TO_DISK, BUFFER_DIR


class FrameBuffer:
    """Buffer to store frames from RTSP before processing"""
    
    def __init__(self, max_size=BUFFER_SIZE, save_to_disk=SAVE_BUFFER_TO_DISK, 
                 buffer_dir=BUFFER_DIR):
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