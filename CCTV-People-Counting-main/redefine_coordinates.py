# redefine_coordinates.py
import cv2
import numpy as np
import json

class CoordinateRedefineTool:
    """Tool to redefine ROI and lines on cropped video"""
    
    def __init__(self):
        self.points = []
        self.drawing = False
        self.current_frame = None
        self.display_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for drawing points"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.drawing = True
            print(f"Point added: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                removed = self.points.pop()
                print(f"Point removed: {removed}")
        
        # Redraw
        self.display_frame = self.current_frame.copy()
        
        # Draw all points
        for i, pt in enumerate(self.points):
            cv2.circle(self.display_frame, pt, 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, str(i), (pt[0]+10, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw lines connecting points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                cv2.line(self.display_frame, self.points[i], self.points[i+1], (0, 255, 0), 2)
            # Close the polygon if in ROI mode
            if param == "roi" and len(self.points) > 2:
                cv2.line(self.display_frame, self.points[-1], self.points[0], (0, 255, 0), 2)
        
        cv2.imshow("Redefine Coordinates", self.display_frame)
    
    def define_roi_polygon(self, frame, video_name):
        """
        Define ROI polygon by clicking points
        
        Returns: numpy array of points or None
        """
        self.current_frame = frame.copy()
        self.display_frame = frame.copy()
        self.points = []
        
        window_name = "Redefine Coordinates"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback, "roi")
        
        print(f"\n{'='*60}")
        print(f"Define ROI Polygon for {video_name}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  - LEFT CLICK to add points")
        print("  - RIGHT CLICK to remove last point")
        print("  - Press 'c' to confirm (minimum 3 points)")
        print("  - Press 'r' to reset")
        print("  - Press 's' to skip")
        print(f"{'='*60}\n")
        
        while True:
            cv2.imshow(window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if len(self.points) >= 3:
                    break
                else:
                    print("Need at least 3 points for ROI polygon!")
            
            elif key == ord('r'):
                self.points = []
                self.display_frame = self.current_frame.copy()
                print("Reset")
            
            elif key == ord('s'):
                print("Skipped ROI definition")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        roi_array = np.array(self.points, dtype=np.int32)
        
        print(f"\n✓ ROI polygon defined with {len(self.points)} points:")
        print("ROI1_POINTS = np.array([")
        for pt in self.points:
            print(f"    [{pt[0]}, {pt[1]}],")
        print("], dtype=np.int32)")
        
        return roi_array
    
    def define_lines(self, frame, video_name):
        """
        Define two horizontal lines (enter and exit)
        
        Returns: dict with line coordinates or None
        """
        self.current_frame = frame.copy()
        self.display_frame = frame.copy()
        
        frame_h, frame_w = frame.shape[:2]
        
        # Initialize with default positions
        enter_y = frame_h // 3
        exit_y = 2 * frame_h // 3
        x1 = int(frame_w * 0.2)
        x2 = int(frame_w * 0.8)
        
        window_name = "Define Lines"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Trackbars for adjusting lines
        cv2.createTrackbar('Enter Y', window_name, enter_y, frame_h-1, lambda x: None)
        cv2.createTrackbar('Exit Y', window_name, exit_y, frame_h-1, lambda x: None)
        cv2.createTrackbar('Line X1', window_name, x1, frame_w-1, lambda x: None)
        cv2.createTrackbar('Line X2', window_name, x2, frame_w-1, lambda x: None)
        
        print(f"\n{'='*60}")
        print(f"Define Lines for {video_name}")
        print(f"{'='*60}")
        print("Instructions:")
        print("  - Use trackbars to adjust line positions")
        print("  - BLUE line = ENTER (cross DOWN = +1)")
        print("  - RED line = EXIT (cross UP = -1)")
        print("  - Press 'c' to confirm")
        print("  - Press 's' to skip")
        print(f"{'='*60}\n")
        
        while True:
            display = self.current_frame.copy()
            
            # Get current trackbar values
            enter_y = cv2.getTrackbarPos('Enter Y', window_name)
            exit_y = cv2.getTrackbarPos('Exit Y', window_name)
            x1 = cv2.getTrackbarPos('Line X1', window_name)
            x2 = cv2.getTrackbarPos('Line X2', window_name)
            
            # Draw lines
            cv2.line(display, (x1, enter_y), (x2, enter_y), (255, 0, 0), 3)
            cv2.putText(display, "ENTER LINE (cross DOWN = +1)", (x1, enter_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.line(display, (x1, exit_y), (x2, exit_y), (0, 0, 255), 3)
            cv2.putText(display, "EXIT LINE (cross UP = -1)", (x1, exit_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show coordinates
            cv2.putText(display, f"Enter Y: {enter_y}, Exit Y: {exit_y}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"X1: {x1}, X2: {x2}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                break
            elif key == ord('s'):
                print("Skipped line definition")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        lines_config = {
            'x1': x1,
            'x2': x2,
            'enter_y': enter_y,
            'exit_y': exit_y
        }
        
        print(f"\n✓ Lines defined:")
        print(f"LINE2_X1 = {x1}")
        print(f"LINE2_X2 = {x2}")
        print(f"LINE2_ENTER_Y = {enter_y}")
        print(f"LINE2_EXIT_Y = {exit_y}")
        
        return lines_config


def apply_crop_to_frame(frame, crop_config):
    """Apply crop configuration to a frame"""
    if crop_config is None:
        return frame
    
    x = crop_config['x']
    y = crop_config['y']
    w = crop_config['width']
    h = crop_config['height']
    
    return frame[y:y+h, x:x+w]


def main():
    """Main function to redefine coordinates on cropped videos"""
    
    print("\n" + "="*60)
    print("COORDINATE REDEFINITION TOOL")
    print("="*60)
    
    # Load existing crop configuration
    try:
        with open('crop_config.json', 'r') as f:
            crop_config = json.load(f)
        print("✓ Loaded existing crop configuration")
    except FileNotFoundError:
        print("⚠ No crop_config.json found")
        print("Run 'python preprocess.py' first to create crop regions")
        return
    
    # Import video paths
    try:
        from config import VIDEO1_PATH, VIDEO2_PATH
    except ImportError:
        print("Error: Cannot import VIDEO paths from config.py")
        return
    
    tool = CoordinateRedefineTool()
    new_config = {
        'crop': crop_config,
        'video1_roi': None,
        'video2_lines': None
    }
    
    # ====================================
    # VIDEO 1 - ROI DEFINITION
    # ====================================
    print(f"\n{'='*60}")
    print("VIDEO 1 - ROI POLYGON DEFINITION")
    print(f"{'='*60}")
    
    cap1 = cv2.VideoCapture(VIDEO1_PATH)
    ret1, frame1 = cap1.read()
    cap1.release()
    
    if not ret1:
        print("Error: Cannot read Video 1")
        return
    
    # Apply crop to Video 1
    frame1_cropped = apply_crop_to_frame(frame1, crop_config['video1'])
    print(f"Video 1 cropped size: {frame1_cropped.shape[1]}x{frame1_cropped.shape[0]}")
    
    # Define ROI on cropped frame
    roi_points = tool.define_roi_polygon(frame1_cropped, "Video 1 (Cropped)")
    
    if roi_points is not None:
        new_config['video1_roi'] = roi_points.tolist()
    
    # ====================================
    # VIDEO 2 - LINE DEFINITION
    # ====================================
    print(f"\n{'='*60}")
    print("VIDEO 2 - LINE DEFINITION")
    print(f"{'='*60}")
    
    cap2 = cv2.VideoCapture(VIDEO2_PATH)
    print("Connecting to RTSP stream...")
    import time
    time.sleep(2)
    
    ret2, frame2 = cap2.read()
    cap2.release()
    
    if not ret2:
        print("Error: Cannot read Video 2")
        return
    
    # Apply crop to Video 2
    frame2_cropped = apply_crop_to_frame(frame2, crop_config['video2'])
    print(f"Video 2 cropped size: {frame2_cropped.shape[1]}x{frame2_cropped.shape[0]}")
    
    # Define lines on cropped frame
    lines_config = tool.define_lines(frame2_cropped, "Video 2 (Cropped)")
    
    if lines_config is not None:
        new_config['video2_lines'] = lines_config
    
    # ====================================
    # SAVE CONFIGURATION
    # ====================================
    output_file = "coordinates_cropped.json"
    with open(output_file, 'w') as f:
        json.dump(new_config, f, indent=4)
    
    print(f"\n{'='*60}")
    print("CONFIGURATION SAVED")
    print(f"{'='*60}")
    print(f"✓ Saved to {output_file}")
    
    # Generate Python code
    print(f"\n{'='*60}")
    print("PYTHON CODE FOR CONFIG.PY")
    print(f"{'='*60}\n")
    
    if new_config['video1_roi'] is not None:
        print("# Video 1 ROI (for cropped video)")
        print("ROI1_POINTS_CROPPED = np.array([")
        for pt in new_config['video1_roi']:
            print(f"    [{pt[0]}, {pt[1]}],")
        print("], dtype=np.int32)\n")
    
    if new_config['video2_lines'] is not None:
        print("# Video 2 Lines (for cropped video)")
        lines = new_config['video2_lines']
        print(f"LINE2_X1_CROPPED = {lines['x1']}")
        print(f"LINE2_X2_CROPPED = {lines['x2']}")
        print(f"LINE2_ENTER_Y_CROPPED = {lines['enter_y']}")
        print(f"LINE2_EXIT_Y_CROPPED = {lines['exit_y']}\n")
    
    print(f"{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Copy the code above to your config.py")
    print("2. Or use the JSON file directly in your main.py")
    print("3. Run your main application")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()