# preprocess.py
import cv2
import json
import os
import numpy as np

class CropRegionSelector:
    """Interactive tool to select and save crop regions for videos"""
    
    def __init__(self):
        self.ref_point = []
        self.cropping = False
        self.clone = None
        self.current_frame = None
    
    def click_and_crop(self, event, x, y, flags, param):
        """Mouse callback function for selecting crop region"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ref_point = [(x, y)]
            self.cropping = True
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping and len(self.ref_point) > 0:
                image = self.clone.copy()
                cv2.rectangle(image, self.ref_point[0], (x, y), (0, 255, 0), 2)
                # Show coordinates
                cv2.putText(image, f"({self.ref_point[0][0]}, {self.ref_point[0][1]}) to ({x}, {y})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Select Crop Region", image)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.ref_point.append((x, y))
            self.cropping = False
            
            cv2.rectangle(self.clone, self.ref_point[0], self.ref_point[1], (0, 255, 0), 2)
            cv2.imshow("Select Crop Region", self.clone)
    
    def select_region(self, video_source, video_name="Video", is_rtsp=False):
        """
        Select crop region from video source
        
        Args:
            video_source: Path to video file or RTSP URL
            video_name: Display name for the video
            is_rtsp: Whether the source is RTSP stream
        
        Returns: (crop_x, crop_y, crop_w, crop_h) or None if cancelled
        """
        
        cap = cv2.VideoCapture(video_source)
        
        if is_rtsp:
            print(f"Connecting to RTSP stream: {video_name}...")
            # Give RTSP stream time to initialize
            import time
            time.sleep(2)
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Cannot read from {video_name}")
            cap.release()
            return None
        
        cap.release()
        
        # Store frame copies
        self.current_frame = frame.copy()
        self.clone = frame.copy()
        self.ref_point = []
        
        # Setup window
        window_name = f"Select Crop Region - {video_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.click_and_crop)
        
        print(f"\n{'='*50}")
        print(f"Selecting crop region for {video_name}")
        print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
        print(f"{'='*50}")
        print("Instructions:")
        print("  - Click and drag to draw crop region")
        print("  - Press 'r' to reset selection")
        print("  - Press 'c' to confirm and save")
        print("  - Press 's' to skip (no cropping)")
        print("  - Press 'q' to quit")
        print(f"{'='*50}\n")
        
        while True:
            cv2.imshow(window_name, self.clone)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("r"):
                self.clone = self.current_frame.copy()
                self.ref_point = []
                print("Selection reset")
            
            elif key == ord("c"):
                if len(self.ref_point) == 2:
                    break
                else:
                    print("Please select a region first!")
            
            elif key == ord("s"):
                print(f"Skipping crop for {video_name}")
                cv2.destroyAllWindows()
                return None
            
            elif key == ord("q"):
                print("Cancelled")
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Calculate coordinates
        x1, y1 = self.ref_point[0]
        x2, y2 = self.ref_point[1]
        
        crop_x = min(x1, x2)
        crop_y = min(y1, y2)
        crop_w = abs(x2 - x1)
        crop_h = abs(y2 - y1)
        
        # Validate crop region
        if crop_w < 50 or crop_h < 50:
            print("Warning: Crop region too small!")
            return None
        
        print(f"\n✓ Crop coordinates for {video_name}:")
        print(f"  crop_x = {crop_x}")
        print(f"  crop_y = {crop_y}")
        print(f"  crop_w = {crop_w}")
        print(f"  crop_h = {crop_h}")
        print(f"  Crop size: {crop_w}x{crop_h}")
        
        # Show preview
        cropped = self.current_frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        preview_window = f"Cropped Preview - {video_name}"
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.imshow(preview_window, cropped)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return (crop_x, crop_y, crop_w, crop_h)


def save_crop_config(config, filename="crop_config.json"):
    """Save crop configuration to JSON file"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n✓ Configuration saved to {filename}")


def load_crop_config(filename="crop_config.json"):
    """Load crop configuration from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None


def apply_crop(frame, crop_region):
    """
    Apply crop to frame
    
    Args:
        frame: Input frame
        crop_region: dict with 'x', 'y', 'width', 'height' or None
    
    Returns:
        Cropped frame or original frame if crop_region is None
    """
    if crop_region is None:
        return frame
    
    x = crop_region['x']
    y = crop_region['y']
    w = crop_region['width']
    h = crop_region['height']
    
    # Safety check to prevent index errors
    frame_h, frame_w = frame.shape[:2]
    if x + w > frame_w or y + h > frame_h:
        print(f"Warning: Crop region exceeds frame bounds. Using full frame.")
        return frame
    
    return frame[y:y+h, x:x+w]


def adjust_roi_points(roi_points, crop_region):
    """
    Adjust ROI points based on crop region
    
    Args:
        roi_points: Original ROI polygon points (numpy array)
        crop_region: Crop configuration dict or None
    
    Returns:
        Adjusted ROI points (numpy array)
    """
    if crop_region is None:
        return roi_points
    
    # Subtract crop offset from all points
    adjusted_points = roi_points.copy()
    adjusted_points[:, 0] -= crop_region['x']  # Adjust x coordinates
    adjusted_points[:, 1] -= crop_region['y']  # Adjust y coordinates
    
    return adjusted_points


def adjust_line_coordinates(line_coords, crop_region):
    """
    Adjust line crossing coordinates based on crop region
    
    Args:
        line_coords: dict with 'x1', 'x2', 'enter_y', 'exit_y'
        crop_region: Crop configuration dict or None
    
    Returns:
        Adjusted line coordinates dict
    """
    if crop_region is None:
        return line_coords
    
    adjusted = line_coords.copy()
    adjusted['x1'] -= crop_region['x']
    adjusted['x2'] -= crop_region['x']
    adjusted['enter_y'] -= crop_region['y']
    adjusted['exit_y'] -= crop_region['y']
    
    return adjusted


def setup_crop_regions(video1_path, video2_path):
    """
    Interactive setup for crop regions
    
    Returns:
        config dict with crop regions for both videos
    """
    
    selector = CropRegionSelector()
    config = {
        'video1': None,
        'video2': None
    }
    
    print("\n" + "="*60)
    print("CROP REGION SETUP FOR PEOPLE COUNTING SYSTEM")
    print("="*60)
    
    # Video 1 (Local file)
    print("\nSetting up Video 1 (Local file) crop region...")
    crop1 = selector.select_region(video1_path, "Video 1 (ROI Detection)", is_rtsp=False)
    if crop1:
        config['video1'] = {
            'x': crop1[0],
            'y': crop1[1],
            'width': crop1[2],
            'height': crop1[3]
        }
    
    # Video 2 (RTSP)
    print("\nSetting up Video 2 (RTSP) crop region...")
    print("⚠ Note: RTSP connection may take a few seconds...")
    crop2 = selector.select_region(video2_path, "Video 2 (Line Crossing)", is_rtsp=True)
    if crop2:
        config['video2'] = {
            'x': crop2[0],
            'y': crop2[1],
            'width': crop2[2],
            'height': crop2[3]
        }
    
    return config


def generate_adjusted_config(crop_config):
    """
    Generate a Python config snippet with adjusted coordinates
    
    Args:
        crop_config: Crop configuration dict
    
    Returns:
        String with adjusted config code
    """
    from config import ROI1_POINTS, LINE2_X1, LINE2_X2, LINE2_ENTER_Y, LINE2_EXIT_Y
    
    output = "\n" + "="*60 + "\n"
    output += "ADJUSTED COORDINATES FOR CROPPED VIDEOS\n"
    output += "="*60 + "\n\n"
    
    # Video 1 ROI adjustment
    if crop_config['video1']:
        adjusted_roi = adjust_roi_points(ROI1_POINTS, crop_config['video1'])
        output += "# Adjusted ROI1_POINTS for Video 1:\n"
        output += f"ROI1_POINTS_ADJUSTED = np.array([\n"
        for point in adjusted_roi:
            output += f"    [{point[0]}, {point[1]}],\n"
        output += "], dtype=np.int32)\n\n"
    else:
        output += "# Video 1 - No cropping applied\n"
        output += "# ROI1_POINTS_ADJUSTED = ROI1_POINTS  # Use original\n\n"
    
    # Video 2 Line adjustment
    if crop_config['video2']:
        line_coords = {
            'x1': LINE2_X1,
            'x2': LINE2_X2,
            'enter_y': LINE2_ENTER_Y,
            'exit_y': LINE2_EXIT_Y
        }
        adjusted_line = adjust_line_coordinates(line_coords, crop_config['video2'])
        output += "# Adjusted line coordinates for Video 2:\n"
        output += f"LINE2_X1_ADJUSTED = {adjusted_line['x1']}\n"
        output += f"LINE2_X2_ADJUSTED = {adjusted_line['x2']}\n"
        output += f"LINE2_ENTER_Y_ADJUSTED = {adjusted_line['enter_y']}\n"
        output += f"LINE2_EXIT_Y_ADJUSTED = {adjusted_line['exit_y']}\n\n"
    else:
        output += "# Video 2 - No cropping applied\n"
        output += "# Use original LINE2_X1, LINE2_X2, LINE2_ENTER_Y, LINE2_EXIT_Y\n\n"
    
    output += "="*60 + "\n"
    
    return output


def main():
    """Main function to run crop region setup"""
    
    # Try to import video paths from config
    try:
        from config import VIDEO1_PATH, VIDEO2_PATH
    except ImportError:
        print("Error: Cannot import VIDEO1_PATH and VIDEO2_PATH from config.py")
        print("Please make sure config.py exists and contains these variables.")
        return
    
    print("\n" + "="*60)
    print("PEOPLE COUNTING SYSTEM - CROP SETUP TOOL")
    print("="*60)
    print(f"\nVideo 1: {VIDEO1_PATH}")
    print(f"Video 2: {VIDEO2_PATH}")
    
    # Check if config already exists
    if os.path.exists("crop_config.json"):
        print("\n⚠ Found existing crop_config.json")
        response = input("Do you want to recreate it? (y/n): ")
        if response.lower() != 'y':
            print("Using existing configuration")
            
            # Show existing config
            existing_config = load_crop_config()
            if existing_config:
                print("\nCurrent configuration:")
                print(json.dumps(existing_config, indent=2))
            return
    
    # Setup crop regions
    config = setup_crop_regions(VIDEO1_PATH, VIDEO2_PATH)
    
    # Save configuration
    save_crop_config(config)
    
    # Generate adjusted coordinates
    try:
        adjusted_code = generate_adjusted_config(config)
        print(adjusted_code)
        
        # Save to file
        with open("adjusted_coordinates.py", "w") as f:
            f.write("import numpy as np\n\n")
            f.write(adjusted_code)
        print("✓ Adjusted coordinates saved to adjusted_coordinates.py")
    except Exception as e:
        print(f"⚠ Could not generate adjusted coordinates: {e}")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review crop_config.json")
    print("2. Check adjusted_coordinates.py for new ROI/line coordinates")
    print("3. Update your config.py with adjusted coordinates if needed")
    print("4. Run your main.py - cropping will be applied automatically")
    print("\nUsage in code:")
    print("  from preprocess import load_crop_config, apply_crop")
    print("  crop_config = load_crop_config()")
    print("  cropped_frame = apply_crop(frame, crop_config['video1'])")
    print("="*60)


if __name__ == "__main__":
    main()