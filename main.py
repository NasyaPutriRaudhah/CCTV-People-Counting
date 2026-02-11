import cv2
import time
import numpy as np

# Import configuration
from config import (
    DEVICE, VIDEO1_PATH, VIDEO2_PATH, STARTING_COUNT, 
    DATA_SAVE_INTERVAL, LINE2_ENTER_Y, LINE2_EXIT_Y
)

# Import utilities and components
from utils import print_device_info
from models import load_models
from timers import InferenceTimer
from buffer import FrameBuffer
from data_persistence import DataPersistence
from trackers import SimplifiedLineCrossingTracker
from rtsp_handler import RTSPCaptureThread
from processors import process_video1_roi, process_video2_line
from display import create_combined_view, print_final_report


def main():
    # Print device info
    print_device_info(DEVICE)
    print()
    
    # Load models
    model1, model2 = load_models()
    
    # Initialize data persistence
    data_persistence = DataPersistence()
    
    # Initialize line tracker
    line_tracker = SimplifiedLineCrossingTracker(LINE2_ENTER_Y, LINE2_EXIT_Y)
    print("✓ Simplified tracker initialized")
    
    # Open Video 1
    cap1 = cv2.VideoCapture(VIDEO1_PATH)
    assert cap1.isOpened(), "Video 1 cannot be opened"
    print("✓ Video 1 loaded")
    
    # Initialize frame buffer for Video 2
    frame_buffer_v2 = FrameBuffer()
    print("✓ Frame buffer created")
    
    # Initialize timers
    timer_v1 = InferenceTimer(name="Video 1")
    timer_v2 = InferenceTimer(name="Video 2")
    
    # Start RTSP capture thread
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
    
    # Create display window
    cv2.namedWindow("People Counting System", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # ==========================================
            # PROCESS VIDEO 1 - ROI DETECTION
            # ==========================================
            ret1, frame1 = cap1.read()
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            processed_frame1, c1 = process_video1_roi(frame1, model1, id_history_v1, timer_v1)
            count_v1 = c1
            
            # ==========================================
            # PROCESS VIDEO 2 - LINE CROSSING
            # ==========================================
            if not frame_buffer_v2.is_empty():
                frame2 = frame_buffer_v2.get_frame()
                
                if frame2 is not None:
                    processed_frame2, count_v2 = process_video2_line(
                        frame2, model2, timer_v2, line_tracker, data_persistence
                    )
                else:
                    processed_frame2 = np.zeros((600, 1020, 3), dtype=np.uint8)
                    cv2.putText(processed_frame2, "Buffer read error", (300, 300),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                processed_frame2 = np.zeros((600, 1020, 3), dtype=np.uint8)
                cv2.putText(processed_frame2, "Waiting for frames...", (300, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # ==========================================
            # CALCULATE FPS
            # ==========================================
            process_frame_count += 1
            elapsed = time.time() - process_start_time
            if elapsed >= 1.0:
                fps_process = process_frame_count / elapsed
                process_frame_count = 0
                process_start_time = time.time()
            
            # ==========================================
            # AUTO-SAVE DATA
            # ==========================================
            current_time = time.time()
            if current_time - last_save_time >= DATA_SAVE_INTERVAL:
                data_persistence.save_data()
                last_save_time = current_time
            
            # ==========================================
            # DISPLAY COMBINED VIEW
            # ==========================================
            data_summary = data_persistence.get_summary()
            combined_view = create_combined_view(
                processed_frame1, processed_frame2,
                count_v1, count_v2,
                frame_buffer_v2.size(),
                rtsp_thread.fps,
                fps_process,
                data_summary
            )
            
            cv2.imshow("People Counting System", combined_view)
            
            # ==========================================
            # EXIT ON ESC
            # ==========================================
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
        # ==========================================
        # CLEANUP
        # ==========================================
        total_runtime = time.time() - runtime_start
        
        print("\nCleaning up...")
        data_persistence.save_data()
        print("✓ Data saved")
        
        rtsp_thread.stop()
        rtsp_thread.join(timeout=2)
        cap1.release()
        frame_buffer_v2.clear()
        cv2.destroyAllWindows()
        
        # Print final report
        print_final_report(timer_v1, timer_v2, total_runtime, data_persistence)
        
        print("Processing completed!")


if __name__ == "__main__":
    main()