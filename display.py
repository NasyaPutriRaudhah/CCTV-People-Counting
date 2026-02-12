"""
Display and Visualization Functions
"""
import cv2
import numpy as np
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, DATA_SAVE_FILE


def create_combined_view(frame1, frame2, count_v1, count_v2, buffer_size, 
                        fps_capture, fps_process, data_summary):

    frame1_display = cv2.resize(frame1, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    frame2_display = cv2.resize(frame2, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    
    combined_view = np.hstack([frame1_display, frame2_display])
    
    # Statistics overlay
    overlay = combined_view.copy()
    stats_box_height = 300
    stats_box_width = 520
    
    cv2.rectangle(overlay, (10, 10), (stats_box_width, stats_box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, combined_view, 0.25, 0, combined_view)
    cv2.rectangle(combined_view, (10, 10), (stats_box_width, stats_box_height), (255, 255, 255), 2)
    
    y_offset = 35
    line_height = 30
    
    cv2.putText(combined_view, 'STATISTICS', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    y_offset += line_height
    cv2.putText(combined_view, f'Video 1: {count_v1}', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    
    # Line crossing details
    y_offset += line_height + 5
    cv2.putText(combined_view, 'VIDEO 2 :', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    
    y_offset += line_height
    cv2.putText(combined_view, f'Current: {count_v2}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset += line_height - 5
    cv2.putText(combined_view, f'  Start: {data_summary["starting_count"]}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += line_height - 10
    cv2.putText(combined_view, f'  +Entry: {data_summary["total_entries"]}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    y_offset += line_height - 10
    cv2.putText(combined_view, f'  -Exit: {data_summary["total_exits"]}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    y_offset += line_height - 10
    net_change = data_summary["net_change"]
    net_color = (0, 255, 0) if net_change >= 0 else (0, 0, 255)
    cv2.putText(combined_view, f'  Net: {net_change:+d}', (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, net_color, 1)
    
    combined = count_v1 + count_v2
    y_offset += line_height
    cv2.putText(combined_view, f'TOTAL: {combined}', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Note
    y_offset += line_height
    cv2.putText(combined_view, 'Track only for crossing, stop after crossing', (25, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    
    # FPS info
    y_offset += line_height - 5
    cv2.putText(combined_view, f'Buffer: {buffer_size} | Cap: {fps_capture:.1f} | Proc: {fps_process:.1f} fps', 
                (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return combined_view


def print_final_report(timer_v1, timer_v2, total_runtime, data_persistence):
    stats_v1 = timer_v1.get_statistics()
    stats_v2 = timer_v2.get_statistics()
    summary = data_persistence.get_summary()
    
    print("\n" + "="*80)
    print("FINAL REPORT - SIMPLIFIED TRACKING SYSTEM")
    print("="*80)
    
    print(f"\n⏱️  SESSION INFORMATION")
    print(f"   Started:      {summary['session_start']}")
    print(f"   Ended:        {summary['last_update']}")
    print(f"   Runtime:      {total_runtime:.2f} seconds ({total_runtime/3600:.2f} hours)")
    
    print("\n" + "-"*80)
    print("👥 LINE CROSSING SUMMARY")
    print("-"*80)
    print(f"   Starting count:       {summary['starting_count']} people")
    print(f"   Current count:        {summary['current_count']} people")
    print(f"   Total entries:        +{summary['total_entries']} crossings")
    print(f"   Total exits:          -{summary['total_exits']} crossings")
    print(f"   Net change:           {summary['net_change']:+d} people")
    
    print("\n" + "-"*80)
    print("📊 INFERENCE PERFORMANCE")
    print("-"*80)
    print(f"Video 1: {stats_v1['count']} frames, {stats_v1['mean']:.2f} ms avg")
    print(f"Video 2: {stats_v2['count']} frames, {stats_v2['mean']:.2f} ms avg")
    
    avg_inference = (stats_v1['mean'] + stats_v2['mean']) / 2
    print(f"\nAverage: {avg_inference:.2f} ms/frame")
    
    if avg_inference < 33.33:
        print(f"✅ Can handle real-time 30 FPS")
    elif avg_inference < 66.67:
        print(f"⚠️  Can handle ~15 FPS")
    else:
        print(f"❌ May struggle with real-time")
    
    print("\n" + "="*80)
    print(f"Data saved to: {DATA_SAVE_FILE}")
    print("="*80 + "\n")