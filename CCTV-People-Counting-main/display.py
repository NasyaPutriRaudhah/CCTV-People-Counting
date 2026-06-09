import cv2
import numpy as np
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, DATA_SAVE_FILE


def create_combined_view(frame1, frame2, count_v1, count_v2, buffer_size,
                        fps_capture, fps_process, data_summary):

    frame1_display = cv2.resize(frame1, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    frame2_display = cv2.resize(frame2, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    combined_view = np.hstack([frame1_display, frame2_display])

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
