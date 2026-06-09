"""
test_metrics.py
===============
Drop-in testing wrapper for CCTV People Counting System.

Tests:
  - Median latency (target: <= 1.5 s)
  - Frame drop rate (target: < 2%)

Run:
  python test_metrics.py

Results are printed to console AND saved to test_results.json
"""

import os
import cv2
import time
import json
import numpy as np

from config import (
    DEVICE, VIDEO1_PATH, VIDEO2_PATH,
    DATA_SAVE_INTERVAL, VIDEO_RESIZE_WIDTH, VIDEO_RESIZE_HEIGHT
)
from utils import print_device_info
from models import load_models
from timers import InferenceTimer
from buffer import FrameBuffer
from data_persistence import DataPersistence
from trackers import CentroidLineCrossingTracker
from rtsp_handler import RTSPCaptureThread
from processors import process_video1_roi, process_video2_line

import json as _json


# ─── How long to run the test (seconds) ───────────────────────────────────────
TEST_DURATION = 120   # 2 minutes — change to 300 for 5-min test


def load_coordinates_config():
    try:
        with open('coordinates_cropped.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: Missing coordinates_cropped.json")
        return None


def apply_crop(frame, crop_config):
    if crop_config is None:
        return frame
    x = crop_config['x']
    y = crop_config['y']
    w = crop_config['width']
    h = crop_config['height']
    return frame[y:y+h, x:x+w]


def print_banner(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print_banner("CCTV PEOPLE COUNTING — LATENCY & FRAME DROP TEST")
    print(f"Test duration : {TEST_DURATION} seconds")
    print(f"Target latency: median <= 1.5 s")
    print(f"Target drops  : < 2%")
    print()

    # ── Load config ─────────────────────────────────────────────────────────────
    coords_config = load_coordinates_config()
    if coords_config is None:
        return

    crop_config = coords_config['crop']

    roi1_points = np.array(coords_config['video1_roi'], dtype=np.int32)         if coords_config['video1_roi'] else None

    lines = coords_config.get('video2_lines', {})
    line2_x1       = lines.get('x1',       0)
    line2_x2       = lines.get('x2',       640)
    line2_enter_y  = lines.get('enter_y',  240)
    line2_exit_y   = lines.get('exit_y',   280)

    from config import ROI1_POINTS, LINE2_X1, LINE2_X2, LINE2_ENTER_Y, LINE2_EXIT_Y
    if roi1_points is None:
        roi1_points = ROI1_POINTS

    print_device_info(DEVICE)
    model1, model2 = load_models()

    data_persistence = DataPersistence()
    line_tracker = CentroidLineCrossingTracker(line_y=line2_enter_y, max_disappeared=10)

    cap1 = cv2.VideoCapture(VIDEO1_PATH)
    assert cap1.isOpened(), "Video 1 cannot be opened"

    frame_buffer_v2 = FrameBuffer()
    timer_v1 = InferenceTimer(name="Video 1")
    timer_v2 = InferenceTimer(name="Video 2")

    rtsp_thread = RTSPCaptureThread(VIDEO2_PATH, frame_buffer_v2)
    rtsp_thread.start()

    print("\nFilling buffer...")
    time.sleep(3)
    print(f"Buffer ready with {frame_buffer_v2.size()} frames\n")

    # ════════════════════════════════════════════════════════════════════════════
    # TEST METRICS — all counters defined here
    # ════════════════════════════════════════════════════════════════════════════

    # LATENCY: time between crossing event (inference start) and count update
    latency_log_v1 = []   # per-frame inference latency for Video 1 (ms)
    latency_log_v2 = []   # per-frame inference latency for Video 2 (ms)
    crossing_latency_log = []  # latency specifically for line-crossing events (ms)

    # FRAME DROP: Video 2 (RTSP) — how often buffer was empty or frame was None
    total_v2_attempts    = 0   # every loop iteration we TRIED to get a V2 frame
    total_v2_processed   = 0   # frames actually processed
    total_v2_dropped     = 0   # buffer empty or frame None

    # Video 1 drop tracking
    total_v1_attempts    = 0
    total_v1_processed   = 0
    total_v1_dropped     = 0

    # ════════════════════════════════════════════════════════════════════════════

    id_history_v1 = {}
    count_v1 = 0
    count_v2 = data_persistence.get_current_count()

    test_start = time.time()
    last_save_time = test_start
    frame_count = 0

    print(f"Running test for {TEST_DURATION}s — press Ctrl+C to stop early\n")

    try:
        while True:
            elapsed_total = time.time() - test_start
            if elapsed_total >= TEST_DURATION:
                print(f"\nTest duration ({TEST_DURATION}s) reached.")
                break

            # ── VIDEO 1: ROI detection ───────────────────────────────────────
            total_v1_attempts += 1
            ret1, frame1 = cap1.read()
            if not ret1:
                cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
                total_v1_dropped += 1
                continue

            frame1_cropped = apply_crop(frame1, crop_config['video1'])

            # MEASURE LATENCY V1: time from frame read → count updated
            t_v1_start = time.time()
            processed_frame1, count_v1 = process_video1_roi(
                frame1_cropped, model1, id_history_v1, timer_v1,
                roi_points=roi1_points
            )
            t_v1_end = time.time()

            latency_ms_v1 = (t_v1_end - t_v1_start) * 1000
            latency_log_v1.append(latency_ms_v1)
            total_v1_processed += 1

            # ── VIDEO 2: Line crossing detection ────────────────────────────
            total_v2_attempts += 1

            if not frame_buffer_v2.is_empty():
                frame2 = frame_buffer_v2.get_frame()

                if frame2 is not None:
                    frame2_cropped = apply_crop(frame2, crop_config['video2'])

                    prev_count = data_persistence.get_current_count()

                    # MEASURE LATENCY V2: time from frame read → count updated
                    t_v2_start = time.time()
                    processed_frame2, count_v2 = process_video2_line(
                        frame2_cropped, model2, timer_v2, line_tracker, data_persistence,
                        line_x1=line2_x1,
                        line_x2=line2_x2,
                        line_enter_y=line2_enter_y,
                        line_exit_y=line2_exit_y
                    )
                    t_v2_end = time.time()

                    latency_ms_v2 = (t_v2_end - t_v2_start) * 1000
                    latency_log_v2.append(latency_ms_v2)
                    total_v2_processed += 1

                    # CROSSING EVENT LATENCY: only log when count actually changed
                    new_count = data_persistence.get_current_count()
                    if new_count != prev_count:
                        crossing_latency_log.append(latency_ms_v2)
                        print(f"  [CROSSING] count {prev_count} → {new_count} | latency {latency_ms_v2:.1f} ms")

                else:
                    total_v2_dropped += 1  # frame was None
            else:
                total_v2_dropped += 1  # buffer was empty

            # ── Progress print every 10s ─────────────────────────────────────
            frame_count += 1
            if frame_count % 300 == 0:
                pct_done = (elapsed_total / TEST_DURATION) * 100
                drop_rate = (total_v2_dropped / total_v2_attempts * 100) if total_v2_attempts else 0
                med_lat   = np.median(latency_log_v2) / 1000 if latency_log_v2 else 0
                print(f"  [{pct_done:5.1f}%] frame={frame_count:5d} | "
                      f"drop={drop_rate:.2f}% | median_lat={med_lat:.3f}s | "
                      f"count_v2={count_v2}")

            # ── Auto-save ────────────────────────────────────────────────────
            if time.time() - last_save_time >= DATA_SAVE_INTERVAL:
                data_persistence.save_data()
                last_save_time = time.time()

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    finally:
        rtsp_thread.stop()
        rtsp_thread.join(timeout=2)
        cap1.release()
        frame_buffer_v2.clear()
        data_persistence.save_data()

    # ════════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════════════════════
    print_banner("TEST RESULTS")

    # --- Frame Drop Rate --------------------------------------------------------
    drop_rate_v2 = (total_v2_dropped / total_v2_attempts * 100) if total_v2_attempts else 0
    drop_rate_v1 = (total_v1_dropped / total_v1_attempts * 100) if total_v1_attempts else 0
    drop_pass    = drop_rate_v2 < 2.0

    print(f"\n📦 FRAME DROP RATE (Video 2 / RTSP)")
    print(f"  Total attempts  : {total_v2_attempts}")
    print(f"  Processed       : {total_v2_processed}")
    print(f"  Dropped         : {total_v2_dropped}")
    print(f"  Drop rate       : {drop_rate_v2:.2f}%  {'✅ PASS (< 2%)' if drop_pass else '❌ FAIL (>= 2%)'}")

    print(f"\n📦 FRAME DROP RATE (Video 1 / local file)")
    print(f"  Drop rate       : {drop_rate_v1:.2f}%")

    # --- Latency ----------------------------------------------------------------
    if latency_log_v2:
        arr_v2 = np.array(latency_log_v2)
        median_lat_s  = np.median(arr_v2) / 1000
        mean_lat_s    = np.mean(arr_v2) / 1000
        max_lat_s     = np.max(arr_v2) / 1000
        p95_lat_s     = np.percentile(arr_v2, 95) / 1000
        lat_pass      = median_lat_s <= 1.5
    else:
        median_lat_s = mean_lat_s = max_lat_s = p95_lat_s = 0
        lat_pass = False

    print(f"\n⏱  LATENCY (Video 2 — per frame inference)")
    print(f"  Median          : {median_lat_s:.3f} s  {'✅ PASS (<= 1.5s)' if lat_pass else '❌ FAIL (> 1.5s)'}")
    print(f"  Mean            : {mean_lat_s:.3f} s")
    print(f"  Max             : {max_lat_s:.3f} s")
    print(f"  P95             : {p95_lat_s:.3f} s")

    if crossing_latency_log:
        arr_c = np.array(crossing_latency_log)
        print(f"\n⏱  LATENCY at LINE-CROSSING EVENTS only")
        print(f"  Events captured : {len(crossing_latency_log)}")
        print(f"  Median          : {np.median(arr_c)/1000:.3f} s")
        print(f"  Max             : {np.max(arr_c)/1000:.3f} s")
    else:
        print(f"\n⏱  No crossing events captured during test.")

    if latency_log_v1:
        arr_v1 = np.array(latency_log_v1)
        print(f"\n⏱  LATENCY (Video 1 — ROI inference)")
        print(f"  Median          : {np.median(arr_v1)/1000:.3f} s")
        print(f"  Mean            : {np.mean(arr_v1)/1000:.3f} s")

    # --- InferenceTimer stats (already collected by your existing timers) ------
    stats_v2 = timer_v2.get_statistics()
    print(f"\n📊 InferenceTimer (Video 2) — from your existing timer")
    print(f"  Frames          : {stats_v2['count']}")
    print(f"  Mean ms/frame   : {stats_v2['mean']:.1f} ms")
    print(f"  Median ms/frame : {stats_v2['median']:.1f} ms")
    print(f"  P95 ms/frame    : {stats_v2['p95']:.1f} ms")
    print(f"  Max FPS         : {stats_v2['max_fps']:.1f}")

    # --- Overall verdict --------------------------------------------------------
    overall_pass = drop_pass and lat_pass
    print_banner("OVERALL VERDICT")
    print(f"  Frame drop rate < 2%      : {'✅ PASS' if drop_pass else '❌ FAIL'}")
    print(f"  Median latency <= 1.5 s   : {'✅ PASS' if lat_pass else '❌ FAIL'}")
    print(f"\n  RESULT: {'✅ ALL PASS' if overall_pass else '❌ SOME TESTS FAILED'}")

    # --- Save to JSON -----------------------------------------------------------
    results = {
        "test_duration_s": TEST_DURATION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_drop": {
            "v2_total_attempts": total_v2_attempts,
            "v2_processed": total_v2_processed,
            "v2_dropped": total_v2_dropped,
            "v2_drop_rate_pct": round(drop_rate_v2, 3),
            "pass": drop_pass
        },
        "latency": {
            "median_s": round(median_lat_s, 4),
            "mean_s": round(mean_lat_s, 4),
            "max_s": round(max_lat_s, 4),
            "p95_s": round(p95_lat_s, 4),
            "crossing_events": len(crossing_latency_log),
            "crossing_median_s": round(np.median(crossing_latency_log) / 1000, 4) if crossing_latency_log else None,
            "pass": lat_pass
        },
        "inference_timer_v2": stats_v2,
        "overall_pass": overall_pass
    }

    with open("test_results.json", "w") as f:
        _json.dump(results, f, indent=2)

    print(f"\n  Results saved to: test_results.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
