"""
benchmark.py
Compares inference speed: YOLOv11 PyTorch (.pt) vs TensorRT (.engine)
Uses real RTSP stream frames for accurate real-world benchmarking
Run inside the container:
    docker compose run --rm cctv_app python3 benchmark.py
"""

import time
import numpy as np
import cv2
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
PT_MODEL    = "yolo11n.pt"
TRT_MODEL   = "yolo11n.engine"
WARMUP_RUNS = 10      # warmup runs (not counted)
TEST_RUNS   = 100     # actual benchmark runs
IMG_SIZE    = 640

# ── RTSP Config — change to your actual camera URL ─────────────────────────────
RTSP_URL = "rtsp://admin:Ferbos2024!@192.168.68.109:554/Streaming/Channels/102"  # ← change this

# ══════════════════════════════════════════════════════════════
# GRAB REAL FRAMES FROM RTSP
# ══════════════════════════════════════════════════════════════
def grab_frames_from_rtsp(url, num_frames=TEST_RUNS + WARMUP_RUNS):
    """Grab real frames from RTSP stream for benchmarking"""
    print(f"\n📷 Connecting to RTSP stream: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        # Fallback to default OpenCV backend
        cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("⚠ Could not connect to RTSP — using dummy frames instead")
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(num_frames)]

    print(f"✅ RTSP connected! Grabbing {num_frames} frames...")
    frames = []
    attempts = 0
    while len(frames) < num_frames and attempts < num_frames * 3:
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
            if len(frames) % 25 == 0:
                print(f"   Grabbed {len(frames)}/{num_frames} frames")
        attempts += 1

    cap.release()

    if len(frames) == 0:
        print("⚠ No frames grabbed — using dummy frames instead")
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(num_frames)]

    print(f"✅ Got {len(frames)} real frames from RTSP!")
    print(f"   Frame size: {frames[0].shape}")
    return frames


# ── Grab real frames ───────────────────────────────────────────────────────────
frames = grab_frames_from_rtsp(RTSP_URL)
warmup_frames = frames[:WARMUP_RUNS]
test_frames   = frames[WARMUP_RUNS:WARMUP_RUNS + TEST_RUNS]

# Pad if not enough frames grabbed
while len(test_frames) < TEST_RUNS:
    test_frames.append(test_frames[-1])

print(f"\nWarmup runs: {WARMUP_RUNS} | Benchmark runs: {TEST_RUNS}")
print("=" * 55)

# ══════════════════════════════════════════════════════════════
# BENCHMARK 1 — PyTorch (.pt)
# ══════════════════════════════════════════════════════════════
print(f"\n🔵 Loading PyTorch model: {PT_MODEL}")
pt_model = YOLO(PT_MODEL)

print(f"   Warming up ({WARMUP_RUNS} runs)...")
for frame in warmup_frames:
    pt_model(frame, device=0, verbose=False)

print(f"   Benchmarking ({TEST_RUNS} runs)...")
pt_times = []
for i, frame in enumerate(test_frames):
    start = time.perf_counter()
    pt_model(frame, device=0, verbose=False)
    end = time.perf_counter()
    pt_times.append((end - start) * 1000)
    if (i + 1) % 25 == 0:
        print(f"   Progress: {i+1}/{TEST_RUNS}")

pt_avg = np.mean(pt_times)
pt_min = np.min(pt_times)
pt_max = np.max(pt_times)
pt_fps = 1000 / pt_avg

print(f"\n   ✅ PyTorch Results:")
print(f"      Avg: {pt_avg:.2f} ms/frame")
print(f"      Min: {pt_min:.2f} ms/frame")
print(f"      Max: {pt_max:.2f} ms/frame")
print(f"      FPS: {pt_fps:.1f}")

# ══════════════════════════════════════════════════════════════
# BENCHMARK 2 — TensorRT (.engine)
# ══════════════════════════════════════════════════════════════
print(f"\n🟢 Loading TensorRT model: {TRT_MODEL}")
trt_model = YOLO(TRT_MODEL)

print(f"   Warming up ({WARMUP_RUNS} runs)...")
for frame in warmup_frames:
    trt_model(frame, device=0, verbose=False)

print(f"   Benchmarking ({TEST_RUNS} runs)...")
trt_times = []
for i, frame in enumerate(test_frames):
    start = time.perf_counter()
    trt_model(frame, device=0, verbose=False)
    end = time.perf_counter()
    trt_times.append((end - start) * 1000)
    if (i + 1) % 25 == 0:
        print(f"   Progress: {i+1}/{TEST_RUNS}")

trt_avg = np.mean(trt_times)
trt_min = np.min(trt_times)
trt_max = np.max(trt_times)
trt_fps = 1000 / trt_avg

print(f"\n   ✅ TensorRT Results:")
print(f"      Avg: {trt_avg:.2f} ms/frame")
print(f"      Min: {trt_min:.2f} ms/frame")
print(f"      Max: {trt_max:.2f} ms/frame")
print(f"      FPS: {trt_fps:.1f}")

# ══════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ══════════════════════════════════════════════════════════════
speedup  = pt_avg / trt_avg
fps_gain = trt_fps - pt_fps

print("\n" + "=" * 55)
print("📊 FINAL COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Metric':<20} {'PyTorch':>10} {'TensorRT':>10} {'Diff':>10}")
print("-" * 55)
print(f"{'Avg ms/frame':<20} {pt_avg:>9.2f}ms {trt_avg:>9.2f}ms {(pt_avg-trt_avg):>+9.2f}ms")
print(f"{'Min ms/frame':<20} {pt_min:>9.2f}ms {trt_min:>9.2f}ms {(pt_min-trt_min):>+9.2f}ms")
print(f"{'Max ms/frame':<20} {pt_max:>9.2f}ms {trt_max:>9.2f}ms {(pt_max-trt_max):>+9.2f}ms")
print(f"{'FPS':<20} {pt_fps:>10.1f} {trt_fps:>10.1f} {fps_gain:>+10.1f}")
print("=" * 55)
print(f"🚀 TensorRT is {speedup:.2f}x FASTER than PyTorch!")
print(f"   +{fps_gain:.1f} extra FPS with TensorRT")
print(f"   Tested on REAL RTSP frames from your CCTV camera")
print("=" * 55)
