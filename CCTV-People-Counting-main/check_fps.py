import cv2, time, numpy as np
from models import load_models
from config import VIDEO1_PATH, DEVICE

model1, _ = load_models()

cap = cv2.VideoCapture(VIDEO1_PATH)
times = []

print("Measuring ms/frame... (500 frames)")
for i in range(500):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    t_start = time.time()
    model1.predict(frame, device=DEVICE, classes=[0], verbose=False)
    t_end = time.time()

    ms = (t_end - t_start) * 1000
    times.append(ms)
    if (i+1) % 50 == 0:
        print(f"  Frame {i+1}/500 | {ms:.1f} ms | avg so far: {np.mean(times):.1f} ms")

cap.release()

arr = np.array(times)
print(f"\n===== ms/frame RESULTS (500 frames) =====")
print(f"Mean   : {np.mean(arr):.2f} ms  --> {'✅ PASS' if np.mean(arr) < 40 else '❌ FAIL'} (target < 40ms)")
print(f"Median : {np.median(arr):.2f} ms")
print(f"Min    : {np.min(arr):.2f} ms")
print(f"Max    : {np.max(arr):.2f} ms")
print(f"P95    : {np.percentile(arr, 95):.2f} ms")
print(f"FPS    : {1000/np.mean(arr):.1f}")
print(f"==========================================")
