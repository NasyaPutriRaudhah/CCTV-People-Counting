import time
import psutil
import subprocess
import json
from datetime import datetime

LOG_FILE = "stress_test_log.json"
DURATION = 1800  # 30 menit
INTERVAL = 10    # catat setiap 10 detik

logs = []
start_time = time.time()
start_mem = psutil.virtual_memory().used / (1024*1024)

print("=== STRESS TEST MONITOR STARTED ===")
print(f"Duration: {DURATION//60} minutes | Interval: {INTERVAL}s")
print(f"Start memory: {start_mem:.1f} MB\n")

try:
    while True:
        elapsed = time.time() - start_time
        if elapsed >= DURATION:
            break

        # CPU & Memory
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        mem_used = mem.used / (1024*1024)
        mem_growth = mem_used - start_mem

        # GPU (Jetson pakai tegrastats)
        try:
            result = subprocess.run(['tegrastats', '--interval', '1000', '--stop'],
                                    capture_output=True, text=True, timeout=2)
            gpu_line = result.stdout.strip()
        except:
            gpu_line = "N/A"

        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "elapsed_s": round(elapsed),
            "cpu_pct": cpu,
            "mem_used_mb": round(mem_used, 1),
            "mem_growth_mb": round(mem_growth, 1),
            "gpu_raw": gpu_line
        }
        logs.append(entry)

        print(f"[{entry['timestamp']}] CPU={cpu:.1f}% | "
              f"MEM={mem_used:.0f}MB (growth={mem_growth:+.0f}MB)")

        # Save log setiap iterasi
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)

        time.sleep(INTERVAL - 1)

except KeyboardInterrupt:
    print("\nMonitor stopped by user.")

finally:
    # Final report
    if logs:
        mem_values = [l["mem_growth_mb"] for l in logs]
        max_growth = max(mem_values)
        leak_detected = max_growth > 200  # >200MB growth = warning

        print("\n========== STRESS TEST REPORT ==========")
        print(f"Duration monitored : {round(elapsed/60, 1)} minutes")
        print(f"Memory growth      : {max_growth:+.1f} MB  "
              f"{'⚠️  WARNING leak!' if leak_detected else '✅ Stable'}")
        print(f"Max CPU            : {max(l['cpu_pct'] for l in logs):.1f}%")
        print(f"Log saved to       : {LOG_FILE}")
        print("=========================================") 
