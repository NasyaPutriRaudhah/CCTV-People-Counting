"""
wifi_signal.py — reads WiFi signal from Linux (Jetson Orin Nano)
Returns signal as 0-100%.
"""

import subprocess
import re


def get_wifi_signal_dbm() -> int:
    """Returns WiFi signal in dBm (e.g. -55). Returns -100 if unavailable."""
    # Method 1: /proc/net/wireless
    try:
        with open("/proc/net/wireless", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("wlan", "wlp")):
                    parts = line.split()
                    signal = int(float(parts[3].rstrip(".")))
                    if signal > 0:
                        signal = signal - 256
                    return signal
    except Exception:
        pass

    # Method 2: iwconfig fallback
    try:
        result = subprocess.run(["iwconfig"], capture_output=True, text=True, timeout=2)
        match = re.search(r"Signal level=(-?\d+)\s*dBm", result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass

    return -100  # unknown


def get_wifi_signal_percent() -> int:
    """Convert dBm to 0-100% for Grafana gauge panel."""
    dbm = get_wifi_signal_dbm()
    clamped = max(-90, min(-30, dbm))
    return int((clamped + 90) / 60 * 100)
