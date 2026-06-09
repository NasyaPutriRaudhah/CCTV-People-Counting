#!/bin/bash
echo "=== Alert Monitor Started: $(date) ===" >> alert_log.txt

while true; do
    # Cek container masih jalan
    if ! docker ps | grep -q cctv_app; then
        echo "$(date) ⚠️  ALERT: cctv_app container DOWN!" >> alert_log.txt
        echo "$(date) ⚠️  ALERT: cctv_app container DOWN!"
    else
        echo "$(date) ✅ cctv_app running" >> alert_log.txt
    fi

    # Cek memory usage
    MEM=$(docker stats cctv_app --no-stream --format "{{.MemPerc}}" | tr -d '%')
    if (( $(echo "$MEM > 80" | bc -l) )); then
        echo "$(date) ⚠️  ALERT: Memory tinggi! ${MEM}%" >> alert_log.txt
    fi

    # Cek reconnection errors
    ERRORS=$(docker logs cctv_app 2>/dev/null | tail -50 | grep "Too many errors" | wc -l)
    if [ "$ERRORS" -gt 2 ]; then
        echo "$(date) ⚠️  ALERT: RTSP sering disconnect! ($ERRORS kali)" >> alert_log.txt
    fi

    sleep 60  # cek setiap 1 menit
done 
