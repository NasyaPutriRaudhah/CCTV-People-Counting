FROM ultralytics/ultralytics:latest-jetson-jetpack6

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-gst-1.0 \
    v4l-utils \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    libxkbcommon0 \
    libdbus-1-3 \
    libsm6 \
    libice6 \
    libxt6 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# ── OpenCV font fix ────────────────────────────────────────────────────────────
RUN mkdir -p /usr/local/lib/python3.10/dist-packages/cv2/qt/fonts && \
    find /usr/share/fonts -name '*.ttf' | xargs -I{} cp {} \
    /usr/local/lib/python3.10/dist-packages/cv2/qt/fonts/ 2>/dev/null || true

# ── Python dependencies ────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    "numpy<2" \
    supervision \
    shapely \
    psycopg2-binary \
    cvzone \
    flask==3.0.3      # ← NEW: web dashboard

# ── App setup ──────────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

# ── Environment variables ──────────────────────────────────────────────────────
ENV OPENCV_VIDEOIO_PRIORITY_GSTREAMER=1
ENV QT_QPA_PLATFORM=xcb
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/venv/lib/python3.10/site-packages/cv2/qt/plugins
ENV DISPLAY=:0
ENV PYTHONPATH="/usr/lib/python3.10/dist-packages:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1

# ── Flask dashboard port ───────────────────────────────────────────────────────
# No EXPOSE needed — cctv_app uses network_mode: host

CMD ["python3", "-u", "main.py"]
