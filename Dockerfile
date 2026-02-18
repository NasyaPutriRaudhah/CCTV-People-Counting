# Dockerfile — CCTV People Counter
# Optimized for NVIDIA Jetson Orin Nano (ARM64 + CUDA)
#
# NVIDIA L4T base image provides:
#   - ARM64 architecture support
#   - CUDA & cuDNN pre-installed
#   - PyTorch with GPU acceleration
#
# Check your JetPack version before changing the base image:
#   $ cat /etc/nv_tegra_release
#
#   JetPack 6.x → r36.x.x  (use l4t-pytorch:r36.2.0-pth2.1-py3)  ← Orin Nano default
#   JetPack 5.x → r35.x.x  (use l4t-pytorch:r35.4.1-pth2.1-py3)
#
# Full list of available tags:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch

FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

# ------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # psycopg2 build dependencies (no binary wheel for ARM64)
    libpq-dev \
    gcc \
    # GStreamer — required for NVIDIA-accelerated camera capture on Jetson
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    python3-gst-1.0 \
    # Video device access
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Python dependencies
# ------------------------------------------------------------------
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------
# OpenCV — use the host's CUDA-accelerated version, NOT pip's
#
# The Jetson ships with OpenCV built against CUDA at:
#   /usr/lib/python3/dist-packages/cv2/
#
# We add it to the Python path so the container finds it automatically.
# This is handled in docker-compose.yml via a volume mount.
# ------------------------------------------------------------------
ENV PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH}"

# ------------------------------------------------------------------
# Copy application source
# ------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------
# GStreamer pipeline environment for Jetson camera capture
# Override CAP_BACKEND in your code if needed:
#   cv2.VideoCapture("nvarguscamerasrc ! ...", cv2.CAP_GSTREAMER)
# ------------------------------------------------------------------
ENV OPENCV_VIDEOIO_PRIORITY_GSTREAMER=1

CMD ["python3", "main.py"]