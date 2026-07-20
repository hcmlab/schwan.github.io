# Use the latest NVIDIA PyTorch Container as base
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg for video/audio, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Qwen-Omni specific python dependencies
RUN pip install --no-cache-dir transformers tiktoken accelerate \
    decord opencv-python Pillow

# Entry point for Qwen Omni multi-modal analysis
# Output should strictly be written to Schwan_T3_Clean/.../annotation_server/omni_annotations.json
CMD ["/bin/bash"]
