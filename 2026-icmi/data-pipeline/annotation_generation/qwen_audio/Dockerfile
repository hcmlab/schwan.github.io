# Use the latest NVIDIA PyTorch Container as base
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Qwen-Audio specific python dependencies
RUN pip install --no-cache-dir tiktoken transformers accelerate

# If auto-gptq or other quantization libraries are needed:
# RUN pip install auto-gptq optimum

# Command to run by default. Assumes volume mounted at /data/Schwan_T3_Clean
# Entry script should write out to Schwan_T3_Clean/.../annotation_server/audio_annotations.json
CMD ["/bin/bash"]
