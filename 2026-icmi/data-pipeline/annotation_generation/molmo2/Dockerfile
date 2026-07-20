# Use the latest NVIDIA PyTorch Container as base
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install model-specific python dependencies (transformers, flash-attn etc.)
# Note: Flash attention is usually pre-installed in newer NGC Pytorch containers, 
# but we ensure the latest packages required by Molmo are present.
RUN pip install --no-cache-dir einops torchvision peft
RUN pip install --no-cache-dir git+https://github.com/huggingface/transformers.git

# A script to run the Molmo2 pipeline on the dataset
# You will bind mount the script or copy it into the container during development
# COPY scripts/run_molmo2.py /app/run_molmo2.py

# Command to run by default. Assumes volume mounted at /data/Schwan_T3_Clean
# CMD ["python", "run_molmo2.py", "--data_dir", "/data/Schwan_T3_Clean"]
CMD ["/bin/bash"]
