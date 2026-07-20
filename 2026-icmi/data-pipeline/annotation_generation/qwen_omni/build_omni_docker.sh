#!/bin/bash
set -e
echo "Building Schwan Qwen2.5-Omni Docker Image..."
docker build -f vlm/Dockerfile.omni -t schwan_omni:latest .
echo "Build Complete!"
