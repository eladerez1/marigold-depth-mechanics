# ACR image for marigold_depth_mechanics. Clean CUDA base (not nvcr.io/nvidia/pytorch).
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git build-essential \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip

# PyTorch before other deps (V100 / A6000, CUDA 12.1).
RUN python3 -m pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

COPY scripts/acr/requirements-docker.txt /tmp/requirements-docker.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY . /workspace
WORKDIR /workspace
ENV PYTHONPATH=/workspace:/workspace/third_party/Marigold
