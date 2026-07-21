# ==========================================================
# Base image có sẵn CUDA 11.8 + cuDNN 8
# ==========================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ==========================================================
# Cài python + pip
# ==========================================================
RUN apt update && apt install -y \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python default
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# ==========================================================
# Set working directory
# ==========================================================
WORKDIR /project

# ==========================================================
# Copy requirements trước để tận dụng docker cache
# ==========================================================
COPY requirements.txt .

# ==========================================================
# Cài ONNXRuntime GPU bản phù hợp CUDA 11.8
# ==========================================================
RUN pip install --no-cache-dir onnxruntime-gpu==1.16.3

# ==========================================================
# Cài các thư viện khác
# ==========================================================
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================================
# Copy source code
# ==========================================================
#COPY . .

# ==========================================================
# Expose port nếu cần (ví dụ FastAPI)
# ==========================================================
#EXPOSE 42075

# ==========================================================
# Run app
# =====================================
