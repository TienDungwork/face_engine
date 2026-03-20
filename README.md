# Face Recognition Service

A high-performance face recognition service built with FastAPI and PyTorch, featuring face detection and face comparison capabilities.

## Features

- Face Detection and Recognition
- Face Comparison
- GPU Acceleration Support
- Rate Limiting
- Docker Support with CUDA
- Load Balancing with Traefik (future)

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- NVIDIA Container Toolkit (for GPU support)

## Installation

1. Clone the repository:

```bash
git clone https://gitlab.com/atin-t4/lpb/recognize-serivce.git
cd recognize-service
```

2. Create and configure the environment file:

```bash
cp env.example resources/.env
# Edit resources/.env with your configurations
```

3. Build and run with Docker Compose:

```bash
docker-compose up -d
```

## API Documentation

- Go to `http://localhost:8000/docs` to see the API documentation.

### Face Detection API

```bash
POST /api/v1/analyze/detect
```

Request body:

```json
{
  "base64_image": "string", // Base64 encoded image
  "url_image": "string" // Or URL to image
}
```

Response:

```json
{
    "status_code": 200,
    "message": "Success",
    "data": {
        "bbox": [x1, y1, x2, y2],
        "feature": []  // Face embedding vector
    }
}
```

### Face Comparison API

```bash
POST /api/v1/analyze/compare
```

Request body:

```json
{
  "base64_1": "string", // First face image in base64
  "base64_2": "string" // Second face image in base64
}
```

## Contact

- For any questions or feedback, please contact me at [huyquangbka@gmail.com](mailto:huyquangbka@gmail.com).
