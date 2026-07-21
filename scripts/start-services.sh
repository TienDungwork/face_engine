#!/bin/bash

# Script to start all services with proper logging and monitoring

echo "🚀 Starting Face Engine Services..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping services..."
    docker-compose down
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start services with logging
echo "📋 Starting main face engine service..."
docker-compose up -d demo-atin-face-engine

# Wait a bit for main service to initialize
echo "⏳ Waiting for main service to initialize..."
sleep 10

echo "📋 Starting centroid updater service..."
docker-compose up -d centroid-updater

echo "✅ All services started successfully!"
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "📝 To view logs:"
echo "  - Main service: docker-compose logs -f demo-atin-face-engine"
echo "  - Centroid updater: docker-compose logs -f centroid-updater"
echo "  - All services: docker-compose logs -f"
echo ""
echo "🛑 Press Ctrl+C to stop all services"

# Follow logs from all services
docker-compose logs -f
