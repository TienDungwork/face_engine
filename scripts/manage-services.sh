#!/bin/bash

# Script to manage face engine services

case "$1" in
    start)
        echo "🚀 Starting all services..."
        docker-compose up -d
        echo "✅ Services started. Use 'logs' command to view output."
        ;;
    stop)
        echo "🛑 Stopping all services..."
        docker-compose down
        echo "✅ Services stopped."
        ;;
    restart)
        echo "🔄 Restarting all services..."
        docker-compose down
        docker-compose up -d
        echo "✅ Services restarted."
        ;;
    logs)
        echo "📝 Viewing logs for all services..."
        docker-compose logs -f
        ;;
    logs-main)
        echo "📝 Viewing logs for main service..."
        docker-compose logs -f demo-atin-face-engine
        ;;
    logs-centroid)
        echo "📝 Viewing logs for centroid updater..."
        docker-compose logs -f centroid-updater
        ;;
    status)
        echo "📊 Service Status:"
        docker-compose ps
        ;;
    manual-update)
        echo "🔄 Running manual centroid update..."
        docker-compose exec centroid-updater python3 update_centroid.py --manual
        ;;
    shell-main)
        echo "🐚 Opening shell in main service..."
        docker-compose exec demo-atin-face-engine /bin/bash
        ;;
    shell-centroid)
        echo "🐚 Opening shell in centroid updater service..."
        docker-compose exec centroid-updater /bin/bash
        ;;
    build)
        echo "🔨 Building services..."
        docker-compose build
        echo "✅ Build completed."
        ;;
    clean)
        echo "🧹 Cleaning up containers and volumes..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        echo "✅ Cleanup completed."
        ;;
    *)
        echo "Face Engine Service Manager"
        echo ""
        echo "Usage: $0 {start|stop|restart|logs|logs-main|logs-centroid|status|manual-update|shell-main|shell-centroid|build|clean}"
        echo ""
        echo "Commands:"
        echo "  start          - Start all services"
        echo "  stop           - Stop all services"
        echo "  restart        - Restart all services"
        echo "  logs           - View logs from all services"
        echo "  logs-main      - View logs from main service only"
        echo "  logs-centroid  - View logs from centroid updater only"
        echo "  status         - Show service status"
        echo "  manual-update  - Run manual centroid update"
        echo "  shell-main     - Open shell in main service"
        echo "  shell-centroid - Open shell in centroid updater service"
        echo "  build          - Build services"
        echo "  clean          - Clean up containers and volumes"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs"
        echo "  $0 manual-update"
        echo "  $0 status"
        ;;
esac
