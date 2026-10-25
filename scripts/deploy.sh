#!/bin/bash
# Shadow AI - Production Deployment Script
# Usage: ./scripts/deploy.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check .env file
    if [ ! -f "$ENV_FILE" ]; then
        log_warning ".env file not found, creating from template..."
        cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
        log_warning "Please edit .env file with your configuration"
        exit 1
    fi
    
    log_success "Requirements satisfied"
}

create_directories() {
    log_info "Creating required directories..."
    
    mkdir -p "$PROJECT_ROOT/data/memory"
    mkdir -p "$PROJECT_ROOT/data/archival"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/configs/grafana/provisioning/datasources"
    mkdir -p "$PROJECT_ROOT/configs/grafana/provisioning/dashboards"
    mkdir -p "$PROJECT_ROOT/configs/grafana/dashboards"
    
    log_success "Directories created"
}

pull_images() {
    log_info "Pulling Docker images..."
    
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" pull
    
    log_success "Images pulled"
}

build_images() {
    log_info "Building Docker images..."
    
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build --no-cache
    
    log_success "Images built"
}

start_services() {
    log_info "Starting services..."
    
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
    
    log_success "Services started"
    
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    docker-compose -f "$COMPOSE_FILE" ps
}

stop_services() {
    log_info "Stopping services..."
    
    docker-compose -f "$COMPOSE_FILE" down
    
    log_success "Services stopped"
}

restart_services() {
    log_info "Restarting services..."
    
    stop_services
    start_services
    
    log_success "Services restarted"
}

view_logs() {
    local service="$1"
    
    if [ -z "$service" ]; then
        docker-compose -f "$COMPOSE_FILE" logs -f
    else
        docker-compose -f "$COMPOSE_FILE" logs -f "$service"
    fi
}

check_health() {
    log_info "Checking service health..."
    
    echo ""
    echo "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo "API Health Check:"
    curl -s http://localhost:8000/health || log_warning "API not responding"
    
    echo ""
    echo "Redis Health Check:"
    docker exec shadow-cache redis-cli ping || log_warning "Redis not responding"
    
    echo ""
    echo "PostgreSQL Health Check:"
    docker exec shadow-db pg_isready -U shadow || log_warning "PostgreSQL not responding"
    
    echo ""
    echo "Prometheus Health Check:"
    curl -s http://localhost:9090/-/healthy || log_warning "Prometheus not responding"
    
    echo ""
    echo "Loki Health Check:"
    curl -s http://localhost:3100/ready || log_warning "Loki not responding"
}

backup_data() {
    log_info "Backing up data..."
    
    BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup PostgreSQL
    log_info "Backing up PostgreSQL..."
    docker exec shadow-db pg_dump -U shadow shadow > "$BACKUP_DIR/shadow_db.sql"
    
    # Backup Redis
    log_info "Backing up Redis..."
    docker exec shadow-cache redis-cli BGSAVE
    docker cp shadow-cache:/data/dump.rdb "$BACKUP_DIR/redis_dump.rdb"
    
    # Backup data directory
    log_info "Backing up data directory..."
    cp -r "$PROJECT_ROOT/data" "$BACKUP_DIR/"
    
    log_success "Backup completed: $BACKUP_DIR"
}

restore_data() {
    log_error "Restore not implemented yet"
    exit 1
}

update_services() {
    log_info "Updating services..."
    
    git pull origin main
    
    build_images
    
    restart_services
    
    check_health
    
    log_success "Update completed"
}

clean_up() {
    log_info "Cleaning up..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove old logs
    find "$PROJECT_ROOT/logs" -type f -name "*.log" -mtime +30 -delete
    
    log_success "Cleanup completed"
}

show_usage() {
    echo "Shadow AI - Deployment Script"
    echo ""
    echo "Usage: ./scripts/deploy.sh [command]"
    echo ""
    echo "Commands:"
    echo "  check      Check requirements"
    echo "  init       Initialize directories"
    echo "  build      Build Docker images"
    echo "  pull       Pull Docker images"
    echo "  start      Start all services"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  logs       View logs (optional: service name)"
    echo "  health     Check service health"
    echo "  backup     Backup data"
    echo "  restore    Restore data (not implemented)"
    echo "  update     Update and restart services"
    echo "  clean      Clean up unused resources"
    echo "  all        Complete deployment (init, build, start)"
    echo ""
}

case "${1:-all}" in
    check)
        check_requirements
        ;;
    init)
        check_requirements
        create_directories
        ;;
    build)
        check_requirements
        build_images
        ;;
    pull)
        pull_images
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        view_logs "$2"
        ;;
    health)
        check_health
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data
        ;;
    update)
        update_services
        ;;
    clean)
        clean_up
        ;;
    all)
        check_requirements
        create_directories
        build_images
        start_services
        check_health
        ;;
    *)
        show_usage
        exit 1
        ;;
esac