#!/bin/bash

# AI Trading System V2 - Automated Installation Script
# Installs and configures the complete trading system for production use

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
PROJECT_NAME="ai-trading-system-v2"
INSTALL_DIR="/opt/${PROJECT_NAME}"
LOG_DIR="/var/log/${PROJECT_NAME}"
DATA_DIR="/var/lib/${PROJECT_NAME}"
USER="trading"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    elif type lsb_release >/dev/null 2>&1; then
        OS=$(lsb_release -si)
        VER=$(lsb_release -sr)
    else
        print_error "Cannot detect OS"
        exit 1
    fi
    
    print_status "Detected OS: $OS $VER"
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
        apt-get update
        apt-get install -y \
            python3.11 \
            python3.11-dev \
            python3.11-venv \
            python3-pip \
            build-essential \
            libta-lib-dev \
            curl \
            wget \
            git \
            nginx \
            redis-server \
            postgresql \
            postgresql-contrib \
            supervisor \
            cron \
            logrotate
    elif [[ $OS == *"CentOS"* ]] || [[ $OS == *"Red Hat"* ]]; then
        yum update -y
        yum install -y \
            python311 \
            python311-devel \
            python3-pip \
            gcc \
            gcc-c++ \
            make \
            curl \
            wget \
            git \
            nginx \
            redis \
            postgresql \
            postgresql-server \
            supervisor \
            cronie
    else
        print_error "Unsupported OS: $OS"
        exit 1
    fi
    
    print_success "System dependencies installed"
}

# Create system user
create_user() {
    print_status "Creating system user: $USER"
    
    if ! id "$USER" &>/dev/null; then
        useradd -r -m -s /bin/bash -d /home/$USER $USER
        usermod -aG redis $USER
        print_success "User $USER created"
    else
        print_warning "User $USER already exists"
    fi
}

# Create directories
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p $INSTALL_DIR
    mkdir -p $LOG_DIR
    mkdir -p $DATA_DIR/{models,backups,cache}
    mkdir -p /etc/${PROJECT_NAME}
    
    chown -R $USER:$USER $INSTALL_DIR
    chown -R $USER:$USER $LOG_DIR
    chown -R $USER:$USER $DATA_DIR
    
    print_success "Directories created"
}

# Install Python dependencies
install_python_dependencies() {
    print_status "Setting up Python environment..."
    
    # Create virtual environment
    sudo -u $USER python3.11 -m venv $INSTALL_DIR/venv
    
    # Activate and install dependencies
    sudo -u $USER $INSTALL_DIR/venv/bin/pip install --upgrade pip
    sudo -u $USER $INSTALL_DIR/venv/bin/pip install -r requirements.txt
    
    print_success "Python environment ready"
}

# Install application
install_application() {
    print_status "Installing application..."
    
    # Copy application files
    cp -r src/ $INSTALL_DIR/
    cp -r config/ $INSTALL_DIR/
    cp requirements.txt $INSTALL_DIR/
    cp setup.py $INSTALL_DIR/
    
    # Set permissions
    chown -R $USER:$USER $INSTALL_DIR
    chmod +x $INSTALL_DIR/src/api/trading_api.py
    
    print_success "Application installed"
}

# Configure services
configure_services() {
    print_status "Configuring services..."
    
    # Redis configuration
    cat > /etc/redis/redis.conf << EOF
bind 127.0.0.1
port 6379
timeout 0
tcp-keepalive 60
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile /var/log/redis/redis-server.log
databases 16
save 900 1
save 300 10
save 60 10000
rdbcompression yes
dbfilename dump.rdb
dir /var/lib/redis
requirepass $(openssl rand -base64 32)
EOF

    # PostgreSQL setup
    if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
        sudo -u postgres createdb trading_system || true
        sudo -u postgres psql -c "CREATE USER trading WITH PASSWORD '$(openssl rand -base64 32)';" || true
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_system TO trading;" || true
    fi
    
    # Supervisor configuration
    cat > /etc/supervisor/conf.d/trading-system.conf << EOF
[program:trading-api]
command=$INSTALL_DIR/venv/bin/python -m src.api.trading_api
directory=$INSTALL_DIR
user=$USER
autostart=true
autorestart=true
stderr_logfile=$LOG_DIR/api_error.log
stdout_logfile=$LOG_DIR/api.log
environment=TRADING_ENV=production

[program:trading-monitor]
command=$INSTALL_DIR/venv/bin/python -m src.monitoring.health_monitor
directory=$INSTALL_DIR
user=$USER
autostart=true
autorestart=true
stderr_logfile=$LOG_DIR/monitor_error.log
stdout_logfile=$LOG_DIR/monitor.log
environment=TRADING_ENV=production
EOF

    # Nginx configuration
    cat > /etc/nginx/sites-available/trading-system << EOF
server {
    listen 80;
    server_name _;
    
    location /api/ {
        proxy_pass http://127.0.0.1:8000/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

    ln -sf /etc/nginx/sites-available/trading-system /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
    
    print_success "Services configured"
}

# Setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    cat > /etc/${PROJECT_NAME}/environment << EOF
# AI Trading System V2 Environment Configuration
# IMPORTANT: Update these values before starting the system

# Environment
TRADING_ENV=production

# MT5 Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# API Security
API_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 64)

# Database
REDIS_PASSWORD=$(openssl rand -base64 32)
POSTGRES_USER=trading
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Email Alerts (optional)
EMAIL_ALERTS_ENABLED=false
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_RECIPIENTS=admin@yourcompany.com

# Telegram Alerts (optional)
TELEGRAM_ALERTS_ENABLED=false
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_IDS=your_chat_id

# Paths
LOG_DIR=$LOG_DIR
DATA_DIR=$DATA_DIR
INSTALL_DIR=$INSTALL_DIR
EOF

    chown $USER:$USER /etc/${PROJECT_NAME}/environment
    chmod 600 /etc/${PROJECT_NAME}/environment
    
    print_success "Environment file created at /etc/${PROJECT_NAME}/environment"
}

# Setup logrotate
setup_logrotate() {
    print_status "Setting up log rotation..."
    
    cat > /etc/logrotate.d/trading-system << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload supervisor
    endscript
}
EOF

    print_success "Log rotation configured"
}

# Setup systemd services
setup_systemd() {
    print_status "Setting up systemd services..."
    
    cat > /etc/systemd/system/trading-system.service << EOF
[Unit]
Description=AI Trading System V2
After=network.target redis.service postgresql.service
Requires=redis.service postgresql.service

[Service]
Type=forking
User=$USER
Group=$USER
WorkingDirectory=$INSTALL_DIR
Environment=TRADING_ENV=production
EnvironmentFile=/etc/${PROJECT_NAME}/environment
ExecStart=$INSTALL_DIR/venv/bin/supervisord -c /etc/supervisor/supervisord.conf
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable trading-system
    systemctl enable redis-server
    systemctl enable postgresql
    systemctl enable nginx
    
    print_success "Systemd services configured"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    systemctl start redis-server
    systemctl start postgresql
    systemctl start nginx
    systemctl start trading-system
    
    # Wait for services to start
    sleep 5
    
    # Check service status
    if systemctl is-active --quiet trading-system; then
        print_success "Trading system started successfully"
    else
        print_error "Failed to start trading system"
        print_status "Check logs: journalctl -u trading-system -f"
    fi
}

# Create management script
create_management_script() {
    print_status "Creating management script..."
    
    cat > /usr/local/bin/trading-system << 'EOF'
#!/bin/bash

# AI Trading System V2 Management Script

case "$1" in
    start)
        systemctl start trading-system
        echo "Trading system started"
        ;;
    stop)
        systemctl stop trading-system
        echo "Trading system stopped"
        ;;
    restart)
        systemctl restart trading-system
        echo "Trading system restarted"
        ;;
    status)
        systemctl status trading-system
        ;;
    logs)
        journalctl -u trading-system -f
        ;;
    api-logs)
        tail -f /var/log/ai-trading-system-v2/api.log
        ;;
    health)
        curl -s http://localhost:8000/api/v1/health | jq .
        ;;
    config)
        nano /etc/ai-trading-system-v2/environment
        ;;
    update)
        cd /opt/ai-trading-system-v2
        git pull
        systemctl restart trading-system
        echo "System updated and restarted"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|api-logs|health|config|update}"
        exit 1
        ;;
esac
EOF

    chmod +x /usr/local/bin/trading-system
    
    print_success "Management script created: trading-system"
}

# Main installation function
main() {
    print_status "Starting AI Trading System V2 installation..."
    
    check_root
    detect_os
    install_system_dependencies
    create_user
    create_directories
    install_python_dependencies
    install_application
    configure_services
    setup_environment
    setup_logrotate
    setup_systemd
    start_services
    create_management_script
    
    print_success "Installation completed successfully!"
    echo
    print_status "Next steps:"
    echo "1. Edit configuration: /etc/${PROJECT_NAME}/environment"
    echo "2. Update MT5 credentials and other settings"
    echo "3. Restart the system: trading-system restart"
    echo "4. Check status: trading-system status"
    echo "5. View API documentation: http://your-server/docs"
    echo "6. Access dashboard: http://your-server/"
    echo
    print_warning "Remember to:"
    echo "- Configure firewall (ports 80, 443)"
    echo "- Setup SSL certificate for production"
    echo "- Set up monitoring and backups"
    echo "- Test all functionality before live trading"
}

# Run installation
main "$@"