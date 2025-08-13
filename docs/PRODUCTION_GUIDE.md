"""
Professional Documentation for AI Trading System V2
Complete user and technical guides
"""

# AI Trading System V2 - Professional Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation Guide](#installation-guide)
3. [Configuration Guide](#configuration-guide)
4. [API Documentation](#api-documentation)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Trading Operations](#trading-operations)
7. [Troubleshooting](#troubleshooting)
8. [Security Considerations](#security-considerations)
9. [Performance Optimization](#performance-optimization)
10. [Maintenance & Updates](#maintenance--updates)

## System Overview

The AI Trading System V2 is a professional-grade automated trading platform that combines:

- **Multi-Strategy AI Engine**: 5 concurrent trading strategies with AI ensemble
- **Real-Time Monitoring**: Comprehensive health monitoring and alerting
- **Professional API**: RESTful API for remote control and integration
- **Production Infrastructure**: Docker containerization and automated deployment
- **Enterprise Security**: Production-grade security and access controls

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading API   │    │  Health Monitor │    │  Alert System  │
│   (Port 8000)   │    │  (Background)   │    │  (Background)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │             Core Trading System                 │
         │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
         │  │ Data Pipeline│  │ AI Engine   │  │Strategies│ │
         │  └─────────────┘  └─────────────┘  └──────────┘ │
         └─────────────────────────────────────────────────┘
                                 │
         ┌─────────────────────────────────────────────────┐
         │              Infrastructure                     │
         │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
         │  │  Redis   │  │PostgreSQL│  │  MetaTrader5 │   │
         │  └──────────┘  └──────────┘  └──────────────┘   │
         └─────────────────────────────────────────────────┘
```

## Installation Guide

### Quick Start (Docker - Recommended)

1. **Prerequisites**
   ```bash
   # Install Docker and Docker Compose
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

2. **Clone and Configure**
   ```bash
   git clone https://github.com/santiagoossa-maker/ai-trading-system-v2.git
   cd ai-trading-system-v2
   
   # Copy environment template
   cp .env.example .env
   
   # Edit configuration
   nano .env
   ```

3. **Deploy**
   ```bash
   # Production deployment
   docker-compose up -d
   
   # Check status
   docker-compose ps
   ```

### Manual Installation (Linux)

For manual installation on Linux servers:

```bash
# Download and run installation script
wget https://raw.githubusercontent.com/santiagoossa-maker/ai-trading-system-v2/main/scripts/install.sh
chmod +x install.sh
sudo ./install.sh
```

The script automatically:
- Installs all dependencies
- Creates system user and directories
- Configures services (Redis, PostgreSQL, Nginx)
- Sets up systemd services
- Creates management commands

## Configuration Guide

### Environment Variables

The system uses environment variables for configuration. Key variables:

```bash
# Trading Configuration
TRADING_ENV=production
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# API Security
API_KEY=your_secure_api_key
JWT_SECRET=your_jwt_secret

# Database
REDIS_PASSWORD=secure_redis_password
POSTGRES_PASSWORD=secure_postgres_password

# Alerts
EMAIL_ALERTS_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
TELEGRAM_ALERTS_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token
```

### Configuration Files

1. **Production Configuration** (`config/production.yaml`)
   - Enterprise-grade settings
   - Security-first configuration
   - Performance optimization

2. **Development Configuration** (`config/development.yaml`)
   - Safe development settings
   - Relaxed security for testing
   - Verbose logging

### Trading Parameters

Configure risk management in `config/production.yaml`:

```yaml
trading:
  max_daily_loss_percent: 5.0      # Maximum daily loss
  max_concurrent_positions: 10     # Position limit
  position_size_percent: 2.0       # Risk per trade
  emergency_stop_loss_percent: 10.0 # Emergency stop
```

## API Documentation

### Authentication

All API endpoints require authentication via API key:

```bash
curl -H "X-API-Key: your_api_key" http://localhost:8000/api/v1/status
```

### Core Endpoints

#### System Control

```bash
# Get system status
GET /api/v1/status

# Start trading
POST /api/v1/control
{
  "action": "start",
  "force": false
}

# Stop trading
POST /api/v1/control
{
  "action": "stop",
  "force": true
}
```

#### Health Monitoring

```bash
# Get health status
GET /api/v1/health

# Get detailed metrics
GET /api/v1/health/detailed
```

#### Trading Operations

```bash
# Get active positions
GET /api/v1/positions

# Place order
POST /api/v1/order
{
  "symbol": "R_75",
  "action": "BUY",
  "volume": 0.01,
  "sl": 1000,
  "tp": 2000
}

# Close position
DELETE /api/v1/position/12345

# Emergency close all
DELETE /api/v1/positions
```

#### Performance Analytics

```bash
# Get performance metrics
GET /api/v1/performance

# Get performance history
GET /api/v1/performance/history?days=30
```

### Swagger Documentation

Interactive API documentation available at:
- **Production**: `https://your-domain/docs`
- **Development**: `http://localhost:8000/docs`

## Monitoring & Alerting

### Health Monitoring

The system continuously monitors:

- **System Resources**: CPU, memory, disk usage
- **Network**: Connectivity and performance
- **Application**: Process health and performance
- **Trading**: MT5 connectivity and trading status
- **Database**: Redis and PostgreSQL health

### Alert Channels

1. **Email Alerts**
   - SMTP configuration required
   - HTML formatted alerts
   - Detailed system information

2. **Telegram Notifications**
   - Real-time mobile alerts
   - Bot token required
   - Markdown formatted messages

3. **Webhook Integration**
   - Custom webhook endpoints
   - JSON payload format
   - For integration with external systems

### Alert Levels

- **INFO**: System information and status updates
- **WARNING**: Non-critical issues requiring attention
- **CRITICAL**: Serious issues requiring immediate action

### Example Alert Configuration

```yaml
monitoring:
  notifications:
    email:
      enabled: true
      smtp_host: smtp.gmail.com
      username: alerts@yourcompany.com
      to_addresses: ["admin@yourcompany.com"]
    
    telegram:
      enabled: true
      bot_token: "your_bot_token"
      chat_ids: ["your_chat_id"]
```

## Trading Operations

### Starting the System

1. **Verify Configuration**
   ```bash
   # Check configuration
   trading-system config
   
   # Test connections
   curl -H "X-API-Key: your_key" http://localhost:8000/api/v1/health
   ```

2. **Start Trading**
   ```bash
   # Via management script
   trading-system start
   
   # Via API
   curl -X POST -H "X-API-Key: your_key" \
     -H "Content-Type: application/json" \
     -d '{"action": "start"}' \
     http://localhost:8000/api/v1/control
   ```

### Monitoring Operations

1. **Real-Time Status**
   ```bash
   # System status
   trading-system status
   
   # View logs
   trading-system logs
   
   # API logs specifically
   trading-system api-logs
   ```

2. **Web Dashboard**
   - Access at `http://your-server/`
   - Real-time charts and metrics
   - Position management
   - System controls

### Emergency Procedures

1. **Emergency Stop**
   ```bash
   # Stop all trading immediately
   curl -X POST -H "X-API-Key: your_key" \
     -H "Content-Type: application/json" \
     -d '{"action": "stop", "force": true}' \
     http://localhost:8000/api/v1/control
   
   # Close all positions
   curl -X DELETE -H "X-API-Key: your_key" \
     http://localhost:8000/api/v1/positions
   ```

## Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   ```bash
   # Check MT5 credentials
   trading-system config
   
   # Verify MT5 is running
   ps aux | grep terminal
   
   # Check MT5 logs
   tail -f ~/.wine/drive_c/users/trading/AppData/Roaming/MetaQuotes/Terminal/*/Logs/*.log
   ```

2. **API Not Responding**
   ```bash
   # Check API service
   systemctl status trading-system
   
   # Check port availability
   netstat -tulpn | grep 8000
   
   # Restart API
   trading-system restart
   ```

3. **Database Connection Issues**
   ```bash
   # Check Redis
   redis-cli ping
   
   # Check PostgreSQL
   sudo -u postgres psql -c "SELECT 1;"
   
   # Check credentials
   grep -i password /etc/ai-trading-system-v2/environment
   ```

### Log Analysis

1. **System Logs**
   ```bash
   # Application logs
   tail -f /var/log/ai-trading-system-v2/api.log
   
   # Error logs
   tail -f /var/log/ai-trading-system-v2/api_error.log
   
   # System journal
   journalctl -u trading-system -f
   ```

2. **Performance Logs**
   ```bash
   # Health metrics
   curl -H "X-API-Key: your_key" \
     http://localhost:8000/api/v1/health/detailed | jq .
   ```

### Recovery Procedures

1. **System Recovery**
   ```bash
   # Full system restart
   trading-system stop
   systemctl restart redis-server
   systemctl restart postgresql
   trading-system start
   ```

2. **Database Recovery**
   ```bash
   # Redis recovery
   systemctl stop redis-server
   redis-check-rdb /var/lib/redis/dump.rdb
   systemctl start redis-server
   
   # PostgreSQL recovery
   sudo -u postgres pg_resetwal /var/lib/postgresql/data
   ```

## Security Considerations

### Production Security Checklist

- [ ] Strong API keys and JWT secrets
- [ ] Secure database passwords
- [ ] Firewall configuration (ports 80, 443 only)
- [ ] SSL certificate installation
- [ ] Regular security updates
- [ ] Log monitoring for suspicious activity
- [ ] Backup encryption

### Access Control

1. **API Security**
   - API key authentication
   - Rate limiting
   - IP whitelisting (optional)

2. **Database Security**
   - Password authentication
   - Network encryption
   - Regular password rotation

3. **System Security**
   - Dedicated system user
   - Minimal permissions
   - Secure file permissions

## Performance Optimization

### System Requirements

**Minimum Requirements:**
- 2 vCPU
- 4 GB RAM
- 20 GB SSD storage
- 1 Gbps network

**Recommended Requirements:**
- 4 vCPU
- 8 GB RAM
- 50 GB SSD storage
- 1 Gbps network

### Optimization Settings

1. **Database Optimization**
   ```yaml
   # Redis optimization
   redis:
     max_memory: 2gb
     maxmemory_policy: allkeys-lru
   
   # PostgreSQL optimization
   postgresql:
     shared_buffers: 256MB
     effective_cache_size: 1GB
   ```

2. **Application Optimization**
   ```yaml
   performance:
     max_worker_threads: 8
     data_buffer_size: 10000
     cache_ttl_seconds: 300
   ```

### Monitoring Performance

1. **Key Metrics**
   - API response time (<1000ms)
   - Health check duration (<30s)
   - Memory usage (<80%)
   - CPU usage (<70%)

2. **Performance Alerts**
   - Automatic alerts for degraded performance
   - Threshold-based notifications
   - Trend analysis

## Maintenance & Updates

### Regular Maintenance

1. **Daily Tasks**
   - Check system health
   - Review alert notifications
   - Monitor trading performance
   - Verify backup completion

2. **Weekly Tasks**
   - Review system logs
   - Update monitoring dashboards
   - Check security patches
   - Performance analysis

3. **Monthly Tasks**
   - Full system backup verification
   - Security audit
   - Performance optimization review
   - Documentation updates

### Update Procedures

1. **System Updates**
   ```bash
   # Update system packages
   apt update && apt upgrade
   
   # Update application
   trading-system update
   ```

2. **Configuration Updates**
   ```bash
   # Edit configuration
   trading-system config
   
   # Reload configuration
   trading-system restart
   ```

### Backup & Recovery

1. **Automated Backups**
   - Daily configuration backups
   - Weekly database backups
   - Monthly full system backups

2. **Backup Verification**
   ```bash
   # Test backup integrity
   tar -tzf /path/to/backup.tar.gz
   
   # Test database backup
   pg_restore --list backup_file.dump
   ```

---

## Support & Contact

For technical support and questions:

- **Documentation**: Check this guide first
- **API Reference**: `/docs` endpoint
- **Health Check**: `/api/v1/health` endpoint
- **Logs**: Use `trading-system logs` command

Remember to always test changes in a development environment before applying to production.