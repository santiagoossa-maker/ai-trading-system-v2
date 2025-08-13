"""
REST API for AI Trading System V2
Professional API for remote control and monitoring
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import uvicorn
import os
from datetime import datetime, timedelta
import json

# Optional imports for additional features
try:
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.openapi.utils import get_openapi
    DOCS_AVAILABLE = True
except ImportError:
    DOCS_AVAILABLE = False

logger = logging.getLogger(__name__)

# API Models
class SystemStatus(BaseModel):
    """System status response model"""
    status: str = Field(..., description="System status: RUNNING, STOPPED, ERROR")
    uptime: float = Field(..., description="System uptime in seconds")
    trading_active: bool = Field(..., description="Whether trading is active")
    last_signal: Optional[datetime] = Field(None, description="Last trading signal timestamp")
    positions_count: int = Field(..., description="Number of active positions")
    daily_pnl: float = Field(..., description="Daily P&L")
    message: str = Field("", description="Status message")

class TradingCommand(BaseModel):
    """Trading command model"""
    action: str = Field(..., description="Action: start, stop, pause, resume")
    symbol: Optional[str] = Field(None, description="Specific symbol (optional)")
    force: bool = Field(False, description="Force action even if unsafe")

class PositionInfo(BaseModel):
    """Position information model"""
    ticket: int = Field(..., description="Position ticket")
    symbol: str = Field(..., description="Trading symbol")
    type: str = Field(..., description="Position type: BUY or SELL")
    volume: float = Field(..., description="Position volume")
    price_open: float = Field(..., description="Opening price")
    price_current: float = Field(..., description="Current price")
    pnl: float = Field(..., description="Current P&L")
    profit: float = Field(..., description="Profit in account currency")
    time_open: datetime = Field(..., description="Opening time")

class OrderRequest(BaseModel):
    """Order request model"""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="BUY or SELL")
    volume: float = Field(..., description="Order volume")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    sl: Optional[float] = Field(None, description="Stop loss")
    tp: Optional[float] = Field(None, description="Take profit")
    comment: str = Field("API Order", description="Order comment")

class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    total_trades: int = Field(..., description="Total number of trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    total_pnl: float = Field(..., description="Total P&L")

class AlertRequest(BaseModel):
    """Alert request model"""
    level: str = Field(..., description="Alert level: INFO, WARNING, CRITICAL")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SignalWebhook(BaseModel):
    """External signal webhook model"""
    symbol: str = Field(..., description="Trading symbol")
    action: str = Field(..., description="BUY or SELL")
    strength: float = Field(..., description="Signal strength (0-1)")
    strategy: str = Field(..., description="Strategy name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Signal metadata")

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class TradingAPIServer:
    """
    Professional REST API server for the trading system
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.app = FastAPI(
            title="AI Trading System V2 API",
            description="Professional REST API for remote control and monitoring",
            version="2.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Get API configuration
        if self.config:
            api_config = self.config.get_api_config()
            self.host = api_config.host
            self.port = api_config.port
            self.debug = api_config.debug
            self.workers = api_config.workers
            self.api_key = api_config.api_key
            cors_origins = self.config.get('api.cors_origins', ["*"])
        else:
            self.host = "127.0.0.1"
            self.port = 8000
            self.debug = True
            self.workers = 1
            self.api_key = "dev_api_key"
            cors_origins = ["*"]
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Trading system reference (to be injected)
        self.trading_system = None
        self.health_monitor = None
        self.alert_manager = None
        
        logger.info(f"Trading API server initialized on {self.host}:{self.port}")
    
    def _verify_api_key(self, api_key: str = Depends(api_key_header)):
        """Verify API key"""
        if self.api_key and api_key != self.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return api_key
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", tags=["System"])
        async def root():
            """API root endpoint"""
            return {
                "message": "AI Trading System V2 API",
                "version": "2.0.0",
                "status": "online",
                "timestamp": datetime.utcnow().isoformat(),
                "docs": "/docs"
            }
        
        # System Control Endpoints
        @self.app.get("/api/v1/status", response_model=SystemStatus, tags=["System"])
        async def get_system_status(api_key: str = Depends(self._verify_api_key)):
            """Get current system status"""
            try:
                # Get status from trading system
                if self.trading_system:
                    status_data = self.trading_system.get_status()
                else:
                    status_data = {
                        'status': 'STOPPED',
                        'uptime': 0,
                        'trading_active': False,
                        'positions_count': 0,
                        'daily_pnl': 0.0,
                        'message': 'Trading system not connected'
                    }
                
                return SystemStatus(**status_data)
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/control", tags=["System"])
        async def control_system(
            command: TradingCommand,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Control trading system (start/stop/pause/resume)"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                action = command.action.lower()
                
                if action == "start":
                    background_tasks.add_task(self.trading_system.start_trading)
                    message = "Trading system start initiated"
                elif action == "stop":
                    background_tasks.add_task(self.trading_system.stop_trading, command.force)
                    message = "Trading system stop initiated"
                elif action == "pause":
                    background_tasks.add_task(self.trading_system.pause_trading)
                    message = "Trading system paused"
                elif action == "resume":
                    background_tasks.add_task(self.trading_system.resume_trading)
                    message = "Trading system resumed"
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {action}")
                
                return {"message": message, "action": action, "timestamp": datetime.utcnow()}
                
            except Exception as e:
                logger.error(f"Error controlling system: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/health", tags=["Monitoring"])
        async def get_health_status(api_key: str = Depends(self._verify_api_key)):
            """Get system health status"""
            try:
                if self.health_monitor:
                    health_data = self.health_monitor.get_health_summary()
                else:
                    health_data = {
                        'status': 'UNKNOWN',
                        'message': 'Health monitor not available'
                    }
                
                return health_data
                
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/health/detailed", tags=["Monitoring"])
        async def get_detailed_health(api_key: str = Depends(self._verify_api_key)):
            """Get detailed health metrics"""
            try:
                if self.health_monitor:
                    return self.health_monitor.get_detailed_metrics()
                else:
                    raise HTTPException(status_code=503, detail="Health monitor not available")
                
            except Exception as e:
                logger.error(f"Error getting detailed health: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Trading Endpoints
        @self.app.get("/api/v1/positions", response_model=List[PositionInfo], tags=["Trading"])
        async def get_positions(api_key: str = Depends(self._verify_api_key)):
            """Get all active positions"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                positions = self.trading_system.get_positions()
                return [PositionInfo(**pos) for pos in positions]
                
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/order", tags=["Trading"])
        async def place_order(
            order: OrderRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Place a trading order"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                result = self.trading_system.place_order(
                    symbol=order.symbol,
                    action=order.action,
                    volume=order.volume,
                    price=order.price,
                    sl=order.sl,
                    tp=order.tp,
                    comment=order.comment
                )
                
                return {"message": "Order placed successfully", "result": result}
                
            except Exception as e:
                logger.error(f"Error placing order: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/v1/position/{ticket}", tags=["Trading"])
        async def close_position(
            ticket: int,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Close a specific position"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                result = self.trading_system.close_position(ticket)
                return {"message": "Position closed successfully", "result": result}
                
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/v1/positions", tags=["Trading"])
        async def close_all_positions(api_key: str = Depends(self._verify_api_key)):
            """Close all positions (emergency stop)"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                result = self.trading_system.close_all_positions()
                return {"message": "All positions closed", "result": result}
                
            except Exception as e:
                logger.error(f"Error closing all positions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Performance Endpoints
        @self.app.get("/api/v1/performance", response_model=PerformanceMetrics, tags=["Analytics"])
        async def get_performance_metrics(api_key: str = Depends(self._verify_api_key)):
            """Get trading performance metrics"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                metrics = self.trading_system.get_performance_metrics()
                return PerformanceMetrics(**metrics)
                
            except Exception as e:
                logger.error(f"Error getting performance metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/performance/history", tags=["Analytics"])
        async def get_performance_history(
            days: int = 30,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Get performance history for specified days"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                history = self.trading_system.get_performance_history(days)
                return {"history": history, "days": days}
                
            except Exception as e:
                logger.error(f"Error getting performance history: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Alert Endpoints
        @self.app.post("/api/v1/alert", tags=["Monitoring"])
        async def send_alert(
            alert: AlertRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Send a custom alert"""
            try:
                if self.alert_manager:
                    self.alert_manager.send_alert(
                        level=alert.level,
                        title=alert.title,
                        message=alert.message,
                        source="api",
                        metadata=alert.metadata
                    )
                    return {"message": "Alert sent successfully"}
                else:
                    raise HTTPException(status_code=503, detail="Alert manager not available")
                
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/alerts", tags=["Monitoring"])
        async def get_alerts(
            level: Optional[str] = None,
            limit: int = 100,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Get alert history"""
            try:
                if self.alert_manager:
                    alerts = self.alert_manager.get_alert_history(level, limit)
                    return {
                        "alerts": [
                            {
                                "level": alert.level,
                                "title": alert.title,
                                "message": alert.message,
                                "timestamp": alert.timestamp.isoformat(),
                                "source": alert.source,
                                "metadata": alert.metadata
                            }
                            for alert in alerts
                        ],
                        "count": len(alerts)
                    }
                else:
                    raise HTTPException(status_code=503, detail="Alert manager not available")
                
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Webhook Endpoints
        @self.app.post("/api/v1/webhook/signal", tags=["Integration"])
        async def receive_signal_webhook(
            signal: SignalWebhook,
            background_tasks: BackgroundTasks,
            api_key: str = Depends(self._verify_api_key)
        ):
            """Receive external trading signal via webhook"""
            try:
                if not self.trading_system:
                    raise HTTPException(status_code=503, detail="Trading system not available")
                
                # Process signal in background
                background_tasks.add_task(
                    self.trading_system.process_external_signal,
                    signal.dict()
                )
                
                return {
                    "message": "Signal received and queued for processing",
                    "signal": signal.dict(),
                    "timestamp": datetime.utcnow()
                }
                
            except Exception as e:
                logger.error(f"Error processing webhook signal: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Configuration Endpoints
        @self.app.get("/api/v1/config", tags=["Configuration"])
        async def get_configuration(api_key: str = Depends(self._verify_api_key)):
            """Get current system configuration (safe values only)"""
            try:
                if self.config:
                    safe_config = {
                        'environment': self.config.get('environment.mode'),
                        'trading': {
                            'max_daily_loss_percent': self.config.get('trading.max_daily_loss_percent'),
                            'max_concurrent_positions': self.config.get('trading.max_concurrent_positions'),
                            'position_size_percent': self.config.get('trading.position_size_percent')
                        },
                        'monitoring': {
                            'health_check_interval': self.config.get('monitoring.health_check_interval'),
                            'email_alerts_enabled': self.config.get('monitoring.notifications.email.enabled'),
                            'telegram_alerts_enabled': self.config.get('monitoring.notifications.telegram.enabled')
                        }
                    }
                    return safe_config
                else:
                    raise HTTPException(status_code=503, detail="Configuration not available")
                
            except Exception as e:
                logger.error(f"Error getting configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Test Endpoints
        @self.app.post("/api/v1/test/notifications", tags=["Testing"])
        async def test_notifications(api_key: str = Depends(self._verify_api_key)):
            """Test all notification channels"""
            try:
                if self.alert_manager:
                    results = self.alert_manager.test_notifications()
                    return {"message": "Notification test completed", "results": results}
                else:
                    raise HTTPException(status_code=503, detail="Alert manager not available")
                
            except Exception as e:
                logger.error(f"Error testing notifications: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def set_trading_system(self, trading_system):
        """Set trading system reference"""
        self.trading_system = trading_system
        logger.info("Trading system reference set")
    
    def set_health_monitor(self, health_monitor):
        """Set health monitor reference"""
        self.health_monitor = health_monitor
        logger.info("Health monitor reference set")
    
    def set_alert_manager(self, alert_manager):
        """Set alert manager reference"""
        self.alert_manager = alert_manager
        logger.info("Alert manager reference set")
    
    def run(self):
        """Run the API server"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            workers=1 if self.debug else self.workers,
            log_level="debug" if self.debug else "info"
        )

# Global API server instance
_api_server = None

def get_api_server(config_manager=None) -> TradingAPIServer:
    """Get global API server instance"""
    global _api_server
    if _api_server is None:
        _api_server = TradingAPIServer(config_manager)
    return _api_server

def start_api_server(config_manager=None, trading_system=None, health_monitor=None, alert_manager=None):
    """Start the API server with system references"""
    server = get_api_server(config_manager)
    
    if trading_system:
        server.set_trading_system(trading_system)
    if health_monitor:
        server.set_health_monitor(health_monitor)
    if alert_manager:
        server.set_alert_manager(alert_manager)
    
    logger.info("Starting API server...")
    server.run()