"""
Production Configuration Manager
Handles environment-based configuration with environment variable substitution
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_system"
    postgres_user: str = ""
    postgres_password: str = ""

@dataclass
class MT5Config:
    """MT5 connection configuration"""
    login: str = ""
    password: str = ""
    server: str = ""
    path: str = ""
    timeout: int = 60000
    max_retries: int = 3
    retry_delay: int = 5

@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    max_daily_loss_percent: float = 5.0
    max_concurrent_positions: int = 10
    position_size_percent: float = 2.0
    max_spread_points: float = 3.0
    emergency_stop_loss_percent: float = 10.0
    max_drawdown_percent: float = 15.0
    daily_profit_target_percent: float = 8.0

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    jwt_secret: str = ""
    api_key: str = ""

@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    health_check_interval: int = 60
    performance_metrics_interval: int = 300
    email_alerts_enabled: bool = False
    telegram_alerts_enabled: bool = False
    webhook_alerts_enabled: bool = False

class ConfigManager:
    """
    Manages application configuration with environment variable substitution
    """
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('TRADING_ENV', 'development')
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration based on environment"""
        config_file = self.config_dir / f"{self.environment}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Config file {config_file} not found, falling back to development")
            config_file = self.config_dir / "development.yaml"
        
        try:
            with open(config_file, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Substitute environment variables
            self._config = self._substitute_env_vars(raw_config)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = self._get_default_config()
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string(config)
        else:
            return config
    
    def _substitute_string(self, text: str) -> str:
        """Substitute environment variables in string with ${VAR} or ${VAR:-default} syntax"""
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_expr = match.group(1)
            
            # Handle default values: VAR:-default
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name.strip(), default_value)
            else:
                return os.getenv(var_expr.strip(), match.group(0))
        
        return re.sub(pattern, replace_var, text)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get minimal default configuration"""
        return {
            'environment': {'mode': 'development', 'debug': True, 'log_level': 'INFO'},
            'mt5': {'login': '', 'password': '', 'server': '', 'path': ''},
            'trading': {
                'max_daily_loss_percent': 2.0,
                'max_concurrent_positions': 3,
                'position_size_percent': 0.5,
                'max_spread_points': 5.0
            },
            'api': {'host': '127.0.0.1', 'port': 8000, 'debug': True, 'workers': 1},
            'database': {
                'redis': {'host': 'localhost', 'port': 6379, 'db': 0},
                'postgresql': {'host': 'localhost', 'port': 5432, 'database': 'trading_system'}
            },
            'monitoring': {
                'health_check_interval': 300,
                'notifications': {'email': {'enabled': False}, 'telegram': {'enabled': False}}
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'mt5.login')"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_mt5_config(self) -> MT5Config:
        """Get MT5 configuration"""
        mt5_config = self.get('mt5', {})
        return MT5Config(
            login=mt5_config.get('login', ''),
            password=mt5_config.get('password', ''),
            server=mt5_config.get('server', ''),
            path=mt5_config.get('path', ''),
            timeout=mt5_config.get('timeout', 60000),
            max_retries=mt5_config.get('max_retries', 3),
            retry_delay=mt5_config.get('retry_delay', 5)
        )
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration"""
        trading_config = self.get('trading', {})
        return TradingConfig(
            max_daily_loss_percent=trading_config.get('max_daily_loss_percent', 5.0),
            max_concurrent_positions=trading_config.get('max_concurrent_positions', 10),
            position_size_percent=trading_config.get('position_size_percent', 2.0),
            max_spread_points=trading_config.get('max_spread_points', 3.0),
            emergency_stop_loss_percent=trading_config.get('emergency_stop_loss_percent', 10.0),
            max_drawdown_percent=trading_config.get('max_drawdown_percent', 15.0),
            daily_profit_target_percent=trading_config.get('daily_profit_target_percent', 8.0)
        )
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        db_config = self.get('database', {})
        redis_config = db_config.get('redis', {})
        postgres_config = db_config.get('postgresql', {})
        
        return DatabaseConfig(
            redis_host=redis_config.get('host', 'localhost'),
            redis_port=redis_config.get('port', 6379),
            redis_password=redis_config.get('password'),
            redis_db=redis_config.get('db', 0),
            postgres_host=postgres_config.get('host', 'localhost'),
            postgres_port=postgres_config.get('port', 5432),
            postgres_db=postgres_config.get('database', 'trading_system'),
            postgres_user=postgres_config.get('username', ''),
            postgres_password=postgres_config.get('password', '')
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration"""
        api_config = self.get('api', {})
        return APIConfig(
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 8000),
            debug=api_config.get('debug', False),
            workers=api_config.get('workers', 4),
            jwt_secret=api_config.get('jwt_secret', ''),
            api_key=api_config.get('api_key', '')
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        monitoring_config = self.get('monitoring', {})
        notifications = monitoring_config.get('notifications', {})
        
        return MonitoringConfig(
            health_check_interval=monitoring_config.get('health_check_interval', 60),
            performance_metrics_interval=monitoring_config.get('performance_metrics_interval', 300),
            email_alerts_enabled=notifications.get('email', {}).get('enabled', False),
            telegram_alerts_enabled=notifications.get('telegram', {}).get('enabled', False),
            webhook_alerts_enabled=notifications.get('webhook', {}).get('enabled', False)
        )
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.get('environment.mode') == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.get('environment.mode') == 'development'
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return self.get('logging.level', 'INFO')
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
        logger.info("Configuration reloaded")

# Global configuration instance
config = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload global configuration"""
    global config
    config.reload_config()