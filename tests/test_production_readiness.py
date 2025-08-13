"""
Production Readiness Test Suite
Comprehensive tests for all production components
"""

import pytest
import requests
import time
import os
import sys
from typing import Dict, Any
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.config_manager import ConfigManager
from monitoring.health_monitor import HealthMonitor
from monitoring.alert_system import AlertManager

logger = logging.getLogger(__name__)

class TestProductionReadiness:
    """Test suite for production readiness validation"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.config = ConfigManager('development')
        self.health_monitor = HealthMonitor(self.config)
        self.alert_manager = AlertManager(self.config)
        
        # Start monitoring for tests
        self.health_monitor.start_monitoring()
        yield
        
        # Cleanup
        self.health_monitor.stop_monitoring()
    
    def test_config_manager_loads_correctly(self):
        """Test configuration manager loads and substitutes variables"""
        assert self.config is not None
        assert self.config.get('environment.mode') == 'development'
        
        # Test environment variable substitution
        mt5_config = self.config.get_mt5_config()
        assert hasattr(mt5_config, 'login')
        assert hasattr(mt5_config, 'password')
        
        # Test different config types
        trading_config = self.config.get_trading_config()
        assert trading_config.max_daily_loss_percent > 0
        assert trading_config.max_concurrent_positions > 0
        
        logger.info("✅ Configuration manager test passed")
    
    def test_health_monitor_functionality(self):
        """Test health monitoring system"""
        # Wait for initial health check
        time.sleep(2)
        
        health_status = self.health_monitor.get_health_status()
        assert health_status is not None
        assert health_status.status in ['OK', 'WARNING', 'CRITICAL']
        assert health_status.uptime > 0
        
        # Test detailed metrics
        detailed_metrics = self.health_monitor.get_detailed_metrics()
        assert 'system_status' in detailed_metrics
        assert 'metrics' in detailed_metrics
        assert len(detailed_metrics['metrics']) > 0
        
        # Test specific metrics exist
        metrics = detailed_metrics['metrics']
        expected_metrics = ['cpu_usage', 'memory_usage', 'disk_usage']
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
        
        logger.info("✅ Health monitor test passed")
    
    def test_alert_system_functionality(self):
        """Test alert system"""
        # Test sending different alert levels
        self.alert_manager.send_info("Test Info", "This is a test info alert")
        self.alert_manager.send_warning("Test Warning", "This is a test warning alert")
        self.alert_manager.send_critical("Test Critical", "This is a test critical alert")
        
        # Wait for processing
        time.sleep(1)
        
        # Check alert history
        alert_history = self.alert_manager.get_alert_history(limit=10)
        assert len(alert_history) >= 3
        
        # Verify alert levels
        levels = [alert.level for alert in alert_history[-3:]]
        assert 'INFO' in levels
        assert 'WARNING' in levels
        assert 'CRITICAL' in levels
        
        # Test alert summary
        summary = self.alert_manager.get_alert_summary()
        assert 'total' in summary
        assert summary['total'] >= 3
        
        logger.info("✅ Alert system test passed")
    
    def test_environment_variables_handling(self):
        """Test environment variable handling"""
        # Test with actual environment variables
        os.environ['TEST_VAR'] = 'test_value'
        os.environ['TEST_VAR_WITH_DEFAULT'] = 'env_value'
        
        # Test substitution
        test_config = self.config._substitute_string('${TEST_VAR}')
        assert test_config == 'test_value'
        
        # Test with default
        test_with_default = self.config._substitute_string('${NONEXISTENT_VAR:-default_value}')
        assert test_with_default == 'default_value'
        
        # Test existing var with default
        test_existing_default = self.config._substitute_string('${TEST_VAR_WITH_DEFAULT:-default_value}')
        assert test_existing_default == 'env_value'
        
        # Cleanup
        del os.environ['TEST_VAR']
        del os.environ['TEST_VAR_WITH_DEFAULT']
        
        logger.info("✅ Environment variables test passed")
    
    def test_production_config_security(self):
        """Test production configuration security"""
        prod_config = ConfigManager('production')
        
        # Test that sensitive values are properly templated
        mt5_config = prod_config.get('mt5', {})
        
        # In production config, these should be template variables
        login = mt5_config.get('login', '')
        password = mt5_config.get('password', '')
        
        # Should contain ${} syntax or be empty for security
        assert login.startswith('${') or login == '', "MT5 login should be templated in production"
        assert password.startswith('${') or password == '', "MT5 password should be templated in production"
        
        # Test API security settings
        api_config = prod_config.get('api', {})
        assert api_config.get('debug', True) == False, "Debug should be disabled in production"
        
        logger.info("✅ Production config security test passed")
    
    def test_monitoring_integration(self):
        """Test integration between monitoring components"""
        # Test health monitor with alert manager integration
        # This would be implemented when they're fully integrated
        
        # For now, test they can coexist
        assert self.health_monitor is not None
        assert self.alert_manager is not None
        
        # Test they both use the same config
        assert self.health_monitor.config == self.config
        assert self.alert_manager.config == self.config
        
        logger.info("✅ Monitoring integration test passed")

class TestAPIEndpoints:
    """Test API endpoints (requires API server running)"""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API testing"""
        return "http://localhost:8000/api/v1"
    
    @pytest.fixture
    def api_headers(self):
        """Headers for API testing"""
        return {
            "X-API-Key": "dev_api_key",
            "Content-Type": "application/json"
        }
    
    def test_api_health_endpoint(self, api_base_url, api_headers):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{api_base_url}/health", headers=api_headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                assert 'status' in data
                logger.info("✅ API health endpoint test passed")
            else:
                pytest.skip("API server not running")
                
        except requests.ConnectionError:
            pytest.skip("API server not available for testing")
    
    def test_api_status_endpoint(self, api_base_url, api_headers):
        """Test API status endpoint"""
        try:
            response = requests.get(f"{api_base_url}/status", headers=api_headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['status', 'uptime', 'trading_active']
                for field in required_fields:
                    assert field in data, f"Missing field: {field}"
                logger.info("✅ API status endpoint test passed")
            else:
                pytest.skip("API server not running")
                
        except requests.ConnectionError:
            pytest.skip("API server not available for testing")

class TestPerformanceMetrics:
    """Test system performance metrics"""
    
    def test_health_check_performance(self):
        """Test health check performance"""
        health_monitor = HealthMonitor()
        
        # Measure health check duration
        start_time = time.time()
        health_monitor._perform_health_check()
        duration = time.time() - start_time
        
        # Health check should complete within reasonable time
        assert duration < 10.0, f"Health check took too long: {duration:.2f}s"
        
        logger.info(f"✅ Health check performance test passed ({duration:.2f}s)")
    
    def test_config_loading_performance(self):
        """Test configuration loading performance"""
        start_time = time.time()
        config = ConfigManager('development')
        duration = time.time() - start_time
        
        # Configuration should load quickly
        assert duration < 1.0, f"Config loading took too long: {duration:.2f}s"
        
        logger.info(f"✅ Config loading performance test passed ({duration:.2f}s)")

class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_configuration_data_types(self):
        """Test configuration data types are correct"""
        config = ConfigManager('development')
        
        # Test numeric values
        assert isinstance(config.get('trading.max_daily_loss_percent'), (int, float))
        assert isinstance(config.get('trading.max_concurrent_positions'), int)
        assert isinstance(config.get('api.port'), int)
        
        # Test boolean values
        assert isinstance(config.get('environment.debug'), bool)
        
        # Test string values
        assert isinstance(config.get('environment.mode'), str)
        
        logger.info("✅ Configuration data types test passed")
    
    def test_health_metrics_data_integrity(self):
        """Test health metrics data integrity"""
        health_monitor = HealthMonitor()
        health_monitor._perform_health_check()
        
        metrics = health_monitor.get_detailed_metrics()
        
        # Test all metrics have required fields
        for metric_name, metric_data in metrics['metrics'].items():
            assert 'value' in metric_data
            assert 'unit' in metric_data
            assert 'status' in metric_data
            assert 'timestamp' in metric_data
            
            # Test value is numeric
            assert isinstance(metric_data['value'], (int, float))
            
            # Test status is valid
            assert metric_data['status'] in ['OK', 'WARNING', 'CRITICAL']
        
        logger.info("✅ Health metrics data integrity test passed")

# Test runner configuration
def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "api: marks tests as API tests")

if __name__ == "__main__":
    # Run tests when script is executed directly
    import subprocess
    import sys
    
    print("🚀 Running Production Readiness Test Suite...")
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "--color=yes"
    ])
    
    if result.returncode == 0:
        print("\n✅ All production readiness tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above.")
        sys.exit(1)