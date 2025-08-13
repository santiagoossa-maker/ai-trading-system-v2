"""
Advanced Health Monitoring System
Real-time system health checks and performance metrics
"""

import time
import psutil
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import os

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    unit: str
    threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    status: str = "OK"  # OK, WARNING, CRITICAL
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""

@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str = "OK"  # OK, WARNING, CRITICAL, UNKNOWN
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    uptime: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    issues: List[str] = field(default_factory=list)

class HealthMonitor:
    """
    Comprehensive health monitoring system
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.start_time = time.time()
        self.monitoring = False
        self.health_data = SystemHealth()
        self._monitor_thread = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Health check intervals (seconds)
        self.check_interval = 60
        if self.config:
            self.check_interval = self.config.get('monitoring.health_check_interval', 60)
        
        # Initialize metric collectors
        self._metric_collectors = {
            'cpu': self._collect_cpu_metrics,
            'memory': self._collect_memory_metrics,
            'disk': self._collect_disk_metrics,
            'network': self._collect_network_metrics,
            'processes': self._collect_process_metrics,
        }
        
        logger.info("Health monitor initialized")
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring:
            logger.warning("Health monitoring already running")
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._perform_health_check()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait before retry
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        start_time = time.time()
        
        # Collect all metrics concurrently
        futures = {}
        for name, collector in self._metric_collectors.items():
            future = self._executor.submit(collector)
            futures[name] = future
        
        # Gather results
        new_metrics = {}
        for name, future in futures.items():
            try:
                metrics = future.result(timeout=30)
                new_metrics.update(metrics)
            except Exception as e:
                logger.error(f"Failed to collect {name} metrics: {e}")
                new_metrics[f"{name}_error"] = HealthMetric(
                    name=f"{name}_error",
                    value=1,
                    unit="boolean",
                    status="CRITICAL",
                    message=str(e)
                )
        
        # Update health data
        self.health_data.metrics = new_metrics
        self.health_data.uptime = time.time() - self.start_time
        self.health_data.last_check = datetime.now()
        
        # Determine overall status
        self._determine_overall_status()
        
        # Log health check duration
        duration = time.time() - start_time
        logger.debug(f"Health check completed in {duration:.2f}s")
    
    def _collect_cpu_metrics(self) -> Dict[str, HealthMetric]:
        """Collect CPU metrics"""
        metrics = {}
        
        try:
            # CPU usage percentage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_usage'] = HealthMetric(
                name='cpu_usage',
                value=cpu_percent,
                unit='percent',
                threshold=70.0,
                critical_threshold=90.0,
                status='OK' if cpu_percent < 70 else 'WARNING' if cpu_percent < 90 else 'CRITICAL'
            )
            
            # CPU frequency
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    metrics['cpu_frequency'] = HealthMetric(
                        name='cpu_frequency',
                        value=cpu_freq.current,
                        unit='MHz'
                    )
            except:
                pass
            
            # Load average (Unix only)
            try:
                load_avg = os.getloadavg()
                metrics['load_average_1m'] = HealthMetric(
                    name='load_average_1m',
                    value=load_avg[0],
                    unit='load',
                    threshold=psutil.cpu_count(),
                    critical_threshold=psutil.cpu_count() * 1.5
                )
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
        
        return metrics
    
    def _collect_memory_metrics(self) -> Dict[str, HealthMetric]:
        """Collect memory metrics"""
        metrics = {}
        
        try:
            # Virtual memory
            vmem = psutil.virtual_memory()
            memory_usage_percent = vmem.percent
            
            metrics['memory_usage'] = HealthMetric(
                name='memory_usage',
                value=memory_usage_percent,
                unit='percent',
                threshold=80.0,
                critical_threshold=95.0,
                status='OK' if memory_usage_percent < 80 else 'WARNING' if memory_usage_percent < 95 else 'CRITICAL'
            )
            
            metrics['memory_total'] = HealthMetric(
                name='memory_total',
                value=vmem.total / (1024**3),  # GB
                unit='GB'
            )
            
            metrics['memory_available'] = HealthMetric(
                name='memory_available',
                value=vmem.available / (1024**3),  # GB
                unit='GB'
            )
            
            # Swap memory
            swap = psutil.swap_memory()
            if swap.total > 0:
                metrics['swap_usage'] = HealthMetric(
                    name='swap_usage',
                    value=swap.percent,
                    unit='percent',
                    threshold=50.0,
                    critical_threshold=80.0
                )
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
        
        return metrics
    
    def _collect_disk_metrics(self) -> Dict[str, HealthMetric]:
        """Collect disk metrics"""
        metrics = {}
        
        try:
            # Disk usage for root partition
            disk_usage = psutil.disk_usage('/')
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            metrics['disk_usage'] = HealthMetric(
                name='disk_usage',
                value=usage_percent,
                unit='percent',
                threshold=85.0,
                critical_threshold=95.0,
                status='OK' if usage_percent < 85 else 'WARNING' if usage_percent < 95 else 'CRITICAL'
            )
            
            metrics['disk_total'] = HealthMetric(
                name='disk_total',
                value=disk_usage.total / (1024**3),  # GB
                unit='GB'
            )
            
            metrics['disk_free'] = HealthMetric(
                name='disk_free',
                value=disk_usage.free / (1024**3),  # GB
                unit='GB'
            )
            
            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics['disk_read_bytes'] = HealthMetric(
                        name='disk_read_bytes',
                        value=disk_io.read_bytes / (1024**2),  # MB
                        unit='MB'
                    )
                    
                    metrics['disk_write_bytes'] = HealthMetric(
                        name='disk_write_bytes',
                        value=disk_io.write_bytes / (1024**2),  # MB
                        unit='MB'
                    )
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
        
        return metrics
    
    def _collect_network_metrics(self) -> Dict[str, HealthMetric]:
        """Collect network metrics"""
        metrics = {}
        
        try:
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                metrics['network_bytes_sent'] = HealthMetric(
                    name='network_bytes_sent',
                    value=net_io.bytes_sent / (1024**2),  # MB
                    unit='MB'
                )
                
                metrics['network_bytes_recv'] = HealthMetric(
                    name='network_bytes_recv',
                    value=net_io.bytes_recv / (1024**2),  # MB
                    unit='MB'
                )
                
                metrics['network_packets_sent'] = HealthMetric(
                    name='network_packets_sent',
                    value=net_io.packets_sent,
                    unit='packets'
                )
                
                metrics['network_packets_recv'] = HealthMetric(
                    name='network_packets_recv',
                    value=net_io.packets_recv,
                    unit='packets'
                )
            
            # Network connections
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            metrics['active_connections'] = HealthMetric(
                name='active_connections',
                value=active_connections,
                unit='connections',
                threshold=100,
                critical_threshold=500
            )
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
        
        return metrics
    
    def _collect_process_metrics(self) -> Dict[str, HealthMetric]:
        """Collect process-related metrics"""
        metrics = {}
        
        try:
            # Current process info
            current_process = psutil.Process()
            
            # Process memory
            memory_info = current_process.memory_info()
            metrics['process_memory_rss'] = HealthMetric(
                name='process_memory_rss',
                value=memory_info.rss / (1024**2),  # MB
                unit='MB'
            )
            
            metrics['process_memory_vms'] = HealthMetric(
                name='process_memory_vms',
                value=memory_info.vms / (1024**2),  # MB
                unit='MB'
            )
            
            # Process CPU
            cpu_percent = current_process.cpu_percent()
            metrics['process_cpu_usage'] = HealthMetric(
                name='process_cpu_usage',
                value=cpu_percent,
                unit='percent'
            )
            
            # Process threads
            num_threads = current_process.num_threads()
            metrics['process_threads'] = HealthMetric(
                name='process_threads',
                value=num_threads,
                unit='threads',
                threshold=50,
                critical_threshold=100
            )
            
            # Open file descriptors
            try:
                num_fds = current_process.num_fds()
                metrics['process_open_fds'] = HealthMetric(
                    name='process_open_fds',
                    value=num_fds,
                    unit='descriptors',
                    threshold=100,
                    critical_threshold=500
                )
            except:
                pass
            
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
        
        return metrics
    
    def _determine_overall_status(self):
        """Determine overall system health status"""
        critical_count = 0
        warning_count = 0
        issues = []
        
        for metric in self.health_data.metrics.values():
            if metric.status == 'CRITICAL':
                critical_count += 1
                issues.append(f"CRITICAL: {metric.name} = {metric.value} {metric.unit}")
            elif metric.status == 'WARNING':
                warning_count += 1
                issues.append(f"WARNING: {metric.name} = {metric.value} {metric.unit}")
        
        # Determine overall status
        if critical_count > 0:
            self.health_data.status = 'CRITICAL'
        elif warning_count > 0:
            self.health_data.status = 'WARNING'
        else:
            self.health_data.status = 'OK'
        
        self.health_data.issues = issues
    
    def get_health_status(self) -> SystemHealth:
        """Get current health status"""
        return self.health_data
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary as dictionary"""
        return {
            'status': self.health_data.status,
            'uptime': self.health_data.uptime,
            'last_check': self.health_data.last_check.isoformat(),
            'issues_count': len(self.health_data.issues),
            'metrics_count': len(self.health_data.metrics),
            'critical_metrics': [
                m.name for m in self.health_data.metrics.values() 
                if m.status == 'CRITICAL'
            ],
            'warning_metrics': [
                m.name for m in self.health_data.metrics.values() 
                if m.status == 'WARNING'
            ]
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics as dictionary"""
        metrics_dict = {}
        for name, metric in self.health_data.metrics.items():
            metrics_dict[name] = {
                'value': metric.value,
                'unit': metric.unit,
                'status': metric.status,
                'timestamp': metric.timestamp.isoformat(),
                'threshold': metric.threshold,
                'critical_threshold': metric.critical_threshold,
                'message': metric.message
            }
        
        return {
            'system_status': self.health_data.status,
            'uptime_seconds': self.health_data.uptime,
            'last_check': self.health_data.last_check.isoformat(),
            'issues': self.health_data.issues,
            'metrics': metrics_dict
        }
    
    def export_metrics_json(self, filename: str = None) -> str:
        """Export metrics to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_metrics_{timestamp}.json"
        
        data = self.get_detailed_metrics()
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Health metrics exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            raise
    
    def check_mt5_connectivity(self) -> HealthMetric:
        """Check MT5 connection health"""
        try:
            # This would be implemented to check MT5 connection
            # For now, return a placeholder
            return HealthMetric(
                name='mt5_connectivity',
                value=1,
                unit='boolean',
                status='OK',
                message='MT5 connection check not implemented'
            )
        except Exception as e:
            return HealthMetric(
                name='mt5_connectivity',
                value=0,
                unit='boolean',
                status='CRITICAL',
                message=f"MT5 connection failed: {e}"
            )
    
    def check_database_connectivity(self) -> HealthMetric:
        """Check database connection health"""
        try:
            # This would be implemented to check database connections
            # For now, return a placeholder
            return HealthMetric(
                name='database_connectivity',
                value=1,
                unit='boolean',
                status='OK',
                message='Database connection check not implemented'
            )
        except Exception as e:
            return HealthMetric(
                name='database_connectivity',
                value=0,
                unit='boolean',
                status='CRITICAL',
                message=f"Database connection failed: {e}"
            )

# Global health monitor instance
_health_monitor = None

def get_health_monitor(config_manager=None) -> HealthMonitor:
    """Get global health monitor instance"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor(config_manager)
    return _health_monitor

def start_health_monitoring(config_manager=None):
    """Start global health monitoring"""
    monitor = get_health_monitor(config_manager)
    monitor.start_monitoring()

def stop_health_monitoring():
    """Stop global health monitoring"""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()