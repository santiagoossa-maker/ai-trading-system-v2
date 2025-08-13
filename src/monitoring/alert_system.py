"""
Advanced Alert System
Email, Telegram, and webhook notifications for critical events
"""

import logging
import smtplib
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Alert message structure"""
    level: str  # INFO, WARNING, CRITICAL
    title: str
    message: str
    timestamp: datetime
    source: str = "trading_system"
    metadata: Dict[str, Any] = None

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.smtp_host = config.get('smtp_host', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_address = config.get('from_address', '')
        self.to_addresses = config.get('to_addresses', [])
        
        if isinstance(self.to_addresses, str):
            self.to_addresses = [addr.strip() for addr in self.to_addresses.split(',')]
    
    def send_alert(self, alert: Alert) -> bool:
        """Send email alert"""
        if not self.enabled or not self.to_addresses:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = f"[{alert.level}] {alert.title}"
            
            # Create HTML body
            html_body = self._create_html_body(alert)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _create_html_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        color_map = {
            'INFO': '#17a2b8',
            'WARNING': '#ffc107',
            'CRITICAL': '#dc3545'
        }
        
        color = color_map.get(alert.level, '#6c757d')
        
        html = f"""
        <html>
        <body>
            <div style="max-width: 600px; margin: 0 auto; font-family: Arial, sans-serif;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h2 style="margin: 0;">{alert.level} Alert</h2>
                    <h3 style="margin: 10px 0 0 0;">{alert.title}</h3>
                </div>
                
                <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                    <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    <p><strong>Source:</strong> {alert.source}</p>
                    <p><strong>Message:</strong></p>
                    <div style="background-color: white; padding: 15px; border-left: 4px solid {color}; margin: 10px 0;">
                        {alert.message.replace('\n', '<br>')}
                    </div>
                </div>
                
                {self._format_metadata(alert.metadata) if alert.metadata else ''}
                
                <div style="background-color: #e9ecef; padding: 10px; text-align: center; border-radius: 0 0 8px 8px; font-size: 12px;">
                    AI Trading System V2 - Automated Alert
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as HTML table"""
        if not metadata:
            return ""
        
        rows = ""
        for key, value in metadata.items():
            rows += f"<tr><td><strong>{key}:</strong></td><td>{value}</td></tr>"
        
        return f"""
        <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; border-top: none;">
            <h4>Additional Details:</h4>
            <table style="width: 100%; border-collapse: collapse;">
                {rows}
            </table>
        </div>
        """

class TelegramNotifier:
    """Telegram notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.bot_token = config.get('bot_token', '')
        self.chat_ids = config.get('chat_ids', [])
        
        if isinstance(self.chat_ids, str):
            self.chat_ids = [chat_id.strip() for chat_id in self.chat_ids.split(',')]
    
    def send_alert(self, alert: Alert) -> bool:
        """Send Telegram alert"""
        if not self.enabled or not self.bot_token or not self.chat_ids or not REQUESTS_AVAILABLE:
            return False
        
        try:
            message = self._format_telegram_message(alert)
            
            success_count = 0
            for chat_id in self.chat_ids:
                if self._send_to_chat(chat_id, message):
                    success_count += 1
            
            if success_count > 0:
                logger.info(f"Telegram alert sent to {success_count}/{len(self.chat_ids)} chats")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False
    
    def _send_to_chat(self, chat_id: str, message: str) -> bool:
        """Send message to specific Telegram chat"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message to {chat_id}: {e}")
            return False
    
    def _format_telegram_message(self, alert: Alert) -> str:
        """Format alert as Telegram message"""
        emoji_map = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🚨'
        }
        
        emoji = emoji_map.get(alert.level, '📢')
        
        message = f"{emoji} *{alert.level} Alert*\n\n"
        message += f"*{alert.title}*\n\n"
        message += f"📅 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        message += f"🏷️ Source: {alert.source}\n\n"
        message += f"📝 {alert.message}\n"
        
        if alert.metadata:
            message += "\n*Additional Details:*\n"
            for key, value in alert.metadata.items():
                message += f"• {key}: {value}\n"
        
        return message

class WebhookNotifier:
    """Webhook notification handler"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get('enabled', False)
        self.url = config.get('url', '')
        self.headers = config.get('headers', {})
        self.timeout = config.get('timeout', 10)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send webhook alert"""
        if not self.enabled or not self.url or not REQUESTS_AVAILABLE:
            return False
        
        try:
            payload = {
                'level': alert.level,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'metadata': alert.metadata or {}
            }
            
            headers = {'Content-Type': 'application/json'}
            headers.update(self.headers)
            
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

class AlertManager:
    """
    Centralized alert management system
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        
        # Initialize notifiers
        self._setup_notifiers()
        
        # Rate limiting
        self._alert_counts = {}
        self._reset_time = time.time()
        self.rate_limit_window = 300  # 5 minutes
        self.max_alerts_per_window = 10
        
        # Background processing
        self._executor = ThreadPoolExecutor(max_workers=3)
        
        logger.info("Alert manager initialized")
    
    def _setup_notifiers(self):
        """Setup notification handlers"""
        self.notifiers = {}
        
        if self.config:
            notifications_config = self.config.get('monitoring.notifications', {})
            
            # Email notifier
            email_config = notifications_config.get('email', {})
            self.notifiers['email'] = EmailNotifier(email_config)
            
            # Telegram notifier
            telegram_config = notifications_config.get('telegram', {})
            self.notifiers['telegram'] = TelegramNotifier(telegram_config)
            
            # Webhook notifier
            webhook_config = notifications_config.get('webhook', {})
            self.notifiers['webhook'] = WebhookNotifier(webhook_config)
        else:
            # Default disabled notifiers
            self.notifiers = {
                'email': EmailNotifier({'enabled': False}),
                'telegram': TelegramNotifier({'enabled': False}),
                'webhook': WebhookNotifier({'enabled': False})
            }
    
    def send_alert(self, level: str, title: str, message: str, 
                   source: str = "trading_system", metadata: Dict[str, Any] = None):
        """Send alert through all configured channels"""
        alert = Alert(
            level=level.upper(),
            title=title,
            message=message,
            timestamp=datetime.utcnow(),
            source=source,
            metadata=metadata or {}
        )
        
        # Check rate limiting
        if not self._check_rate_limit(alert):
            logger.warning(f"Alert rate limited: {title}")
            return
        
        # Add to history
        self._add_to_history(alert)
        
        # Send notifications asynchronously
        self._executor.submit(self._process_alert, alert)
        
        logger.info(f"Alert queued: [{level}] {title}")
    
    def _check_rate_limit(self, alert: Alert) -> bool:
        """Check if alert should be rate limited"""
        current_time = time.time()
        
        # Reset counters if window expired
        if current_time - self._reset_time > self.rate_limit_window:
            self._alert_counts.clear()
            self._reset_time = current_time
        
        # Count alerts by level
        count_key = f"{alert.level}:{alert.title}"
        current_count = self._alert_counts.get(count_key, 0)
        
        if current_count >= self.max_alerts_per_window:
            return False
        
        self._alert_counts[count_key] = current_count + 1
        return True
    
    def _add_to_history(self, alert: Alert):
        """Add alert to history with rotation"""
        self.alert_history.append(alert)
        
        # Rotate history if needed
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
    
    def _process_alert(self, alert: Alert):
        """Process alert through all notifiers"""
        results = {}
        
        for name, notifier in self.notifiers.items():
            try:
                success = notifier.send_alert(alert)
                results[name] = success
            except Exception as e:
                logger.error(f"Error in {name} notifier: {e}")
                results[name] = False
        
        # Log results
        successful = [name for name, success in results.items() if success]
        if successful:
            logger.info(f"Alert sent via: {', '.join(successful)}")
        else:
            logger.warning(f"Alert failed to send via all channels: {alert.title}")
    
    def send_info(self, title: str, message: str, **kwargs):
        """Send info level alert"""
        self.send_alert('INFO', title, message, **kwargs)
    
    def send_warning(self, title: str, message: str, **kwargs):
        """Send warning level alert"""
        self.send_alert('WARNING', title, message, **kwargs)
    
    def send_critical(self, title: str, message: str, **kwargs):
        """Send critical level alert"""
        self.send_alert('CRITICAL', title, message, **kwargs)
    
    def get_alert_history(self, level: str = None, limit: int = None) -> List[Alert]:
        """Get alert history with optional filtering"""
        alerts = self.alert_history
        
        if level:
            alerts = [a for a in alerts if a.level == level.upper()]
        
        if limit:
            alerts = alerts[-limit:]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        if not self.alert_history:
            return {'total': 0, 'by_level': {}, 'last_24h': 0}
        
        # Count by level
        by_level = {}
        last_24h = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for alert in self.alert_history:
            # Count by level
            by_level[alert.level] = by_level.get(alert.level, 0) + 1
            
            # Count last 24h
            if alert.timestamp > cutoff_time:
                last_24h += 1
        
        return {
            'total': len(self.alert_history),
            'by_level': by_level,
            'last_24h': last_24h,
            'rate_limited': len(self._alert_counts)
        }
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification channels"""
        test_alert = Alert(
            level='INFO',
            title='Test Alert',
            message='This is a test alert from the AI Trading System V2.',
            timestamp=datetime.utcnow(),
            source='alert_manager_test',
            metadata={'test': True}
        )
        
        results = {}
        for name, notifier in self.notifiers.items():
            try:
                success = notifier.send_alert(test_alert)
                results[name] = success
            except Exception as e:
                logger.error(f"Test failed for {name}: {e}")
                results[name] = False
        
        return results

# Global alert manager instance
_alert_manager = None

def get_alert_manager(config_manager=None) -> AlertManager:
    """Get global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager(config_manager)
    return _alert_manager

def send_alert(level: str, title: str, message: str, **kwargs):
    """Global function to send alerts"""
    manager = get_alert_manager()
    manager.send_alert(level, title, message, **kwargs)

def send_info(title: str, message: str, **kwargs):
    """Send info alert"""
    send_alert('INFO', title, message, **kwargs)

def send_warning(title: str, message: str, **kwargs):
    """Send warning alert"""
    send_alert('WARNING', title, message, **kwargs)

def send_critical(title: str, message: str, **kwargs):
    """Send critical alert"""
    send_alert('CRITICAL', title, message, **kwargs)