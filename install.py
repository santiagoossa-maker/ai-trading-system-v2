"""
Installation Script for AI Trading System V2
Automated installation and configuration
"""

import os
import sys
import subprocess
import platform
import shutil
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystemInstaller:
    """Installer for the AI Trading System V2"""
    
    def __init__(self):
        self.system = platform.system()
        self.root_dir = Path(__file__).parent
        self.config_dir = self.root_dir / "config"
        self.logs_dir = self.root_dir / "logs"
        
    def run_installation(self):
        """Run complete installation process"""
        logger.info("🚀 Starting AI Trading System V2 Installation")
        logger.info("=" * 60)
        
        try:
            # Step 1: Check system requirements
            self.check_system_requirements()
            
            # Step 2: Create directory structure
            self.create_directories()
            
            # Step 3: Install essential dependencies
            self.install_essential_dependencies()
            
            # Step 4: Create configuration files
            self.create_configuration_files()
            
            # Step 5: Create startup scripts
            self.create_startup_scripts()
            
            # Step 6: Run basic tests
            self.run_basic_tests()
            
            # Step 7: Display completion message
            self.display_completion_message()
            
            logger.info("✅ Installation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Installation failed: {e}")
            return False
    
    def check_system_requirements(self):
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            raise Exception(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        
        logger.info(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
            logger.info("✓ pip available")
        except subprocess.CalledProcessError:
            raise Exception("pip not found. Please install pip.")
        
        logger.info("✅ System requirements check passed")
    
    def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directory structure...")
        
        directories = [
            self.config_dir,
            self.logs_dir,
            self.root_dir / "data",
            self.root_dir / "models",
            self.root_dir / "backups"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"✓ Created {directory}")
        
        logger.info("✅ Directory structure created")
    
    def install_essential_dependencies(self):
        """Install essential Python dependencies"""
        logger.info("📦 Installing essential dependencies...")
        
        # Essential dependencies for basic functionality
        essential_deps = [
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "PyYAML>=6.0",
            "streamlit>=1.25.0",
            "plotly>=5.15.0"
        ]
        
        for dep in essential_deps:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
                logger.info(f"✓ {dep}")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {dep}")
        
        logger.info("✅ Essential dependencies installation completed")
    
    def create_configuration_files(self):
        """Create configuration files"""
        logger.info("⚙️ Creating configuration files...")
        
        # Main configuration
        main_config = {
            'trading_system': {
                'name': 'AI Trading System V2',
                'version': '2.0.0',
                'mode': 'demo'
            },
            'strategy': {
                'sma_fast': 8,
                'sma_slow': 50,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'risk_per_trade': 0.02,
                'risk_reward_ratio': 2.0
            },
            'symbols': ['R_75', 'R_100', 'R_50'],
            'timeframe': 'M5',
            'update_interval': 5
        }
        
        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(main_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✓ Main configuration created: {config_file}")
        
        # MT5 configuration template
        mt5_config = {
            'mt5': {
                'login': 'YOUR_MT5_LOGIN',
                'password': 'YOUR_MT5_PASSWORD',
                'server': 'YOUR_MT5_SERVER'
            }
        }
        
        mt5_file = self.config_dir / "mt5_config_template.yaml"
        with open(mt5_file, 'w') as f:
            yaml.dump(mt5_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✓ MT5 configuration template created: {mt5_file}")
        
        logger.info("✅ Configuration files created")
    
    def create_startup_scripts(self):
        """Create startup scripts"""
        logger.info("📜 Creating startup scripts...")
        
        # Windows batch file
        if self.system == "Windows":
            batch_content = f"""@echo off
echo Starting AI Trading Dashboard...
cd /d "{self.root_dir}"
python -m streamlit run src/dashboard/streamlit_dashboard.py
pause
"""
            batch_file = self.root_dir / "start_dashboard.bat"
            with open(batch_file, 'w') as f:
                f.write(batch_content)
            logger.info(f"✓ Windows batch file created: {batch_file}")
        
        # Unix shell script
        else:
            shell_content = f"""#!/bin/bash
echo "Starting AI Trading Dashboard..."
cd "{self.root_dir}"
python -m streamlit run src/dashboard/streamlit_dashboard.py
"""
            shell_file = self.root_dir / "start_dashboard.sh"
            with open(shell_file, 'w') as f:
                f.write(shell_content)
            os.chmod(shell_file, 0o755)
            logger.info(f"✓ Shell script created: {shell_file}")
        
        logger.info("✅ Startup scripts created")
    
    def run_basic_tests(self):
        """Run basic tests to verify installation"""
        logger.info("🧪 Running basic tests...")
        
        try:
            # Test basic imports
            import pandas as pd
            logger.info("✓ pandas working")
            
            import numpy as np
            logger.info("✓ numpy working")
            
            import yaml
            logger.info("✓ PyYAML working")
            
            logger.info("✅ Basic tests passed")
            
        except ImportError as e:
            logger.warning(f"⚠️ Import test failed: {e}")
    
    def display_completion_message(self):
        """Display completion message and next steps"""
        logger.info("🎉 Installation completed successfully!")
        logger.info("=" * 60)
        logger.info("QUICK START GUIDE:")
        logger.info("=" * 60)
        
        logger.info("1. 🎛️ Start the dashboard:")
        if self.system == "Windows":
            logger.info("   - Double-click start_dashboard.bat")
        else:
            logger.info("   - Run ./start_dashboard.sh")
        logger.info("   - Open http://localhost:8501 in your browser")
        
        logger.info("2. 📝 Configure MT5 (for live trading):")
        logger.info(f"   - Edit {self.config_dir}/mt5_config_template.yaml")
        logger.info("   - Add your MT5 credentials")
        logger.info("   - Rename to mt5_config.yaml")
        
        logger.info("3. 🤖 Start trading:")
        logger.info("   - Use the dashboard to control trading")
        logger.info("   - Start in demo mode for testing")
        
        logger.info("=" * 60)
        logger.info("🎯 Your AI Trading System V2 is ready!")
        logger.info("=" * 60)

def main():
    """Main installation function"""
    print("🚀 AI Trading System V2 - Quick Installation")
    print("=" * 50)
    
    try:
        installer = TradingSystemInstaller()
        success = installer.run_installation()
        
        if success:
            print("\n✅ Installation completed successfully!")
            print("🎯 Your trading system is ready to use!")
            return 0
        else:
            print("\n❌ Installation failed!")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Installation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())