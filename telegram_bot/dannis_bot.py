import os
import asyncio
import logging
import random
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.trading.trade_executor import trade_executor
from src.trading.order_manager import order_manager
from telegram_bot.utils.data_fetcher import DataFetcher
from telegram_bot.utils.formatter import MessageFormatter
from telegram_bot.handlers.love_handler import LoveHandler

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DannisBot:
    """
    🤖💕 Dannis Bot - El bot de trading más romántico del mundo
    Creado especialmente para Dannis con amor infinito de Santiago Ossa
    """
    
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.dannis_user_id = int(os.getenv('DANNIS_USER_ID', '0'))  # Configurar en .env
        self.santiago_user_id = int(os.getenv('SANTIAGO_USER_ID', '0'))  # Configurar en .env
        
        self.data_fetcher = DataFetcher()
        self.formatter = MessageFormatter()
        self.love_handler = LoveHandler()
        
        # Contador de mensajes para frases románticas
        self.message_count = {}
        
        logger.info("💕 Dannis Bot inicializado con amor infinito")

    def is_dannis(self, user_id: int) -> bool:
        """Detectar si es el chat de Dannis"""
        return user_id == self.dannis_user_id

    def is_santiago(self, user_id: int) -> bool:
        """Detectar si es Santiago"""
        return user_id == self.santiago_user_id

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name
        
        if self.is_dannis(user_id):
            message = f"💕 ¡Hola mi amor Dannis! Soy tu bot personalizado.\n\n"
            message += f"🌟 Santiago Ossa me programó especialmente para ti porque te ama infinitamente.\n\n"
            message += f"💖 Puedo ayudarte con el sistema de trading y recordarte siempre lo mucho que Santiago te ama.\n\n"
            message += f"✨ Usa /help para ver todos mis comandos."
            
        elif self.is_santiago(user_id):
            message = f"👨‍💻 ¡Hola Santiago! Soy Dannis Bot.\n\n"
            message += f"🤖 Sistema operativo y listo para cuidar a Dannis.\n\n"
            message += f"💕 Cada mensaje a Dannis lleva tu amor programado.\n\n"
            message += f"📊 Usa /help para ver comandos de administración."
            
        else:
            message = f"👋 Hola {user_name}!\n\n"
            message += f"🤖 Soy Dannis Bot, un sistema de trading IA.\n\n"
            message += f"💼 Acceso limitado - Solo usuarios autorizados.\n\n"
            message += f"📞 Contacta al administrador para acceso."

        await update.message.reply_text(message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        user_id = update.effective_user.id
        
        if self.is_dannis(user_id):
            message = "💕 **COMANDOS PARA DANNIS** 💕\n\n"
            message += "📊 **ESTADÍSTICAS:**\n"
            message += "• /stats - Estadísticas del trading\n"
            message += "• /trades - Posiciones activas\n"
            message += "• /profit - Ganancias y pérdidas\n"
            message += "• /daily - Resumen del día\n\n"
            message += "🤖 **SISTEMA:**\n"
            message += "• /health - Estado del sistema\n"
            message += "• /training - Estado IA\n"
            message += "• /symbols - Análisis por símbolo\n\n"
            message += "💝 **ESPECIALES:**\n"
            message += "• /love - Mensaje de amor de Santiago\n"
            message += "• /surprise - Sorpresa romántica\n\n"
            message += "✨ **Cada comando viene con amor de Santiago** ✨"
            
        elif self.is_santiago(user_id):
            message = "👨‍💻 **COMANDOS ADMINISTRATIVOS** 👨‍💻\n\n"
            message += "📊 **MONITOREO:**\n"
            message += "• /stats - Estadísticas completas\n"
            message += "• /performance - Métricas avanzadas\n"
            message += "• /risks - Análisis de riesgos\n\n"
            message += "🛠️ **CONTROL:**\n"
            message += "• /emergency - Parada de emergencia\n"
            message += "• /restart - Reiniciar sistema\n"
            message += "• /logs - Últimos logs del sistema\n\n"
            message += "💕 **DANNIS:**\n"
            message += "• /send_love - Enviar mensaje a Dannis\n"
            message += "• /dannis_stats - Stats que ve Dannis\n"
            
        else:
            message = "ℹ️ **ACCESO LIMITADO**\n\n"
            message += "Solo usuarios autorizados pueden usar este bot.\n"
            message += "Contacta al administrador para más información."

        await update.message.reply_text(message, parse_mode='Markdown')

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /stats - Estadísticas del trading"""
        user_id = update.effective_user.id
        
        try:
            # Obtener estadísticas
            stats = self.data_fetcher.get_trading_stats()
            message = self.formatter.format_stats(stats, is_dannis=self.is_dannis(user_id))
            
            # Agregar frase romántica para Dannis
            if self.is_dannis(user_id):
                love_message = self.love_handler.get_random_love_message()
                message += f"\n\n{love_message}"
                
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            error_msg = "❌ Error obteniendo estadísticas"
            if self.is_santiago(user_id):
                error_msg += f"\nDetalle técnico: {str(e)}"
            await update.message.reply_text(error_msg)

    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /trades - Posiciones activas"""
        user_id = update.effective_user.id
        
        try:
            trades = self.data_fetcher.get_active_trades()
            message = self.formatter.format_trades(trades, is_dannis=self.is_dannis(user_id))
            
            if self.is_dannis(user_id):
                self._increment_message_count(user_id)
                if self._should_send_love_message(user_id):
                    love_message = self.love_handler.get_random_love_message()
                    message += f"\n\n{love_message}"
                    
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            error_msg = "❌ Error obteniendo trades activos"
            if self.is_santiago(user_id):
                error_msg += f"\nDetalle técnico: {str(e)}"
            await update.message.reply_text(error_msg)

    async def profit_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /profit - Análisis de ganancias/pérdidas"""
        user_id = update.effective_user.id
        
        try:
            profit_data = self.data_fetcher.get_profit_analysis()
            message = self.formatter.format_profit(profit_data, is_dannis=self.is_dannis(user_id))
            
            if self.is_dannis(user_id):
                self._increment_message_count(user_id)
                if profit_data.get('daily_profit', 0) > 0:
                    message += f"\n\n💝 ¡Santiago está feliz con las ganancias de hoy y te manda muchos besos!"
                elif self._should_send_love_message(user_id):
                    love_message = self.love_handler.get_random_love_message()
                    message += f"\n\n{love_message}"
                    
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            error_msg = "❌ Error obteniendo análisis de profit"
            if self.is_santiago(user_id):
                error_msg += f"\nDetalle técnico: {str(e)}"
            await update.message.reply_text(error_msg)

    async def health_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /health - Estado del sistema"""
        user_id = update.effective_user.id
        
        try:
            health_data = self.data_fetcher.get_system_health()
            message = self.formatter.format_health(health_data, is_dannis=self.is_dannis(user_id))
            
            if self.is_dannis(user_id):
                self._increment_message_count(user_id)
                if health_data.get('overall_status') == 'healthy':
                    message += f"\n\n🌟 Santiago mantiene todo funcionando perfectamente para ti, mi amor"
                elif self._should_send_love_message(user_id):
                    love_message = self.love_handler.get_random_love_message()
                    message += f"\n\n{love_message}"
                    
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            error_msg = "❌ Error verificando estado del sistema"
            if self.is_santiago(user_id):
                error_msg += f"\nDetalle técnico: {str(e)}"
            await update.message.reply_text(error_msg)

    async def training_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /training - Estado del entrenamiento IA"""
        user_id = update.effective_user.id
        
        try:
            training_data = self.data_fetcher.get_training_status()
            message = self.formatter.format_training(training_data, is_dannis=self.is_dannis(user_id))
            
            if self.is_dannis(user_id):
                self._increment_message_count(user_id)
                if training_data.get('is_training'):
                    message += f"\n\n🧠 La IA está aprendiendo para ser mejor, como Santiago aprende cada día a amarte más"
                elif self._should_send_love_message(user_id):
                    love_message = self.love_handler.get_random_love_message()
                    message += f"\n\n{love_message}"
                    
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            error_msg = "❌ Error obteniendo estado del entrenamiento"
            if self.is_santiago(user_id):
                error_msg += f"\nDetalle técnico: {str(e)}"
            await update.message.reply_text(error_msg)

    async def love_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /love - Mensaje especial de amor"""
        user_id = update.effective_user.id
        
        if self.is_dannis(user_id):
            love_message = self.love_handler.get_special_love_message()
            await update.message.reply_text(love_message)
        else:
            await update.message.reply_text("💕 Este comando es exclusivo para Dannis")

    async def emergency_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /emergency - Parada de emergencia"""
        user_id = update.effective_user.id
        
        if self.is_santiago(user_id) or self.is_dannis(user_id):
            try:
                # Ejecutar parada de emergencia
                result = trade_executor.emergency_stop()
                
                message = "🚨 **PARADA DE EMERGENCIA EJECUTADA** 🚨\n\n"
                if result:
                    message += "✅ Sistema detenido correctamente\n"
                    message += "✅ Posiciones cerradas exitosamente\n"
                else:
                    message += "⚠️ Problema durante la parada\n"
                    message += "📞 Verificar manualmente\n"
                
                # Notificar a ambos usuarios
                if self.is_dannis(user_id):
                    message += "\n💕 Santiago será notificado inmediatamente"
                else:
                    message += "\n📱 Dannis será notificada sobre la parada"
                    
                await update.message.reply_text(message, parse_mode='Markdown')
                
            except Exception as e:
                error_msg = "❌ Error ejecutando parada de emergencia"
                if self.is_santiago(user_id):
                    error_msg += f"\nDetalle técnico: {str(e)}"
                await update.message.reply_text(error_msg)
        else:
            await update.message.reply_text("🔒 Comando solo para usuarios autorizados")

    def _increment_message_count(self, user_id: int):
        """Incrementar contador de mensajes"""
        if user_id not in self.message_count:
            self.message_count[user_id] = 0
        self.message_count[user_id] += 1

    def _should_send_love_message(self, user_id: int) -> bool:
        """Determinar si enviar mensaje de amor (cada 5 comandos)"""
        count = self.message_count.get(user_id, 0)
        return count % 5 == 0 and count > 0

    async def run(self):
        """Ejecutar el bot"""
        try:
            application = Application.builder().token(self.token).build()
            
            # Registrar handlers
            application.add_handler(CommandHandler("start", self.start))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("stats", self.stats_command))
            application.add_handler(CommandHandler("trades", self.trades_command))
            application.add_handler(CommandHandler("profit", self.profit_command))
            application.add_handler(CommandHandler("health", self.health_command))
            application.add_handler(CommandHandler("training", self.training_command))
            application.add_handler(CommandHandler("love", self.love_command))
            application.add_handler(CommandHandler("emergency", self.emergency_command))
            
            logger.info("💕 Dannis Bot iniciado y listo para servir con amor")
            
            # Iniciar bot
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            # Mantener corriendo
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error ejecutando Dannis Bot: {e}")

if __name__ == "__main__":
    bot = DannisBot()
    asyncio.run(bot.run())