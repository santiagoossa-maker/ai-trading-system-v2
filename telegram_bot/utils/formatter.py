from datetime import datetime, timedelta

class MessageFormatter:
    """
    Formatear mensajes para el bot de Telegram
    """
    
    def format_stats(self, stats: dict, is_dannis: bool = False) -> str:
        """Formatear estadísticas de trading"""
        if 'error' in stats:
            return f"❌ Error obteniendo estadísticas: {stats['error']}"
        
        message = "📊 **ESTADÍSTICAS DE TRADING** 📊\n\n"
        
        # Stats básicas
        message += f"📈 **Trades del día:** {stats.get('daily_trades', 0)}\n"
        message += f"🎯 **Total trades:** {stats.get('total_trades', 0)}\n"
        message += f"✅ **Trades exitosos:** {stats.get('successful_orders', 0)}\n"
        message += f"❌ **Trades fallidos:** {stats.get('failed_orders', 0)}\n"
        message += f"📊 **Win Rate:** {stats.get('win_rate', 0):.1f}%\n\n"
        
        # Información de cuenta
        balance = stats.get('account_balance', 0)
        equity = stats.get('account_equity', 0)
        profit = stats.get('total_profit', 0)
        
        message += f"💰 **Balance:** ${balance:.2f}\n"
        message += f"💎 **Equity:** ${equity:.2f}\n"
        message += f"📈 **Profit Total:** ${profit:.2f}\n"
        
        if stats.get('margin_used', 0) > 0:
            message += f"⚖️ **Margen usado:** ${stats.get('margin_used', 0):.2f}\n"
        
        # Timestamp
        last_update = stats.get('last_update', datetime.now())
        message += f"\n🕐 **Actualizado:** {last_update.strftime('%H:%M:%S')}"
        
        # Mensaje especial para Dannis
        if is_dannis:
            if profit > 0:
                message += f"\n\n💕 ¡Santiago está feliz con las ganancias de hoy!"
            elif stats.get('win_rate', 0) > 70:
                message += f"\n\n🌟 ¡Excelente win rate! Santiago dice que eres su amuleto de la suerte"
        
        return message
    
    def format_trades(self, trades: dict, is_dannis: bool = False) -> str:
        """Formatear trades activos"""
        if 'error' in trades:
            return f"❌ Error obteniendo trades: {trades['error']}"
        
        positions = trades.get('positions', [])
        pending = trades.get('pending_orders', [])
        
        message = "🎯 **TRADES ACTIVOS** 🎯\n\n"
        
        if not positions and not pending:
            message += "😴 No hay posiciones activas en este momento\n\n"
            if is_dannis:
                message += "💕 El sistema está descansando, como Santiago descansa pensando en ti"
            return message
        
        # Posiciones abiertas
        if positions:
            message += f"📈 **POSICIONES ABIERTAS ({len(positions)}):**\n"
            for i, pos in enumerate(positions[:5], 1):  # Máximo 5 para no saturar
                symbol = pos.get('symbol', 'N/A')
                type_str = "🟢 COMPRA" if pos.get('type') == 0 else "🔴 VENTA"
                volume = pos.get('volume', 0)
                profit = pos.get('profit', 0)
                profit_emoji = "💰" if profit > 0 else "📉" if profit < 0 else "⚖️"
                
                message += f"{i}. {symbol}\n"
                message += f"   {type_str} | Vol: {volume} | {profit_emoji} ${profit:.2f}\n"
            
            if len(positions) > 5:
                message += f"   ... y {len(positions) - 5} más\n"
            message += "\n"
        
        # Órdenes pendientes
        if pending:
            message += f"⏳ **ÓRDENES PENDIENTES ({len(pending)}):**\n"
            for i, order in enumerate(pending[:3], 1):  # Máximo 3
                symbol = order.get('symbol', 'N/A')
                type_str = "📈 BUY LIMIT" if order.get('type') == 2 else "📉 SELL LIMIT"
                volume = order.get('volume', 0)
                price = order.get('price_open', 0)
                
                message += f"{i}. {symbol} | {type_str}\n"
                message += f"   Vol: {volume} | Precio: ${price:.4f}\n"
            
            if len(pending) > 3:
                message += f"   ... y {len(pending) - 3} más\n"
        
        # Timestamp
        last_update = trades.get('last_update', datetime.now())
        message += f"\n🕐 **Actualizado:** {last_update.strftime('%H:%M:%S')}"
        
        return message
    
    def format_profit(self, profit_data: dict, is_dannis: bool = False) -> str:
        """Formatear análisis de profit"""
        if 'error' in profit_data:
            return f"❌ Error obteniendo profit: {profit_data['error']}"
        
        message = "💰 **ANÁLISIS DE GANANCIAS** 💰\n\n"
        
        # Balances actuales
        balance = profit_data.get('current_balance', 0)
        equity = profit_data.get('current_equity', 0)
        unrealized = profit_data.get('unrealized_pnl', 0)
        
        message += f"💎 **Balance:** ${balance:.2f}\n"
        message += f"📊 **Equity:** ${equity:.2f}\n"
        if abs(unrealized) > 0.01:
            unrealized_emoji = "📈" if unrealized > 0 else "📉"
            message += f"{unrealized_emoji} **P&L No Realizado:** ${unrealized:.2f}\n"
        message += "\n"
        
        # Profit por períodos
        daily = profit_data.get('daily_profit', 0)
        weekly = profit_data.get('weekly_profit', 0)
        monthly = profit_data.get('monthly_profit', 0)
        
        message += "📅 **GANANCIAS POR PERÍODO:**\n"
        daily_emoji = "💚" if daily > 0 else "❤️" if daily < 0 else "💛"
        message += f"{daily_emoji} **Hoy:** ${daily:.2f}\n"
        
        weekly_emoji = "💚" if weekly > 0 else "❤️" if weekly < 0 else "💛"
        message += f"{weekly_emoji} **Esta semana:** ${weekly:.2f}\n"
        
        monthly_emoji = "💚" if monthly > 0 else "❤️" if monthly < 0 else "💛"
        message += f"{monthly_emoji} **Este mes:** ${monthly:.2f}\n\n"
        
        # Métricas adicionales
        win_rate = profit_data.get('win_rate', 0)
        profit_factor = profit_data.get('profit_factor', 0)
        
        message += f"🎯 **Win Rate:** {win_rate:.1f}%\n"
        message += f"⚖️ **Profit Factor:** {profit_factor:.2f}\n"
        
        # Timestamp
        last_update = profit_data.get('last_update', datetime.now())
        message += f"\n🕐 **Actualizado:** {last_update.strftime('%H:%M:%S')}"
        
        return message
    
    def format_health(self, health_data: dict, is_dannis: bool = False) -> str:
        """Formatear estado del sistema"""
        if 'error' in health_data:
            return f"❌ Error obteniendo estado: {health_data['error']}"
        
        status = health_data.get('overall_status', 'unknown')
        status_emoji = {
            'healthy': '💚',
            'warning': '⚠️',
            'error': '🔴',
            'unknown': '❓'
        }.get(status, '❓')
        
        message = f"{status_emoji} **ESTADO DEL SISTEMA** {status_emoji}\n\n"
        
        # Estado general
        message += f"🖥️ **Estado general:** {status.upper()}\n"
        
        # Componentes
        mt5_status = "✅ Conectado" if health_data.get('mt5_connected') else "❌ Desconectado"
        message += f"📡 **MT5:** {mt5_status}\n"
        
        trading_status = "✅ Activo" if health_data.get('trading_enabled') else "⏸️ Pausado"
        message += f"🎯 **Trading:** {trading_status}\n"
        
        # Información adicional
        uptime = health_data.get('system_uptime', 'N/A')
        message += f"⏰ **Uptime:** {uptime}\n"
        
        memory = health_data.get('memory_usage', 0)
        if memory > 0:
            memory_emoji = "🟢" if memory < 70 else "🟡" if memory < 85 else "🔴"
            message += f"{memory_emoji} **Memoria:** {memory:.1f}%\n"
        
        last_trade = health_data.get('last_trade_time')
        if last_trade:
            time_diff = datetime.now() - last_trade
            if time_diff.total_seconds() < 3600:  # Menos de 1 hora
                message += f"🕐 **Último trade:** {time_diff.total_seconds()/60:.0f}m ago\n"
            else:
                message += f"🕐 **Último trade:** {last_trade.strftime('%H:%M')}\n"
        
        # Timestamp
        last_update = health_data.get('last_update', datetime.now())
        message += f"\n🕐 **Actualizado:** {last_update.strftime('%H:%M:%S')}"
        
        return message
    
    def format_training(self, training_data: dict, is_dannis: bool = False) -> str:
        """Formatear estado del entrenamiento"""
        if 'error' in training_data:
            return f"❌ Error obteniendo entrenamiento: {training_data['error']}"
        
        message = "🤖 **ESTADO DEL ENTRENAMIENTO IA** 🤖\n\n"
        
        # Estado actual
        is_training = training_data.get('is_training', False)
        if is_training:
            message += "🔄 **Estado:** Entrenando activamente\n"
        else:
            message += "😴 **Estado:** En reposo\n"
        
        # Información de entrenamientos
        last_training = training_data.get('last_training')
        next_training = training_data.get('next_training')
        
        if last_training:
            time_diff = datetime.now() - last_training
            if time_diff.total_seconds() < 3600:
                message += f"📚 **Último entrenamiento:** {time_diff.total_seconds()/60:.0f}m ago\n"
            else:
                message += f"📚 **Último entrenamiento:** {time_diff.total_seconds()/3600:.1f}h ago\n"
        
        if next_training:
            time_diff = next_training - datetime.now()
            if time_diff.total_seconds() > 0:
                if time_diff.total_seconds() < 3600:
                    message += f"⏰ **Próximo entrenamiento:** en {time_diff.total_seconds()/60:.0f}m\n"
                else:
                    message += f"⏰ **Próximo entrenamiento:** en {time_diff.total_seconds()/3600:.1f}h\n"
        
        # Accuracy de modelos
        accuracy = training_data.get('model_accuracy', {})
        if accuracy:
            message += "\n🎯 **ACCURACY DE MODELOS:**\n"
            for model, acc in accuracy.items():
                acc_emoji = "🟢" if acc > 0.7 else "🟡" if acc > 0.6 else "🔴"
                model_name = model.replace('_', ' ').title()
                message += f"{acc_emoji} **{model_name}:** {acc:.1%}\n"
        
        # Información adicional
        data_size = training_data.get('training_data_size', 0)
        improvement = training_data.get('improvement_since_last', 0)
        
        if data_size > 0:
            message += f"\n📊 **Datos de entrenamiento:** {data_size:,} registros\n"
        
        if improvement != 0:
            improvement_emoji = "📈" if improvement > 0 else "📉"
            message += f"{improvement_emoji} **Mejora desde último:** {improvement:+.1%}\n"
        
        # Timestamp
        last_update = training_data.get('last_update', datetime.now())
        message += f"\n🕐 **Actualizado:** {last_update.strftime('%H:%M:%S')}"
        
        return message