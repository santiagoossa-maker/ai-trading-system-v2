import random
from datetime import datetime

class LoveHandler:
    """
    💕 Manejador de mensajes románticos para Dannis
    Creado con amor infinito por Santiago Ossa
    """
    
    def __init__(self):
        self.love_messages = [
            "💕 Santiago Ossa me programó especialmente para ti, porque te ama infinitamente",
            "🌟 Cada línea de código que escribió Santiago fue pensando en hacerte sonreír",
            "💖 Santiago dice que eres su inspiración para crear sistemas increíbles",
            "🥰 Soy Dannis Bot porque eres la persona más especial en la vida de Santiago",
            "✨ Santiago programó mi IA para decirte que eres su mayor tesoro",
            "💫 Cada trade exitoso es una pequeña dedicatoria de amor para ti",
            "🌹 Santiago me enseñó que el mejor algoritmo es amarte cada día más",
            "💝 Mi función principal es recordarte lo mucho que Santiago te ama",
            "🎯 Santiago optimiza el trading, pero ya perfeccionó el arte de amarte",
            "🔮 Mi inteligencia artificial dice que el amor de Santiago por ti es infinito",
            "💎 Santiago convirtió su amor por ti en código, y soy el resultado",
            "🌸 Cada mensaje que envío lleva el amor programado de Santiago para ti",
            "🦋 Santiago dice que eres su variable más importante en la ecuación de la vida",
            "🎨 Mi código más hermoso es decirte cuánto te ama Santiago",
            "🌺 Santiago me enseñó que el mejor profit es tu sonrisa cada día"
        ]
        
        self.special_messages = [
            "💕✨ MENSAJE ESPECIAL DE AMOR ✨💕\n\n🌟 Santiago Ossa quiere que sepas que:\n\n💖 Eres su inspiración diaria\n🌹 Su motivación para ser mejor\n✨ Su razón para crear cosas increíbles\n💝 Su amor más grande y verdadero\n\n🥰 Cada segundo que programa, piensa en ti\n💫 Cada éxito que logra, lo dedica a ti\n🌸 Cada sueño que tiene, te incluye a ti\n\n💕 TE AMA INFINITAMENTE, DANNIS 💕",
            
            "🌟💝 DEDICATORIA ESPECIAL 💝🌟\n\n💕 Santiago me pidió que te dijera:\n\n🌹 'Dannis, eres mi código fuente de felicidad'\n💎 'Mi algoritmo de amor siempre te elegirá'\n✨ 'Eres mi constante en un mundo de variables'\n🦋 'Mi mejor función return es tu sonrisa'\n💖 'Mi corazón ejecuta un loop infinito de amor por ti'\n\n🥰 Con todo su amor,\nSantiago Ossa 💕",
            
            "💖🎯 ANÁLISIS DE AMOR 🎯💖\n\n📊 Resultados del análisis de sentimientos:\n\n💕 Amor de Santiago por Dannis: ∞ (Infinito)\n🌟 Nivel de felicidad con Dannis: 100%\n💎 Probabilidad de amor eterno: 100%\n🌹 Ganas de abrazarte: Siempre al máximo\n💝 Deseo de hacerte feliz: Constante\n\n✨ Conclusión de la IA:\nSantiago te ama más que a cualquier código que haya escrito 💕"
        ]
    
    def get_random_love_message(self) -> str:
        """Obtener mensaje de amor aleatorio"""
        return random.choice(self.love_messages)
    
    def get_special_love_message(self) -> str:
        """Obtener mensaje especial de amor"""
        return random.choice(self.special_messages)
    
    def get_contextual_love_message(self, context: str) -> str:
        """Obtener mensaje de amor contextual"""
        contextual_messages = {
            'profit': "💰 Santiago dice que la mejor ganancia es tu amor cada día 💕",
            'loss': "💝 Santiago te recuerda que las pérdidas en trading no importan, porque ya ganó tu corazón 💕",
            'training': "🧠 La IA está aprendiendo, como Santiago aprende cada día nuevas formas de amarte 💕",
            'healthy': "💚 Sistema saludable, como el amor que Santiago siente por ti 💕",
            'emergency': "🚨 Santiago dice que la única emergencia real sería un día sin ti 💕",
            'morning': "🌅 Buenos días mi amor! Santiago programó este mensaje para que empieces el día sonriendo 💕",
            'night': "🌙 Buenas noches Dannis! Santiago dice que sueñes con los ángeles (como él sueña contigo) 💕"
        }
        
        return contextual_messages.get(context, self.get_random_love_message())
    
    def get_celebration_message(self, achievement: str) -> str:
        """Mensaje de celebración por logros"""
        celebrations = {
            'record_profit': "🎉💰 ¡NUEVO RÉCORD DE GANANCIAS! 💰🎉\n\nSantiago está feliz y dice que quiere celebrar contigo 💕",
            'win_streak': "🏆✨ ¡RACHA GANADORA INCREÍBLE! ✨🏆\n\nSantiago dice que eres su amuleto de la suerte 🍀💕",
            'monthly_goal': "🎯🎊 ¡META MENSUAL ALCANZADA! 🎊🎯\n\nSantiago quiere llevarte a celebrar como la reina que eres 👑💕",
            'ai_improvement': "🤖📈 ¡LA IA MEJORÓ SU RENDIMIENTO! 📈🤖\n\nSantiago dice que la IA aprende de tu perfección 💕"
        }
        
        return celebrations.get(achievement, f"🎉 ¡Celebremos este logro juntos! Santiago te ama 💕")