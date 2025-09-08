import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración del bot
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DANNIS_USER_ID = int(os.getenv('DANNIS_USER_ID', '0'))
SANTIAGO_USER_ID = int(os.getenv('SANTIAGO_USER_ID', '0'))

# Configuración de alertas
ALERT_SETTINGS = {
    'profit_threshold': 50.0,  # Alert si profit > $50
    'loss_threshold': -20.0,   # Alert si loss < -$25
    'win_streak_alert': 5,     # Alert después de 5 wins consecutivos
    'system_downtime_minutes': 10  # Alert si sistema down > 10 min
}

# Configuración de frases
LOVE_MESSAGE_FREQUENCY = 5  # Cada 5 comandos una frase de amor