#!/usr/bin/env python3
"""
🤖💕 Dannis Bot - Ejecutor principal
Creado con amor infinito para Dannis por Santiago Ossa
"""

import sys
import os
import asyncio
import logging
from dotenv import load_dotenv

# ⚡ CARGAR VARIABLES DE ENTORNO ANTES QUE CUALQUIER COSA
load_dotenv()

# Agregar path del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from telegram_bot.dannis_bot import DannisBot

async def main():
    """Ejecutar Dannis Bot"""
    print("💕 Iniciando Dannis Bot...")
    print("🌟 Creado especialmente para Dannis con amor de Santiago Ossa")
    
    # Verificar que el token está disponible
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    dannis_id = os.getenv('DANNIS_USER_ID')
    santiago_id = os.getenv('SANTIAGO_USER_ID')
    
    if not token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN no encontrado en .env")
        return
    
    print(f"✅ Token cargado: {token[:10]}...")
    print(f"✅ Dannis ID: {dannis_id}")
    print(f"✅ Santiago ID: {santiago_id}")
    
    bot = DannisBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())